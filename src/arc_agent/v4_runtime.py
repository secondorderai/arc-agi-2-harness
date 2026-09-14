"""Own the server; optional warning tolerance never disables critical/swap guards."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import json
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from arc_agent.v4_config import V4Config
from arc_agent.v4_state import atomic_json


def external_network_denied() -> bool:
    # Literal IP avoids mistaking a DNS failure for enforced network isolation.
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=1):
            return False
    except OSError as error:
        return error.errno in {errno.EPERM, errno.EACCES}


def resident_bytes(pid: int) -> int | None:
    # macOS sys/proc_info.h: PROC_PIDTASKINFO=4; six uint64 and twelve int32 fields.
    # /bin/ps is a privileged executable and cannot be launched inside Seatbelt.
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        function = library.proc_pidinfo
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        function.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(96)
        if function(pid, 4, 0, buffer, len(buffer)) == len(buffer):
            return struct.unpack_from("=Q", buffer.raw, 8)[0]
    except OSError:
        pass
    return None


def memory_sample(pid: int | None = None) -> dict:
    result = {
        "time": time.time(),
        "swap_used_bytes": None,
        "pressure_level": None,
        "server_rss_bytes": None,
    }
    if platform.system() != "Darwin":
        return result
    result["server_rss_bytes"] = resident_bytes(pid or os.getpid())
    for key, command in [
        ("swap", ["/usr/sbin/sysctl", "-n", "vm.swapusage"]),
        ("pressure", ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]),
    ]:
        try:
            process = subprocess.run(command, text=True, capture_output=True, timeout=2)
            if process.returncode:
                continue
            if key == "swap":
                match = re.search(r"used\s*=\s*([\d.]+)([MG])", process.stdout)
                if match:
                    result["swap_used_bytes"] = float(match[1]) * (
                        1024 ** (2 if match[2] == "M" else 3)
                    )
            elif key == "pressure":
                result["pressure_level"] = int(process.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    return result


def pressure_allows_start(level: int | None, config: V4Config) -> bool:
    return level == 1 or (level == 2 and config.warning_pressure_policy == "record")


def prompt_cache_arguments(config: V4Config) -> list[str]:
    return [] if config.prompt_cache_mib is None else ["--cache-ram", "0"]


class LocalServer:
    def __init__(
        self,
        config: V4Config,
        manifest: dict,
        run_dir: Path,
        *,
        deadline: float | None = None,
        swap_baseline_bytes: float | None = None,
    ):
        self.config, self.manifest, self.run_dir = config, manifest, run_dir
        self.process = None
        self.stop = threading.Event()
        self.guard_error = None
        self.samples = []
        self.thread = None
        self.log = None
        self.baseline = None
        self.offline_verified = False
        self.log_start_offset = 0
        self.runtime_lock = None
        self.deadline = deadline
        self.swap_baseline_bytes = swap_baseline_bytes

    def __enter__(self):
        try:
            return self._start()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _claim_runtime(self):
        self.config.manifest.parent.mkdir(parents=True, exist_ok=True)
        handle = (self.config.manifest.parent / "active-server.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise ValueError("another V4 process already owns the local Nanbeige runtime") from None
        self.runtime_lock = handle

    def _start(self):
        if platform.system() != "Darwin":
            raise ValueError("Phase 0 must run on the local Mac before a GPU port")
        if not external_network_denied():
            raise ValueError(
                "launch through scripts/v4-loopback.sb; external network must be denied"
            )
        self.offline_verified = True
        self._claim_runtime()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.run_dir).free < self.config.min_free_disk_gib * 1024**3:
            raise ValueError("free disk below 10 GiB safety floor")
        self.baseline = memory_sample()
        self.baseline["warning_pressure_policy"] = self.config.warning_pressure_policy
        atomic_json(self.run_dir / "memory-baseline.json", self.baseline)
        if any(
            self.baseline[k] is None
            for k in ("pressure_level", "swap_used_bytes", "server_rss_bytes")
        ):
            raise ValueError("memory telemetry unavailable; cannot certify local safety")
        if not pressure_allows_start(self.baseline["pressure_level"], self.config):
            self.guard_error = "memory preflight: pressure exceeds the configured safety policy"
            raise ValueError(self.guard_error)
        if self.swap_baseline_bytes is not None:
            self.baseline["observed_swap_used_bytes"] = self.baseline["swap_used_bytes"]
            if self.baseline["swap_used_bytes"] - self.swap_baseline_bytes > (
                self.config.max_swap_growth_gib * 1024**3
            ):
                self.guard_error = "memory safety stop: cumulative swap growth across contexts"
                raise ValueError(self.guard_error)
            self.baseline["swap_used_bytes"] = self.swap_baseline_bytes
            atomic_json(self.run_dir / "memory-baseline.json", self.baseline)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("task deadline reached before context startup")
        log_path = self.run_dir / "server.log"
        self.log_start_offset = log_path.stat().st_size if log_path.exists() else 0
        self.log = log_path.open("a")
        port = urlsplit(self.config.base_url).port or 80
        with socket.socket() as probe:
            probe.settimeout(0.3)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(
                    "loopback port is already occupied; refusing a second model process"
                )
        command = [
            self.manifest["files"]["server"]["path"],
            "-m",
            self.manifest["files"]["model"]["path"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(self.config.context_tokens),
            "-np",
            "1",
            "-ngl",
            "99",
            "-t",
            "6",
            "-b",
            "128",
            "-ub",
            "64",
            "--flash-attn",
            "on",
            "--cache-type-k",
            self.config.cache_type,
            "--cache-type-v",
            self.config.cache_type,
            "--no-context-shift",
            "--cors-origins",
            "http://127.0.0.1",
            "-lv",
            "4",
        ]
        command.extend(prompt_cache_arguments(self.config))
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        atomic_json(
            self.run_dir / "server-process.json",
            {"pid": self.process.pid, "started_at": time.time(), "command": command},
        )
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        try:
            with httpx.Client(trust_env=False, timeout=2) as client:
                until = min(time.monotonic() + 120, self.deadline or float("inf"))
                while time.monotonic() < until:
                    self.check()
                    try:
                        health = client.get(self.config.base_url + "/health")
                        if health.status_code == 200:
                            props = client.get(self.config.base_url + "/props").json()
                            expected = Path(self.manifest["files"]["model"]["path"]).resolve()
                            if Path(props.get("model_path", "")).resolve() != expected:
                                raise ValueError(
                                    "server is not using the verified Nanbeige artifact"
                                )
                            if props.get("total_slots") != 1:
                                raise ValueError("local pilot requires exactly one server slot")
                            return self
                    except httpx.HTTPError:
                        pass
                    self.stop.wait(0.25)
                raise TimeoutError("Nanbeige server did not become healthy in 120 seconds")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        if self.guard_error:
            raise RuntimeError(self.guard_error)
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(
                f"Nanbeige server exited: {self.process.returncode}; inspect server.log"
            )

    def _watch(self):
        pressure_streak = 0
        with (self.run_dir / "memory.jsonl").open("a") as stream:
            while not self.stop.is_set():
                sample = memory_sample(self.process.pid)
                self.samples.append(sample)
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                level = sample["pressure_level"]
                if any(
                    sample.get(k) is None
                    for k in ("pressure_level", "swap_used_bytes", "server_rss_bytes")
                ):
                    self.guard_error = "memory safety stop: telemetry lost"
                    self._stop_owned_process()
                    return
                if level not in {1, 2}:
                    self.guard_error = "memory safety stop: critical or unrecognized pressure level"
                    self._stop_owned_process()
                    return
                pressure_streak = (
                    pressure_streak + 1
                    if level == 2 and self.config.warning_pressure_policy == "stop"
                    else 0
                )
                growth = (sample["swap_used_bytes"] or 0) - self.baseline["swap_used_bytes"]
                if pressure_streak >= 3 or growth > self.config.max_swap_growth_gib * 1024**3:
                    self.guard_error = (
                        "memory safety stop: sustained pressure or excessive swap growth"
                    )
                    self._stop_owned_process()
                    return
                if shutil.disk_usage(self.run_dir).free < self.config.min_free_disk_gib * 1024**3:
                    self.guard_error = "disk safety stop: less than 10 GiB free"
                    self._stop_owned_process()
                    return
                self.stop.wait(2)

    def _stop_owned_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=8)
        self._stop_owned_process()
        if self.log:
            self.log.close()
        if self.runtime_lock:
            fcntl.flock(self.runtime_lock, fcntl.LOCK_UN)
            self.runtime_lock.close()
            self.runtime_lock = None
