from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import threading
import time
from base64 import b64encode
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from hashlib import sha1
from pathlib import Path
from typing import Any, Protocol

from arc_agent.v2_config import ResponsesConfig
from arc_agent.v2_models import ResponseSnapshot, ResponseUsage
from arc_agent.v2_openai import (
    INSTRUCTIONS,
    PROGRAM_SCHEMA,
    ConfigurationError,
    QuotaExhausted,
    ResponseExpired,
    TransientAPIError,
)

_RESPONSE_PREFIX = "codex-app"
_ALLOWED_TURN_ITEMS = {
    "agentMessage",
    "contextCompaction",
    "plan",
    "reasoning",
    "userMessage",
}
_MAC_CHATGPT_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
_SUBSCRIPTION_INSTRUCTIONS = """Operate as a non-agentic synthesis model.
Do not call tools, inspect files, browse, use MCP, delegate, or execute commands. The complete
problem and verifier evidence are in the user message. Return only the JSON object constrained by
the supplied output schema. Never mention or infer evaluation labels that are not in the message.
"""
_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_DAEMON_START_LOCK = threading.Lock()


class _UnixWebSocket:
    """Minimal RFC 6455 text client for Codex App Server's Unix listener."""

    def __init__(self, connection: socket.socket) -> None:
        self._connection = connection
        self._send_lock = threading.Lock()
        self._closed = False

    @classmethod
    def connect(cls, path: Path, timeout: float) -> _UnixWebSocket:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(timeout)
        try:
            connection.connect(str(path))
            nonce = b64encode(os.urandom(16)).decode("ascii")
            request = (
                "GET / HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {nonce}\r\n"
                "Sec-WebSocket-Version: 13\r\n\r\n"
            )
            connection.sendall(request.encode("ascii"))
            response = cls._read_headers(connection)
            lines = response.decode("latin-1").split("\r\n")
            if not lines or " 101 " not in f" {lines[0]} ":
                raise OSError(f"WebSocket upgrade failed: {lines[0] if lines else response!r}")
            headers = {
                name.strip().lower(): value.strip()
                for line in lines[1:]
                if ":" in line
                for name, value in [line.split(":", 1)]
            }
            expected = b64encode(sha1((nonce + _WEBSOCKET_GUID).encode()).digest()).decode()
            if headers.get("sec-websocket-accept") != expected:
                raise OSError("WebSocket upgrade returned an invalid accept key")
            connection.settimeout(None)
            return cls(connection)
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _read_headers(connection: socket.socket) -> bytes:
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = connection.recv(4_096)
            if not chunk:
                raise OSError("WebSocket connection closed during upgrade")
            response.extend(chunk)
            if len(response) > 65_536:
                raise OSError("WebSocket upgrade headers were too large")
        header, remainder = bytes(response).split(b"\r\n\r\n", 1)
        if remainder:
            raise OSError("unexpected WebSocket data during upgrade")
        return header

    @staticmethod
    def _read_exact(connection: socket.socket, length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            chunk = connection.recv(length - len(chunks))
            if not chunk:
                raise EOFError("WebSocket connection closed")
            chunks.extend(chunk)
        return bytes(chunks)

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if self._closed:
            raise OSError("WebSocket connection is closed")
        first = 0x80 | opcode
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", first, 0x80 | length)
        elif length <= 0xFFFF:
            header = struct.pack("!BBH", first, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", first, 0x80 | 127, length)
        mask = os.urandom(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        with self._send_lock:
            self._connection.sendall(header + mask + masked)

    def send_text(self, value: str) -> None:
        self._send_frame(0x1, value.encode("utf-8"))

    def receive_text(self) -> str:
        message = bytearray()
        message_opcode: int | None = None
        while True:
            head = self._read_exact(self._connection, 2)
            final = bool(head[0] & 0x80)
            opcode = head[0] & 0x0F
            masked = bool(head[1] & 0x80)
            length = head[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(self._connection, 2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(self._connection, 8))[0]
            if length > 64 * 1024 * 1024:
                raise OSError("Codex App Server WebSocket frame exceeds 64 MiB")
            mask = self._read_exact(self._connection, 4) if masked else b""
            payload = self._read_exact(self._connection, length)
            if mask:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 0x8:
                self._closed = True
                raise EOFError("Codex App Server closed the WebSocket")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in {0x1, 0x2}:
                if message_opcode is not None:
                    raise OSError("unexpected WebSocket data frame")
                message_opcode = opcode
            elif opcode != 0x0 or message_opcode is None:
                raise OSError(f"unsupported WebSocket opcode {opcode}")
            message.extend(payload)
            if final:
                if message_opcode != 0x1:
                    raise OSError("Codex App Server sent a non-text WebSocket message")
                return message.decode("utf-8")

    @property
    def is_alive(self) -> bool:
        return not self._closed

    def close(self) -> None:
        if self._closed:
            return
        with suppress(OSError):
            self._send_frame(0x8, b"")
        self._closed = True
        with suppress(OSError):
            self._connection.shutdown(socket.SHUT_RDWR)
        self._connection.close()


class CodexRPC(Protocol):
    latest_rate_limits: dict[str, Any] | None

    def request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]: ...

    def turn_usage(self, thread_id: str, turn_id: str, *, wait: float = 0.0) -> ResponseUsage: ...

    def close(self) -> None: ...


def subscription_home(workspace: str | Path) -> Path:
    return Path(workspace) / ".codex-subscription"


def resolve_codex_cli(configured: str) -> str:
    """Resolve Codex from PATH, an explicit path, or the macOS ChatGPT bundle."""
    discovered = shutil.which(configured)
    if discovered is not None:
        return discovered
    explicit = Path(configured).expanduser()
    if (
        ("/" in configured or "\\" in configured)
        and explicit.is_file()
        and os.access(explicit, os.X_OK)
    ):
        return str(explicit.resolve())
    if (
        configured == "codex"
        and _MAC_CHATGPT_CODEX.is_file()
        and os.access(_MAC_CHATGPT_CODEX, os.X_OK)
    ):
        return str(_MAC_CHATGPT_CODEX)
    raise ConfigurationError(
        f"Codex CLI executable {configured!r} was not found. Install it with the official "
        "installer or set openai.codex_cli to an executable path.",
        code="missing_codex_cli",
    )


def codex_isolation_options() -> list[str]:
    return [
        "-c",
        'forced_login_method="chatgpt"',
        "-c",
        "agents.enabled=false",
        "-c",
        "features.shell_tool=false",
        "-c",
        "features.unified_exec=false",
        "-c",
        "features.skill_mcp_dependency_install=false",
        "-c",
        "tools.view_image=false",
        "-c",
        "tools.web_search=false",
        "-c",
        'web_search="disabled"',
        "-c",
        "memories.generate_memories=false",
        "-c",
        'history.persistence="none"',
        "--disable",
        "apps",
        "--disable",
        "plugins",
    ]


def codex_auth_command(
    settings: ResponsesConfig, workspace: str | Path
) -> tuple[list[str], dict[str, str]]:
    binary = resolve_codex_cli(settings.codex_cli)
    home = subscription_home(workspace)
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    home.chmod(0o700)
    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(home.resolve())
    return [binary, "login"], environment


def _daemon_paths(home: Path) -> tuple[Path, Path, Path]:
    return (
        home / "app-server.sock",
        home / "app-server.pid",
        home / "app-server.log",
    )


def _owned_daemon_pid(home: Path) -> int | None:
    socket_path, pid_path, _ = _daemon_paths(home)
    try:
        pid = int(pid_path.read_text().strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    if pid <= 1 or not socket_path.exists():
        return None
    inspected = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        capture_output=True,
        check=False,
    )
    command = inspected.stdout.strip()
    if inspected.returncode or "app-server" not in command or str(socket_path) not in command:
        return None
    return pid


def _clear_stale_daemon_files(home: Path) -> None:
    socket_path, pid_path, _ = _daemon_paths(home)
    with suppress(FileNotFoundError):
        socket_path.unlink()
    with suppress(FileNotFoundError):
        pid_path.unlink()


def restart_subscription_daemon(settings: ResponsesConfig, workspace: str | Path) -> None:
    if not settings.subscription_daemon:
        return
    home = subscription_home(workspace)
    pid = _owned_daemon_pid(home)
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        stopped = False
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                stopped = True
                break
            time.sleep(0.05)
        if not stopped and _owned_daemon_pid(home) == pid:
            raise ConfigurationError(
                "ChatGPT login succeeded but the workspace Codex daemon did not stop",
                code="codex_daemon_restart",
            )
    _clear_stale_daemon_files(home)


def _rpc_error(error: Any) -> Exception:
    raw = error if isinstance(error, dict) else {"message": str(error)}
    message = str(raw.get("message") or raw)
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    info = data.get("codexErrorInfo") or raw.get("codexErrorInfo")
    marker = f"{info} {message}".lower().replace("_", "")
    if any(
        value in marker
        for value in (
            "usagelimitexceeded",
            "creditsdepleted",
            "usagelimitreached",
            "insufficientquota",
        )
    ):
        return QuotaExhausted(message, code=str(info or "usage_limit_exceeded"))
    if any(value in marker for value in ("unauthorized", "badrequest", "authentication")):
        return ConfigurationError(message, code=str(info or "codex_configuration"))
    return TransientAPIError(message, code=str(info or "codex_rpc"))


class CodexAppServerRPC:
    def __init__(self, settings: ResponsesConfig, workspace: Path) -> None:
        self.settings = settings
        self.workspace = workspace
        self.home = subscription_home(workspace)
        self.sandbox = self.home / "model-sandbox"
        self.latest_rate_limits: dict[str, Any] | None = None
        self._condition = threading.Condition()
        self._responses: dict[int, dict[str, Any]] = {}
        self._turn_usages: dict[tuple[str, str], ResponseUsage] = {}
        self._stderr: deque[str] = deque(maxlen=20)
        self._next_id = 1
        self._closed_error: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._websocket: _UnixWebSocket | None = None
        self._start_transport()
        reader = self._read_websocket if self._websocket is not None else self._read_stdout
        self._stdout_thread = threading.Thread(target=reader, daemon=True)
        self._stdout_thread.start()
        if self._process is not None:
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._stderr_thread.start()
        self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "arc-agent-v2",
                    "title": "ARC Agent V2 Program Synthesis",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": False},
            },
        )
        self._send({"method": "initialized"})

    def _start_transport(self) -> None:
        binary = resolve_codex_cli(self.settings.codex_cli)
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.home.chmod(0o700)
        self.sandbox.mkdir(parents=True, exist_ok=True, mode=0o700)
        environment = os.environ.copy()
        environment["CODEX_HOME"] = str(self.home.resolve())
        options = codex_isolation_options()
        if self.settings.subscription_daemon:
            socket_path = self._ensure_workspace_daemon(binary, options, environment)
            deadline = time.monotonic() + self.settings.request_timeout_seconds
            while True:
                try:
                    self._websocket = _UnixWebSocket.connect(
                        socket_path, self.settings.request_timeout_seconds
                    )
                    return
                except OSError as exc:
                    if time.monotonic() >= deadline:
                        raise TransientAPIError(
                            f"failed to connect to workspace Codex App Server: {exc}"
                        ) from exc
                    time.sleep(0.05)
        command = [binary, "app-server", "--listen", "stdio://", *options]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=self.sandbox,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise ConfigurationError(f"failed to launch Codex App Server: {exc}") from exc

    def _ensure_workspace_daemon(
        self,
        binary: str,
        options: list[str],
        environment: dict[str, str],
    ) -> Path:
        with _DAEMON_START_LOCK:
            socket_path, pid_path, log_path = _daemon_paths(self.home)
            if _owned_daemon_pid(self.home) is not None:
                return socket_path
            _clear_stale_daemon_files(self.home)
            listen = f"unix://{socket_path.resolve()}"
            try:
                with log_path.open("a") as log:
                    daemon = subprocess.Popen(
                        [binary, "app-server", "--listen", listen, *options],
                        cwd=self.sandbox,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        start_new_session=True,
                        close_fds=True,
                    )
                pid_path.write_text(str(daemon.pid))
                pid_path.chmod(0o600)
            except OSError as exc:
                raise ConfigurationError(
                    f"failed to launch workspace Codex App Server: {exc}"
                ) from exc
            deadline = time.monotonic() + self.settings.request_timeout_seconds
            while time.monotonic() < deadline:
                if socket_path.exists():
                    return socket_path
                if daemon.poll() is not None:
                    detail = ""
                    with suppress(OSError, UnicodeError):
                        detail = log_path.read_text()[-2_000:]
                    _clear_stale_daemon_files(self.home)
                    raise ConfigurationError(
                        f"workspace Codex App Server exited during startup: {detail}",
                        code="codex_daemon_start",
                    )
                time.sleep(0.05)
            daemon.terminate()
            _clear_stale_daemon_files(self.home)
            raise TransientAPIError("workspace Codex App Server socket did not become ready")

    def _send(self, payload: dict[str, Any]) -> None:
        serialized = json.dumps(payload, separators=(",", ":"))
        if self._websocket is not None:
            try:
                self._websocket.send_text(serialized)
                return
            except OSError as exc:
                raise TransientAPIError(f"Codex App Server connection failed: {exc}") from exc
        process = self._process
        stream = process.stdin if process is not None else None
        if stream is None or process is None or process.poll() is not None:
            raise TransientAPIError("Codex App Server connection is not running")
        try:
            stream.write(serialized + "\n")
            stream.flush()
        except (BrokenPipeError, OSError) as exc:
            raise TransientAPIError(f"Codex App Server connection failed: {exc}") from exc

    def _read_stdout(self) -> None:
        process = self._process
        stream = process.stdout if process is not None else None
        if stream is None:
            return
        try:
            for line in stream:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with self._condition:
                    request_id = message.get("id")
                    if isinstance(request_id, int):
                        self._responses[request_id] = message
                    else:
                        self._record_notification(message)
                    self._condition.notify_all()
        finally:
            with self._condition:
                detail = "\n".join(self._stderr)
                self._closed_error = detail or "Codex App Server connection closed"
                self._condition.notify_all()

    def _read_websocket(self) -> None:
        connection = self._websocket
        if connection is None:
            return
        try:
            while True:
                try:
                    message = json.loads(connection.receive_text())
                except json.JSONDecodeError:
                    continue
                with self._condition:
                    request_id = message.get("id")
                    if isinstance(request_id, int):
                        self._responses[request_id] = message
                    else:
                        self._record_notification(message)
                    self._condition.notify_all()
        except (EOFError, OSError, UnicodeError) as exc:
            self._stderr.append(str(exc))
        finally:
            with self._condition:
                detail = "\n".join(self._stderr)
                self._closed_error = detail or "Codex App Server connection closed"
                self._condition.notify_all()

    def _read_stderr(self) -> None:
        process = self._process
        stream = process.stderr if process is not None else None
        if stream is None:
            return
        for line in stream:
            self._stderr.append(line.rstrip())

    def _record_notification(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params")
        if not isinstance(params, dict):
            return
        if method == "thread/tokenUsage/updated":
            token_usage = params.get("tokenUsage")
            last = token_usage.get("last") if isinstance(token_usage, dict) else None
            if isinstance(last, dict):
                key = (str(params.get("threadId") or ""), str(params.get("turnId") or ""))
                self._turn_usages[key] = ResponseUsage(
                    input_tokens=int(last.get("inputTokens") or 0),
                    cached_input_tokens=int(last.get("cachedInputTokens") or 0),
                    cache_write_input_tokens=int(last.get("cacheWriteInputTokens") or 0),
                    output_tokens=int(last.get("outputTokens") or 0),
                    reasoning_tokens=int(last.get("reasoningOutputTokens") or 0),
                )
        elif method == "account/rateLimits/updated":
            limits = params.get("rateLimits")
            if isinstance(limits, dict):
                self.latest_rate_limits = limits

    def request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        with self._condition:
            request_id = self._next_id
            self._next_id += 1
        self._send({"id": request_id, "method": method, "params": params})
        deadline = time.monotonic() + (timeout or self.settings.request_timeout_seconds)
        with self._condition:
            while request_id not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransientAPIError(f"Codex App Server {method} timed out")
                if self._closed_error is not None:
                    raise TransientAPIError(self._closed_error)
                self._condition.wait(timeout=remaining)
            response = self._responses.pop(request_id)
        if "error" in response:
            raise _rpc_error(response["error"])
        result = response.get("result")
        if not isinstance(result, dict):
            return {}
        return result

    def turn_usage(self, thread_id: str, turn_id: str, *, wait: float = 0.0) -> ResponseUsage:
        key = (thread_id, turn_id)
        deadline = time.monotonic() + max(0.0, wait)
        with self._condition:
            while key not in self._turn_usages and time.monotonic() < deadline:
                self._condition.wait(timeout=deadline - time.monotonic())
            return self._turn_usages.get(key, ResponseUsage()).model_copy(deep=True)

    @property
    def is_alive(self) -> bool:
        if self._websocket is not None:
            return self._websocket.is_alive and self._closed_error is None
        return (
            self._process is not None
            and self._process.poll() is None
            and self._closed_error is None
        )

    def close(self) -> None:
        if self._websocket is not None:
            self._websocket.close()
            self._websocket = None
            return
        process = self._process
        if process is None:
            return
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


def _response_id(thread_id: str, turn_id: str, request_key: str) -> str:
    return f"{_RESPONSE_PREFIX}:{thread_id}:{turn_id}:{request_key}"


def _parse_response_id(response_id: str) -> tuple[str, str, str]:
    parts = response_id.split(":", 3)
    if len(parts) != 4 or parts[0] != _RESPONSE_PREFIX or not all(parts[1:]):
        raise ResponseExpired("invalid Codex App Server response identifier")
    return parts[1], parts[2], parts[3]


def _usage_body(usage: ResponseUsage) -> dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {
            "cached_tokens": usage.cached_input_tokens,
            "cache_write_tokens": usage.cache_write_input_tokens,
        },
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
    }


def _snapshot(
    *,
    response_id: str,
    status: str,
    thread_id: str,
    turn_id: str,
    usage: ResponseUsage | None = None,
    text: str = "",
    error: dict[str, Any] | None = None,
    token_limit: int | None = None,
) -> ResponseSnapshot:
    measured = usage or ResponseUsage()
    output: list[dict[str, Any]] = []
    if text:
        output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
    body: dict[str, Any] = {
        "id": response_id,
        "status": status,
        "provider": "chatgpt_subscription",
        "codex_thread_id": thread_id,
        "codex_turn_id": turn_id,
        "output": output,
        "usage": _usage_body(measured),
    }
    if error is not None:
        body["error"] = error
    if token_limit is not None:
        body["requested_max_output_tokens"] = token_limit
        body["max_output_tokens_enforced"] = False
    return ResponseSnapshot(
        response_id=response_id,
        status=status,  # type: ignore[arg-type]
        body=body,
        usage=measured,
    )


def _turn_error(turn: dict[str, Any]) -> dict[str, Any]:
    raw = turn.get("error")
    if not isinstance(raw, dict):
        return {"code": "codex_turn_failed", "message": "Codex turn failed"}
    info = raw.get("codexErrorInfo")
    return {
        "code": str(info or "codex_turn_failed"),
        "message": str(raw.get("message") or "Codex turn failed"),
        "additional_details": raw.get("additionalDetails"),
    }


def _turn_text(turn: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in turn.get("items") or []:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type not in _ALLOWED_TURN_ITEMS:
            raise ConfigurationError(
                f"subscription synthesis attempted forbidden external item {item_type!r}",
                code="eval_isolation_violation",
            )
        # App Server emits contextCompaction as internal conversation-history
        # bookkeeping.  It carries no candidate text and performs no external
        # action, so retain the turn while intentionally ignoring the item.
        if item_type == "contextCompaction":
            continue
        if item_type == "agentMessage" and item.get("phase") in {None, "final_answer"}:
            chunks.append(str(item.get("text") or ""))
    return "".join(chunks)


def _find_turn_by_client_id(thread: dict[str, Any], client_id: str) -> dict[str, Any] | None:
    for turn in thread.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        for item in turn.get("items") or []:
            if (
                isinstance(item, dict)
                and item.get("type") == "userMessage"
                and item.get("clientId") == client_id
            ):
                return turn
    return None


class CodexSubscriptionClient:
    def __init__(
        self,
        *,
        settings: ResponsesConfig,
        workspace: Path,
        rpc: CodexRPC | None = None,
    ) -> None:
        self.settings = settings
        self.workspace = workspace
        self._rpc = rpc
        self._ready_models: set[str] = set()
        self._loaded_threads: set[str] = set()

    @property
    def rpc(self) -> CodexRPC:
        if self._rpc is not None and not getattr(self._rpc, "is_alive", True):
            with suppress(Exception):
                self._rpc.close()
            self._rpc = None
            self._ready_models.clear()
            self._loaded_threads.clear()
        if self._rpc is None:
            self._rpc = CodexAppServerRPC(self.settings, self.workspace)
        return self._rpc

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        rpc = self.rpc
        try:
            return rpc.request(method, params)
        except TransientAPIError:
            with suppress(Exception):
                rpc.close()
            if self._rpc is rpc:
                self._rpc = None
                self._ready_models.clear()
                self._loaded_threads.clear()
            raise

    def _ensure_ready(self, model: str) -> None:
        _ = self.rpc
        if model in self._ready_models:
            self._check_quota()
            return
        account_result = self._request("account/read", {"refreshToken": True})
        account = account_result.get("account")
        if not isinstance(account, dict) or account.get("type") != "chatgpt":
            command = f"uv run arc-agent v2-auth-login --workspace {str(self.workspace)!r}"
            raise ConfigurationError(
                "No ChatGPT subscription login is available for this V2 workspace. "
                f"Run {command} and retry.",
                code="missing_chatgpt_subscription",
            )
        models = self._request("model/list", {"includeHidden": True, "limit": 100})
        available = [row for row in models.get("data") or [] if isinstance(row, dict)]
        selected = next(
            (
                row
                for row in available
                if row.get("id") == model or row.get("model") == model
            ),
            None,
        )
        if selected is None:
            raise ConfigurationError(
                f"ChatGPT subscription does not expose {model}",
                code="unsupported_subscription_model",
            )
        efforts = {
            row.get("reasoningEffort")
            for row in selected.get("supportedReasoningEfforts") or []
            if isinstance(row, dict)
        }
        if self.settings.reasoning_effort not in efforts:
            raise ConfigurationError(
                f"{model} does not expose {self.settings.reasoning_effort} effort",
                code="unsupported_reasoning_effort",
            )
        limits = self._request("account/rateLimits/read", None)
        snapshots = limits.get("rateLimitsByLimitId")
        if isinstance(snapshots, dict):
            for value in snapshots.values():
                if isinstance(value, dict) and value.get("limitId") == "codex":
                    self.rpc.latest_rate_limits = value
                    break
        elif isinstance(limits.get("rateLimits"), dict):
            self.rpc.latest_rate_limits = limits["rateLimits"]
        self._ready_models.add(model)
        self._check_quota()

    def _check_quota(self) -> None:
        limits = self.rpc.latest_rate_limits
        if not isinstance(limits, dict) or not limits.get("rateLimitReachedType"):
            return
        primary = limits.get("primary") if isinstance(limits.get("primary"), dict) else {}
        reset = primary.get("resetsAt")
        suffix = f"; resets at Unix time {reset}" if reset else ""
        raise QuotaExhausted(
            f"ChatGPT subscription Codex quota is exhausted{suffix}",
            code=str(limits.get("rateLimitReachedType")),
        )

    def _thread_params(self, model: str) -> dict[str, Any]:
        return {
            "model": model,
            "cwd": str((subscription_home(self.workspace) / "model-sandbox").resolve()),
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": False,
            "baseInstructions": INSTRUCTIONS,
            "developerInstructions": _SUBSCRIPTION_INSTRUCTIONS,
        }

    def _resume_thread(self, thread_id: str, model: str) -> dict[str, Any]:
        result = self._request(
            "thread/resume", {"threadId": thread_id, **self._thread_params(model)}
        )
        self._loaded_threads.add(thread_id)
        thread = result.get("thread")
        if not isinstance(thread, dict):
            raise ResponseExpired(f"Codex thread {thread_id} could not be resumed")
        return thread

    def _read_thread(self, thread_id: str, model: str) -> dict[str, Any]:
        if thread_id not in self._loaded_threads:
            return self._resume_thread(thread_id, model)
        result = self._request("thread/read", {"threadId": thread_id, "includeTurns": True})
        thread = result.get("thread")
        if not isinstance(thread, dict):
            raise ResponseExpired(f"Codex thread {thread_id} was not found")
        return thread

    def _start_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        request_key: str,
        token_limit: int,
        model: str,
    ) -> ResponseSnapshot:
        result = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}],
                "effort": self.settings.reasoning_effort,
                "model": model,
                "clientUserMessageId": request_key,
                "outputSchema": PROGRAM_SCHEMA,
                "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
            },
        )
        turn = result.get("turn")
        if not isinstance(turn, dict) or not turn.get("id"):
            raise TransientAPIError("Codex turn/start did not return a turn id")
        turn_id = str(turn["id"])
        response_id = _response_id(thread_id, turn_id, request_key)
        return _snapshot(
            response_id=response_id,
            status="in_progress",
            thread_id=thread_id,
            turn_id=turn_id,
            token_limit=token_limit,
        )

    def create(
        self,
        *,
        prompt: str,
        task_id: str,
        phase: str,
        round_index: int,
        max_output_tokens: int,
        previous_response_id: str | None,
        request_key: str,
        model: str | None = None,
        checkpoint: Callable[[ResponseSnapshot], None] | None = None,
    ) -> ResponseSnapshot:
        del task_id, phase, round_index
        selected_model = model or self.settings.model
        self._ensure_ready(selected_model)
        if previous_response_id:
            thread_id, _, _ = _parse_response_id(previous_response_id)
            self._resume_thread(thread_id, selected_model)
        else:
            result = self._request("thread/start", self._thread_params(selected_model))
            thread = result.get("thread")
            if not isinstance(thread, dict) or not thread.get("id"):
                raise TransientAPIError("Codex thread/start did not return a thread id")
            thread_id = str(thread["id"])
            self._loaded_threads.add(thread_id)
        provisional_id = _response_id(thread_id, "pending", request_key)
        provisional = _snapshot(
            response_id=provisional_id,
            status="queued",
            thread_id=thread_id,
            turn_id="pending",
            token_limit=max_output_tokens,
        )
        if checkpoint is not None:
            checkpoint(provisional)
        return self._start_turn(
            thread_id=thread_id,
            prompt=prompt,
            request_key=request_key,
            token_limit=max_output_tokens,
            model=selected_model,
        )

    def retrieve(
        self,
        response_id: str,
        *,
        request: dict[str, Any] | None = None,
    ) -> ResponseSnapshot:
        selected_model = str((request or {}).get("model") or self.settings.model)
        self._ensure_ready(selected_model)
        thread_id, turn_id, request_key = _parse_response_id(response_id)
        thread = self._read_thread(thread_id, selected_model)
        if turn_id == "pending":
            existing = _find_turn_by_client_id(thread, request_key)
            if existing is None:
                if request is None:
                    raise TransientAPIError("prepared Codex turn requires its checkpoint request")
                return self._start_turn(
                    thread_id=thread_id,
                    prompt=str(request["prompt"]),
                    request_key=request_key,
                    token_limit=int(request["token_limit"]),
                    model=selected_model,
                )
            turn_id = str(existing["id"])
        turn = next(
            (
                item
                for item in thread.get("turns") or []
                if isinstance(item, dict) and str(item.get("id")) == turn_id
            ),
            None,
        )
        if turn is None:
            raise ResponseExpired(f"Codex turn {turn_id} was not found")
        canonical_id = _response_id(thread_id, turn_id, request_key)
        status = str(turn.get("status") or "")
        if status == "inProgress":
            return _snapshot(
                response_id=canonical_id,
                status="in_progress",
                thread_id=thread_id,
                turn_id=turn_id,
            )
        usage = self.rpc.turn_usage(thread_id, turn_id, wait=1.0)
        if status == "completed":
            return _snapshot(
                response_id=canonical_id,
                status="completed",
                thread_id=thread_id,
                turn_id=turn_id,
                usage=usage,
                text=_turn_text(turn),
            )
        if status == "failed":
            return _snapshot(
                response_id=canonical_id,
                status="failed",
                thread_id=thread_id,
                turn_id=turn_id,
                usage=usage,
                error=_turn_error(turn),
            )
        if status == "interrupted":
            return _snapshot(
                response_id=canonical_id,
                status="cancelled",
                thread_id=thread_id,
                turn_id=turn_id,
                usage=usage,
            )
        raise TransientAPIError(f"unknown Codex turn status {status!r}")

    def cancel(self, response_id: str) -> None:
        thread_id, turn_id, _ = _parse_response_id(response_id)
        if turn_id == "pending":
            return
        self._request(
            "turn/interrupt",
            {"threadId": thread_id, "turnId": turn_id},
        )

    def close(self) -> None:
        if self._rpc is not None:
            self._rpc.close()
            self._rpc = None
        self._ready_models.clear()
        self._loaded_threads.clear()
