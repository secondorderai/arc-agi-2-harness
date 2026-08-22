from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from arc_agent.config import load_config
from arc_agent.data import load_tasks, write_submission
from arc_agent.llm import OpenAICompatibleAdapter
from arc_agent.models import Submission, TaskRun
from arc_agent.skills import load_cards
from arc_agent.solver import ArcSolver


def find_challenges(root: Path = Path("/kaggle/input")) -> Path:
    names = (
        "arc-agi_test-challenges.json",
        "arc-agi_test_challenges.json",
        "test_challenges.json",
    )
    for name in names:
        matches = sorted(root.glob(f"**/{name}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not find an ARC test challenges file below {root}")


def _gpu_count() -> int:
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible and visible not in {"-1", ""}:
        return len([item for item in visible.split(",") if item.strip()])
    if not shutil.which("nvidia-smi"):
        return 1
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    return max(1, sum(line.startswith("GPU ") for line in result.stdout.splitlines()))


def _wait_for_server(port: int, *, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(f"llama-server on port {port} did not become healthy")


def start_servers(binary: Path, model: Path, *, count: int) -> list[subprocess.Popen[bytes]]:
    servers: list[subprocess.Popen[bytes]] = []
    try:
        for gpu in range(count):
            port = 8080 + gpu
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            library_path = str(binary.parent)
            if environment.get("LD_LIBRARY_PATH"):
                library_path += f":{environment['LD_LIBRARY_PATH']}"
            environment["LD_LIBRARY_PATH"] = library_path
            process = subprocess.Popen(
                [
                    str(binary),
                    "--model",
                    str(model),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--ctx-size",
                    "8192",
                    "--n-gpu-layers",
                    "999",
                    "--parallel",
                    "1",
                    "--jinja",
                    "--reasoning-budget",
                    "512",
                ],
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            servers.append(process)
            _wait_for_server(port)
    except Exception:
        _stop_servers(servers)
        raise
    return servers


def _stop_servers(servers: list[subprocess.Popen[bytes]]) -> None:
    for process in servers:
        process.terminate()
    for process in servers:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def run_kaggle(
    *,
    input_root: Path = Path("/kaggle/input"),
    output_path: Path = Path("/kaggle/working/submission.json"),
    bundle_root: Path | None = None,
) -> Path:
    bundle = bundle_root or Path(os.environ.get("ARC_BUNDLE_ROOT", "/kaggle/input/arc-bonsai-v1"))
    package_root = bundle / "package"
    config = load_config(package_root / "configs/kaggle-gguf.yaml")
    cards_path = bundle / "skills"
    cards = load_cards(cards_path) if (cards_path / "cards.json").exists() else []
    tasks = load_tasks(find_challenges(input_root))

    disable_model = os.getenv("ARC_DISABLE_MODEL", "0") == "1"
    servers: list[subprocess.Popen[bytes]] = []
    try:
        worker_count = 1
        if disable_model:
            adapters = [None]
        else:
            binary = Path(os.environ.get("ARC_LLAMACPP_BIN", bundle / "bin/llama-server"))
            model = Path(os.environ.get("ARC_MODEL_GGUF", bundle / "model/Bonsai-27B-Q1_0.gguf"))
            if not binary.exists() or not model.exists():
                worker_count = 1
                adapters = [None]
            else:
                worker_count = min(_gpu_count(), 4, len(tasks))
                try:
                    servers = start_servers(binary, model, count=worker_count)
                except Exception:
                    servers = []
                    worker_count = 1
                    adapters = [None]
                else:
                    adapters = [
                        OpenAICompatibleAdapter(
                            base_url=f"http://127.0.0.1:{8080 + index}/v1",
                            model=config.model.model,
                            api_key="none",
                            max_tokens=config.model.max_tokens,
                            thinking_budget_tokens=config.model.thinking_budget_tokens,
                            temperature=config.model.temperature,
                            timeout_seconds=config.model.timeout_seconds,
                            vision_enabled=config.model.vision_enabled,
                            structural_summary_enabled=(
                                config.model.structural_summary_enabled
                            ),
                        )
                        for index in range(worker_count)
                    ]

        solvers = [ArcSolver(config, cards=cards, adapter=adapter) for adapter in adapters]
        indexed_runs: dict[int, TaskRun] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(solvers[index % worker_count].solve_task, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    indexed_runs[index] = future.result()
                except Exception:
                    indexed_runs[index] = ArcSolver(config, cards=cards).solve_task(tasks[index])
        submission: Submission = {
            tasks[index].task_id: indexed_runs[index].attempts for index in range(len(tasks))
        }
        return write_submission(submission, tasks, output_path)
    finally:
        _stop_servers(servers)


if __name__ == "__main__":
    print(run_kaggle())
