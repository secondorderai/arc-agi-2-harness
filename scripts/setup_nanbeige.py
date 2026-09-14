#!/usr/bin/env python3
"""Reproducible local setup. Never deletes weights or changes a system Python installation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from arc_agent.v4_config import MODEL_ID, MODEL_REVISION, RUNTIME_REVISION, file_hash
from arc_agent.v4_state import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime/nanbeige"
REPO = RUNTIME / "llama.cpp"
PYTHON = RUNTIME / "venv/bin/python"
BF16 = RUNTIME / "Nanbeige4.2-3B-BF16.gguf"
Q4 = RUNTIME / "Nanbeige4.2-3B-Q4_K_M.gguf"
WEIGHTS = {
    "model-00001-of-00002.safetensors": (
        "09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94"
    ),
    "model-00002-of-00002.safetensors": (
        "31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389"
    ),
}
PACKAGES = [
    "huggingface-hub==0.36.0",
    "transformers==4.57.6",
    "torch==2.11.0",
    "numpy==1.26.4",
    "protobuf==4.25.8",
    "sentencepiece==0.2.1",
    "safetensors==0.6.2",
    "jinja2==3.1.6",
]


def disk(required_gib=0):
    if shutil.disk_usage(RUNTIME).free < (10 + required_gib) * 1024**3:
        raise RuntimeError(
            f"need {required_gib:g} GiB for this stage plus 10 GiB untouched; no files deleted"
        )


def command(args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def check_revision():
    revision = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if revision != RUNTIME_REVISION or dirty:
        raise ValueError("runtime revision changed or tracked source is modified; refusing reuse")


def setup():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("this installer is for the local Apple Silicon gate")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    disk(2)
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("uv must be available; this script does not replace system runtimes")
    prefix = [uv, "--cache-dir", RUNTIME / "uv-cache"]
    if not PYTHON.exists():
        command(prefix + ["venv", "--python", "3.11.5", RUNTIME / "venv"])
    command(prefix + ["pip", "install", "--python", PYTHON, *PACKAGES, "-e", ".[dev,v4]"])
    if not REPO.exists():
        command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "nanbeige42",
                "https://github.com/Nanbeige/llama.cpp.git",
                REPO,
            ]
        )
    # A moved branch is not accepted implicitly.
    check_revision()
    missing_bytes = sum(4.2 for name in WEIGHTS if not (RUNTIME / "model-hf" / name).exists())
    disk(missing_bytes)
    command(
        [
            RUNTIME / "venv/bin/hf",
            "download",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--local-dir",
            RUNTIME / "model-hf",
            "--include",
            "*.json",
            "*.model",
            "*.safetensors",
            "*.py",
            "README.md",
            "--max-workers",
            "2",
        ]
    )


def build():
    check_revision()
    disk(1)
    command(
        [
            "cmake",
            "-S",
            REPO,
            "-B",
            REPO / "build",
            "-DGGML_METAL=ON",
            "-DGGML_CUDA=OFF",
            "-DGGML_OPENMP=OFF",
            "-DLLAMA_BUILD_UI=OFF",
            "-DLLAMA_USE_PREBUILT_UI=OFF",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    command(
        [
            "cmake",
            "--build",
            REPO / "build",
            "--config",
            "Release",
            "--target",
            "llama-server",
            "llama-quantize",
            "-j",
            "4",
        ]
    )


def build_grammar_validator():
    command(
        [
            "clang++",
            "-std=c++17",
            REPO / "tests/test-gbnf-validator.cpp",
            "-I" + str(REPO / "include"),
            "-I" + str(REPO / "ggml/include"),
            "-I" + str(REPO / "src"),
            "-L" + str(REPO / "build/bin"),
            "-Wl,-rpath," + str(REPO / "build/bin"),
            "-lllama",
            "-lggml-base",
            "-o",
            RUNTIME / "gbnf-validator",
        ]
    )


def verify_weights():
    for name, expected in WEIGHTS.items():
        if file_hash(RUNTIME / "model-hf" / name) != expected:
            raise ValueError(f"official weight checksum mismatch: {name}")


def gguf_metadata(path):
    sys.path.insert(0, str(REPO / "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(path)
    names = [
        "general.architecture",
        "general.file_type",
        "nanbeige.vocab_size",
        "nanbeige.num_loops",
        "nanbeige.block_count",
    ]
    metadata = {name: reader.fields[name].contents() for name in names}
    if (
        metadata["general.architecture"] != "nanbeige"
        or metadata["nanbeige.vocab_size"] != 166144
        or metadata["nanbeige.num_loops"] != 2
        or metadata["nanbeige.block_count"] != 22
    ):
        raise ValueError("GGUF is not the pinned Nanbeige architecture")
    if len(reader.tensors) != 201:
        raise ValueError("incomplete Nanbeige tensor inventory")
    return metadata


def convert():
    check_revision()
    verify_weights()
    if not BF16.exists():
        disk(8)
        temporary = BF16.with_suffix(".partial.gguf")
        if temporary.exists():
            raise ValueError(f"preserving incomplete conversion for inspection: {temporary}")
        command(
            [
                PYTHON,
                REPO / "convert_hf_to_gguf.py",
                RUNTIME / "model-hf",
                "--outfile",
                temporary,
                "--outtype",
                "bf16",
            ]
        )
        gguf_metadata(temporary)
        os.replace(temporary, BF16)
    gguf_metadata(BF16)
    if not Q4.exists():
        disk(3)
        temporary = Q4.with_suffix(".partial.gguf")
        if temporary.exists():
            raise ValueError(f"preserving incomplete quantization for inspection: {temporary}")
        command([REPO / "build/bin/llama-quantize", BF16, temporary, "Q4_K_M", "4"])
        gguf_metadata(temporary)
        os.replace(temporary, Q4)


def manifest():
    check_revision()
    verify_weights()
    disk()
    metadata = gguf_metadata(Q4)
    if metadata["general.file_type"] != 15:
        raise ValueError("Q4_K_M quantization required")
    files = {
        "model": Q4,
        "server": REPO / "build/bin/llama-server",
        "tokenizer_config": RUNTIME / "model-hf/tokenizer_config.json",
        "model_config": RUNTIME / "model-hf/config.json",
        "tokenizer_model": RUNTIME / "model-hf/tokenizer.model",
        "tokenizer_json": RUNTIME / "model-hf/tokenizer.json",
        "model_card": RUNTIME / "model-hf/README.md",
        "runtime_license": REPO / "LICENSE",
    }
    # Include actual shared libraries, not only the small server launcher.
    for path in sorted((REPO / "build/bin").glob("*.dylib")):
        if not path.is_symlink():
            files["runtime_" + path.name] = path
    payload = {
        "version": 1,
        "created_at": time.time(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_revision": RUNTIME_REVISION,
        "runtime_repository": "https://github.com/Nanbeige/llama.cpp.git",
        "runtime_branch": "nanbeige42",
        "quantization": "Q4_K_M",
        "adapter": None,
        "gguf_metadata": metadata,
        "official_weight_sha256": WEIGHTS,
        "build_flags": {"metal": True, "cuda": False, "openmp": False, "ui": False},
        "conversion": {"intermediate": "BF16", "tokenizer_pruned": False},
        "files": {
            key: {
                "path": str(path.resolve()),
                "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            }
            for key, path in files.items()
        },
        "environment": subprocess.check_output([str(PYTHON), "-m", "pip", "freeze"], text=True)
        if (RUNTIME / "venv/bin/pip").exists()
        else subprocess.check_output(
            [
                shutil.which("uv"),
                "--cache-dir",
                str(RUNTIME / "uv-cache"),
                "pip",
                "freeze",
                "--python",
                str(PYTHON),
            ],
            text=True,
        ),
        "license_audit": (
            "Model card Apache-2.0 and runtime MIT recorded; competition eligibility not certified"
        ),
    }
    destination = RUNTIME / "manifest.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["files"] != payload["files"]:
            raise ValueError("existing artifact manifest differs; preserve it and investigate")
        print(f"Verified existing manifest: {destination}")
    else:
        atomic_json(destination, payload)
        print(f"Wrote verified manifest: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["all", "setup", "build", "convert", "manifest"], default="all"
    )
    arguments = parser.parse_args()
    for name, operation in [
        ("setup", setup),
        ("build", build),
        ("convert", convert),
        ("manifest", manifest),
    ]:
        if arguments.stage in {"all", name}:
            if name != "setup" and Path(sys.prefix).resolve() != (RUNTIME / "venv").resolve():
                command([PYTHON, __file__, "--stage", name])
            else:
                operation()
                if name == "build":
                    build_grammar_validator()
