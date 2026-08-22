from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an offline ARC Bonsai bundle")
    parser.add_argument("bundle", type=Path, nargs="?", default=Path("dist/arc-bonsai-v1"))
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        parser.error(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    for relative, expected in manifest.get("files", {}).items():
        path = (bundle / relative).resolve()
        if bundle not in path.parents:
            errors.append(f"path escapes bundle: {relative}")
        elif not path.is_file():
            errors.append(f"missing: {relative}")
        elif path.stat().st_size != expected["bytes"]:
            errors.append(f"size mismatch: {relative}")
        elif checksum(path) != expected["sha256"]:
            errors.append(f"checksum mismatch: {relative}")
    for relative in (
        "bin/llama-server",
        "model/Bonsai-27B-Q1_0.gguf",
        "skills/cards.json",
        "package/src/arc_agent/kaggle_runner.py",
        "package/configs/kaggle-gguf.yaml",
    ):
        if not (bundle / relative).exists():
            errors.append(f"required file missing: {relative}")
    server = bundle / "bin/llama-server"
    if server.exists() and not os.access(server, os.X_OK):
        errors.append("bin/llama-server is not executable")
    if errors:
        raise SystemExit("Invalid bundle:\n- " + "\n- ".join(errors))
    print(f"Valid bundle: {len(manifest['files'])} files at {bundle}")


if __name__ == "__main__":
    main()
