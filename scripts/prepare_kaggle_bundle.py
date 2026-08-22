from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble an offline Kaggle ARC Bonsai bundle")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--skills", type=Path, default=Path("skills/generated-full"))
    parser.add_argument("--output", type=Path, default=Path("dist/arc-bonsai-v1"))
    args = parser.parse_args()
    for source in (args.gguf, args.llama_server, args.skills / "cards.json"):
        if not source.exists():
            parser.error(f"missing required artifact: {source}")

    output: Path = args.output
    (output / "model").mkdir(parents=True, exist_ok=True)
    (output / "bin").mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.gguf, output / "model/Bonsai-27B-Q1_0.gguf")
    shutil.copy2(args.llama_server, output / "bin/llama-server")
    for pattern in ("*.so", "*.so.*"):
        for library in args.llama_server.parent.glob(pattern):
            shutil.copy2(library, output / "bin" / library.name)
    (output / "bin/llama-server").chmod(0o755)
    shutil.copytree(args.skills, output / "skills", dirs_exist_ok=True)
    shutil.copytree("src", output / "package/src", dirs_exist_ok=True)
    shutil.copytree("configs", output / "package/configs", dirs_exist_ok=True)
    shutil.copy2("pyproject.toml", output / "package/pyproject.toml")
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "files": {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": checksum(path),
            }
            for path in files
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
