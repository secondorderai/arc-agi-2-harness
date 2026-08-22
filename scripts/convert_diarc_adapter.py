from __future__ import annotations

import argparse
from pathlib import Path

from arc_agent.diarc import convert_diarc_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a DiARC PEFT adapter to MLX format")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    converted, skipped = convert_diarc_adapter(args.source, args.output)
    print(f"Converted {converted} LoRA tensors; skipped {skipped} frozen base tensors")


if __name__ == "__main__":
    main()
