"""CPU API fixture for completed-adapter recovery; no language model or training.

Use the isolated nanbeige-trainer environment. A single linear projection exercises
the pinned PEFT loader and Transformers state reader without a second Nanbeige
process, model weights, ARC examples, optimizer updates or remote publication.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import train_nanbeige_symbolic as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("preserve prior evidence; choose a new output directory")
    if not args.output.parent.is_dir():
        raise ValueError("output parent must already exist")
    if shutil.disk_usage(args.output.parent).free < 10 * 1024**3:
        raise RuntimeError("preserve at least 10 GiB free disk")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "peft", "transformers")
    }
    if versions != {"torch": "2.11.0", "peft": "0.18.0", "transformers": "4.57.6"}:
        raise ValueError("use the pinned training runtime")

    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from transformers import TrainerState

    torch.set_num_threads(1)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(16, 16)

        def forward(self, values):
            return self.projection(values)

    def adapter():
        torch.manual_seed(42)
        return get_peft_model(Projection(), LoraConfig(r=16, target_modules=["projection"]))

    def state(model):
        return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}

    def forbidden_train(*args, **kwargs):
        raise AssertionError("completed recovery must never train")

    args.output.mkdir()
    folder = args.output / "synthetic-checkpoint-2"
    model = adapter()
    # Deliberately synthetic nonzero tensors, not learned examples or a trained adapter.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.fill_(0.125)
    expected = state(model)
    model.save_pretrained(folder, save_embedding_layers=False)
    TrainerState(global_step=2).save_to_json(str(folder / "trainer_state.json"))
    original_hashes = {path.name: runner.file_hash(path) for path in folder.iterdir()}
    fresh = adapter()
    if all(torch.equal(expected[k], value) for k, value in state(fresh).items()):
        raise AssertionError("fixture must distinguish untouched and restored tensors")
    trainer = SimpleNamespace(model=fresh, state=TrainerState(), train=forbidden_train)
    runner.restore_completed_adapter(trainer, folder, 2)
    recovered = state(trainer.model)
    if set(recovered) != set(expected) or not all(
        torch.equal(expected[k], value) for k, value in recovered.items()
    ):
        raise AssertionError("actual PEFT adapter restoration was not exact")
    if trainer.state.global_step != 2 or original_hashes != {
        path.name: runner.file_hash(path) for path in folder.iterdir()
    }:
        raise AssertionError("recovery changed the checkpoint or reporting cursor")
    report = {
        "passed": True,
        "scope": "single linear projection; actual PEFT/TrainerState APIs, not a language model",
        "environment": versions,
        "runner_sha256": runner.file_hash(runner.__file__),
        "probe_sha256": runner.file_hash(__file__),
        "checkpoint_hashes": original_hashes,
        "adapter_exactly_restored": True,
        "trainer_cursor_restored": True,
        "checkpoint_unchanged": True,
        "language_model_loaded": False,
        "official_weights_loaded": False,
        "optimizer_updates": 0,
        "training_launched": False,
        "student_trained": False,
        "gpu_training_verified": False,
        "promotion_proven": False,
    }
    runner.atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
