"""CPU-only architecture/collator probe, never a trained solver or a paid job.

Use .runtime/nanbeige-trainer/bin/python. The full-size architecture is constructed
on the meta device (no weight storage). Numerical tests use a randomly initialized,
reduced-width Nanbeige fixture with the original vocabulary and two shared loops.
No ARC outputs, teacher witnesses or evaluation records supervise this fixture.
Passing does not validate training the official 3B checkpoint or GPU throughput.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_CODE = {
    "modeling_nanbeige.py": "547737f989f5cb741c1a568acaf85f83521fa67a8f7268e19f9a37e60127c0d5",
    "configuration_nanbeige.py": "c517227741f0fc007061bde1873138545ce15e67716a062513193c06300c7685",
}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve prior probe artifacts; choose a new output directory")
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError("preserve at least 10 GiB free disk")
    with socket.socket() as sock:
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", 8094)) == 0:
            raise RuntimeError("finish the owned inference process before the CPU model probe")
    args.output.mkdir(parents=True)
    os.environ["HF_MODULES_CACHE"] = str(ROOT / ".runtime/nanbeige/trainer-probe-modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRACKIO_DIR"] = str(args.output.resolve() / "trackio")
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    import torch
    import trackio
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from arc_agent.v4_config import MODEL_REVISION, content_hash, file_hash
    from arc_agent.v4_runtime import memory_sample
    from arc_agent.v4_state import atomic_json
    from arc_agent.v5_dataset import prepare_student

    torch.set_num_threads(2)
    if memory_sample()["pressure_level"] not in {1, 2}:
        raise RuntimeError("critical or unknown memory pressure")
    model_dir = ROOT / ".runtime/nanbeige/model-hf"
    for name, expected in MODEL_CODE.items():
        path = model_dir / name
        metadata = (
            (model_dir / ".cache/huggingface/download" / f"{name}.metadata")
            .read_text()
            .splitlines()
        )
        blob = path.read_bytes()
        git_blob = hashlib.sha1(f"blob {len(blob)}\0".encode() + blob).hexdigest()
        if file_hash(path) != expected or metadata[:2] != [MODEL_REVISION, git_blob]:
            raise ValueError("custom model source differs from pinned official download")
    report = {
        "passed": False,
        "scope": "CPU random architecture fixture and real-data collator; not base-weight training",
        "official_weights_loaded": False,
        "student_trained": False,
        "gpu_training_verified": False,
        "promotion_proven": False,
        "model_code_hashes": MODEL_CODE,
        "environment": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }
    trackio.init(
        project="nanbeige-compatibility", name=args.output.name, config={"fixture_only": True}
    )
    try:
        # Re-tokenize every admitted row from its native prompt/target, checking all masks.
        data_report = prepare_student(args.data, model_dir)
        if not data_report["loss_mask_data_gate_passed"]:
            raise ValueError("all probe data must pass native-format preparation")
        rows = json.loads((args.data / "nanbeige-symbolic-tokenized.json").read_text())
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True
        )
        collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id, completion_only_loss=True
        )
        for index in range(0, len(rows), 2):
            group = rows[index : index + 2]
            batch = collator(group)
            for j, row in enumerate(group):
                size = len(row["input_ids"])
                assert batch["input_ids"][j, :size].tolist() == row["input_ids"]
                assert batch["labels"][j, :size].tolist() == row["labels"]
                assert (batch["labels"][j, size:] == -100).all()
                assert (batch["attention_mask"][j, size:] == 0).all()
        report.update(
            real_rows_checked=len(rows),
            real_data_hash=data_report["dataset_hash"],
            collator_masks_verified=True,
        )

        official = AutoConfig.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True
        )
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGETS,
        )
        with torch.device("meta"):
            full = AutoModelForCausalLM.from_config(
                official, trust_remote_code=True, attn_implementation="sdpa"
            )
            full = get_peft_model(full, lora)
        modules = [(n, m) for n, m in full.named_modules() if hasattr(m, "lora_A")]
        assert len(modules) == 22 * len(TARGETS)
        assert len({id(m) for _, m in modules}) == len(modules)
        assert full.base_model.model.model._get_layer_execution_order() == [
            (i, None) for i in range(22)
        ]
        assert full.base_model.model.model._get_num_loops() == 2
        assert all("lora_" in n for n, p in full.named_parameters() if p.requires_grad)
        report.update(
            unique_lora_modules=len(modules),
            physical_layers=22,
            shared_loops=2,
            planned_trainable_parameters=sum(
                p.numel() for p in full.parameters() if p.requires_grad
            ),
        )
        del full, modules

        fixture_cfg = copy.deepcopy(official)
        for key, value in {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "kv_channels": 8,
            "use_cache": False,
        }.items():
            setattr(fixture_cfg, key, value)

        def fixture_base():
            torch.manual_seed(123)
            return AutoModelForCausalLM.from_config(
                fixture_cfg, trust_remote_code=True, attn_implementation="sdpa"
            )

        def fixture_model():
            return get_peft_model(fixture_base(), copy.deepcopy(lora))

        # Numeric token fixtures are not ARC training examples or model-generated targets.
        synthetic = Dataset.from_list(
            [
                {
                    "input_ids": [166100, 41 + i, 32, 42, 33, 166101],
                    "labels": [-100, -100, -100, 42, 33, 166101],
                }
                for i in range(4)
            ]
        )

        class Metrics(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                trackio.log(
                    {
                        "fixture/" + k: v
                        for k, v in (logs or {}).items()
                        if isinstance(v, (int, float))
                    }
                )

        def trainer(model, output, dataset=synthetic):
            return SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dataset,
                data_collator=collator,
                callbacks=[Metrics()],
                args=SFTConfig(
                    output_dir=str(output),
                    use_cpu=True,
                    bf16=False,
                    fp16=False,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    max_steps=2,
                    learning_rate=1e-4,
                    lr_scheduler_type="cosine",
                    seed=123,
                    data_seed=123,
                    save_strategy="steps",
                    save_steps=1,
                    save_total_limit=None,
                    logging_steps=1,
                    report_to=[],
                    disable_tqdm=True,
                    packing=False,
                    padding_free=False,
                    max_length=8192,
                    completion_only_loss=True,
                    dataset_kwargs={"skip_prepare_dataset": True},
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                ),
            )

        # Exercise the actual Trainer data loader on full, untruncated real rows,
        # without forwarding those long rows through the tiny numeric fixture.
        real_trainer = trainer(
            fixture_model(), args.output / "collator-only", Dataset.from_list(rows)
        )
        expected_masks = {tuple(r["input_ids"]): r["labels"] for r in rows}
        seen = 0
        for batch in real_trainer.get_train_dataloader():
            for ids, labels, attention in zip(
                batch["input_ids"], batch["labels"], batch["attention_mask"], strict=True
            ):
                length = int(attention.sum())
                assert labels[:length].tolist() == expected_masks[tuple(ids[:length].tolist())]
                assert (labels[length:] == -100).all()
                seen += 1
        assert seen == len(rows)
        report["actual_trainer_real_data_masks_verified"] = True
        del real_trainer

        original_model = fixture_model()
        original_trainer = trainer(original_model, args.output / "uninterrupted")
        assert original_trainer.train_dataset.to_list() == synthetic.to_list()
        initial_adapter = {
            n: p.detach().clone() for n, p in original_model.named_parameters() if p.requires_grad
        }
        original_trainer.train()
        final_adapter = {
            n: p.detach().clone() for n, p in original_model.named_parameters() if p.requires_grad
        }
        assert any(not torch.equal(initial_adapter[n], p) for n, p in final_adapter.items())
        checkpoint = args.output / "uninterrupted/checkpoint-1"
        required = {
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "adapter_model.safetensors",
        }
        assert required.issubset(p.name for p in checkpoint.iterdir())
        resumed_model = fixture_model()
        resumed = trainer(resumed_model, args.output / "resumed")
        resumed.train(resume_from_checkpoint=str(checkpoint))
        assert resumed.state.global_step == 2
        assert all(
            torch.equal(final_adapter[n], p)
            for n, p in resumed_model.named_parameters()
            if p.requires_grad
        )
        final_path = args.output / "final-fixture-adapter"
        original_model.save_pretrained(final_path)
        reloaded = PeftModel.from_pretrained(fixture_base(), final_path, is_trainable=True)
        assert all(
            torch.equal(final_adapter[n], p)
            for n, p in reloaded.named_parameters()
            if p.requires_grad
        )
        report.update(
            passed=True,
            small_fixture_training_step_verified=True,
            exact_adapter_resume_verified=True,
            adapter_save_reload_verified=True,
            trainer_dataset_unchanged=True,
            fixture_dataset_hash=content_hash(synthetic.to_list()),
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        trackio.finish()
        atomic_json(args.output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
