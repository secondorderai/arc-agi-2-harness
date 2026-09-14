# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.11.0", "transformers==4.57.6", "trl==0.25.1",
#   "peft==0.18.0", "accelerate==1.12.0", "datasets==4.4.1",
#   "trackio==0.11.0", "sentencepiece==0.2.1", "protobuf==4.25.8",
#   "huggingface-hub==0.36.2",
# ]
# ///
"""Approval-gated, single-CUDA Nanbeige symbolic SFT and full-size compatibility probe.

Submit this source inline only after approval. The private bundle must contain the
read-only preflight source plus its ten evidence files. No job is launched by
importing this module. Local Mac execution is deliberately rejected before Hub use.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

MODEL = "Nanbeige/Nanbeige4.2-3B"
REVISION = "3384e426066d1a49c3aea90a7190b81260a6533f"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
WEIGHTS = {
    "model.safetensors.index.json": (
        "30d8da0fa8b97abc6d9eddfd017a0cb7a649bcbd58caa57804c56c767db5c0f1"
    ),
    "model-00001-of-00002.safetensors": (
        "09d265d5ec837bc64462796b7f8c110be9a135a55ed7a6eb5d07e0e90c976a94"
    ),
    "model-00002-of-00002.safetensors": (
        "31019e7870a044f44bc3f7e981f8c5ecd42d341e5ca6cfdbfd07fb95d95be389"
    ),
}
CHECKPOINT_REQUIRED = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
    "data-cursor.json",
    "run-identity.json",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_preflight(path, expected_hash):
    if file_hash(path) != expected_hash:
        raise ValueError("preflight source differs from approved hash")
    spec = importlib.util.spec_from_file_location("nanbeige_training_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_launch(manifest, mode, script_hash):
    """References record human approval; this function never grants it."""
    launch = manifest.get("gpu_launches", {}).get(mode, {})
    for key in (
        "compute_approval_ref",
        "output_use_permission_ref",
        "model_repo",
        "trackio_space",
        "trackio_dataset",
        "run_id",
        "hardware",
        "script_sha256",
    ):
        if not isinstance(launch.get(key), str) or not launch[key].strip():
            raise ValueError(f"missing approved launch field: {key}")
    if launch["script_sha256"] != script_hash:
        raise ValueError("GPU script changed after approval")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", launch["run_id"]):
        raise ValueError("run ID must be a single safe path component")
    for key in ("model_repo", "trackio_space", "trackio_dataset"):
        if not re.fullmatch(r"[\w-]+/[\w.-]+", launch[key]):
            raise ValueError("require exact approved Hub destinations")
    if type(launch.get("timeout_seconds")) is not int or launch["timeout_seconds"] <= 0:
        raise ValueError("approved technical job timeout is required")
    return launch


def require_private(api, repo, kind):
    if api.repo_info(repo, repo_type=kind).private is not True:
        raise ValueError(f"destination is not verified private: {repo}")


def seal_checkpoint(folder, identity, schedule, step):
    folder = Path(folder)
    cursor = {
        "global_step": step,
        "next_draw": min(step * 16, len(schedule)),
        "schedule_hash": digest(schedule),
        "total_draws": len(schedule),
        "microbatch": 1,
        "gradient_accumulation": 16,
        "world_size": 1,
        "sampler": "sequential_explicit_schedule",
    }
    atomic_json(folder / "data-cursor.json", cursor)
    atomic_json(folder / "run-identity.json", identity)
    files = {p.name for p in folder.iterdir() if p.is_file()}
    if not files >= CHECKPOINT_REQUIRED:
        raise ValueError(f"incomplete checkpoint: {sorted(CHECKPOINT_REQUIRED - files)}")
    if any(p.is_symlink() or not p.is_file() for p in folder.iterdir()):
        raise ValueError("checkpoints must contain regular files only")
    if read_json(folder / "trainer_state.json")["global_step"] != step:
        raise ValueError("trainer state and data cursor disagree")
    validate_adapter_config(read_json(folder / "adapter_config.json"))
    manifest = {"files": {name: file_hash(folder / name) for name in sorted(files)}}
    atomic_json(folder / "checkpoint-manifest.json", manifest)
    return file_hash(folder / "checkpoint-manifest.json")


def validate_checkpoint(folder, manifest_hash, identity, schedule):
    folder = Path(folder)
    if file_hash(folder / "checkpoint-manifest.json") != manifest_hash:
        raise ValueError("checkpoint manifest changed")
    manifest = read_json(folder / "checkpoint-manifest.json")
    if not set(manifest["files"]) >= CHECKPOINT_REQUIRED:
        raise ValueError("optimizer, scheduler, RNG and data cursor are all required")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or (folder / name).is_symlink():
            raise ValueError("invalid checkpoint member")
        if file_hash(folder / name) != expected:
            raise ValueError(f"checkpoint file changed: {name}")
    if read_json(folder / "run-identity.json") != identity:
        raise ValueError("resume identity differs from the frozen training protocol")
    validate_adapter_config(read_json(folder / "adapter_config.json"))
    cursor = read_json(folder / "data-cursor.json")
    step = cursor["global_step"]
    if (
        type(step) is not int
        or not 1 <= step <= math.ceil(len(schedule) / 16)
        or cursor["schedule_hash"] != digest(schedule)
        or cursor["next_draw"] != min(step * 16, len(schedule))
        or cursor["total_draws"] != len(schedule)
        or cursor["microbatch"] != 1
        or cursor["gradient_accumulation"] != 16
        or cursor["world_size"] != 1
        or cursor["sampler"] != "sequential_explicit_schedule"
        or read_json(folder / "trainer_state.json")["global_step"] != step
    ):
        raise ValueError("checkpoint data cursor changed")
    return cursor


def validate_adapter_config(config):
    if (
        config.get("base_model_name_or_path") != MODEL
        or config.get("revision") != REVISION
        or config.get("r") != 16
        or config.get("lora_alpha") != 32
        or config.get("lora_dropout") != 0.05
        or config.get("peft_type") != "LORA"
        or config.get("task_type") != "CAUSAL_LM"
        or set(config.get("target_modules", [])) != set(TARGETS)
        or config.get("bias") != "none"
        or config.get("modules_to_save")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
    ):
        raise ValueError("adapter differs from pinned Nanbeige symbolic LoRA protocol")


def restore_completed_adapter(trainer, checkpoint, expected_step):
    """Restore a previously validated finished checkpoint without calling train().

    This restores the adapter and reporting cursor only. No optimizer/RNG state is
    deserialized or claimed to have been exercised during report-only recovery.
    The caller must first validate every checkpoint member and its frozen identity.
    """
    state = type(trainer.state).load_from_json(str(Path(checkpoint) / "trainer_state.json"))
    if type(state.global_step) is not int or state.global_step != expected_step:
        raise ValueError("finished checkpoint step differs from the declared schedule")
    loaded = trainer.model.load_adapter(
        str(checkpoint), adapter_name="default", is_trainable=True, local_files_only=True
    )
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("finished adapter did not restore completely")
    trainer.state = state


def execute_schedule(trainer, identity, schedule, resume=None, resume_hash=None):
    """Train remaining draws, or recover the exact finished adapter without new work."""
    final_step = math.ceil(len(schedule) / 16)
    if not final_step:
        raise ValueError("training schedule must be nonempty")
    cursor = None
    if resume is not None:
        resume = Path(resume)
        cursor = validate_checkpoint(resume, resume_hash, identity, schedule)
        if resume.name != f"checkpoint-{cursor['global_step']}":
            raise ValueError("checkpoint path and validated cursor disagree")
    initial_step = cursor["global_step"] if cursor else 0
    finished = cursor is not None and cursor["next_draw"] == len(schedule)
    if finished:
        restore_completed_adapter(trainer, resume, final_step)
        final_checkpoint = resume
    else:
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        final_checkpoint = Path(trainer.args.output_dir) / f"checkpoint-{final_step}"
    if trainer.state.global_step != final_step:
        raise ValueError("training did not complete the declared schedule")
    # A result must refer to a complete checkpoint, not just in-memory trainer state.
    validate_checkpoint(
        final_checkpoint,
        file_hash(final_checkpoint / "checkpoint-manifest.json"),
        identity,
        schedule,
    )
    return {
        "final_checkpoint": final_checkpoint,
        "optimizer_updates_this_invocation": final_step - initial_step,
        "recovered_completed_checkpoint": finished,
    }


class Publisher:
    """Synchronous, append-only checkpoint commits to an already-private repository."""

    def __init__(self, api, repo, prefix, download):
        self.api, self.repo, self.prefix, self.download = api, repo, prefix, download

    def publish(self, folder, name, marker):
        require_private(self.api, self.repo, "model")
        destination = f"{self.prefix}/{name}"
        head = self.api.repo_info(self.repo, repo_type="model").sha
        existing = self.api.list_repo_files(self.repo, repo_type="model", revision=head)
        if any(path == destination or path.startswith(destination + "/") for path in existing):
            raise ValueError("preserve existing remote artifacts; do not overwrite a checkpoint")
        result = self.api.upload_folder(
            repo_id=self.repo,
            repo_type="model",
            folder_path=str(folder),
            path_in_repo=destination,
            commit_message=f"Durable Nanbeige artifact: {name}",
            parent_commit=head,
        )
        remote = self.download(
            repo_id=self.repo,
            repo_type="model",
            revision=result.oid,
            filename=f"{destination}/{marker}",
            force_download=True,
        )
        if file_hash(remote) != file_hash(Path(folder) / marker):
            raise ValueError("remote checkpoint marker verification failed")
        return {
            "repo": self.repo,
            "revision": result.oid,
            "path": destination,
            "manifest_sha256": file_hash(remote),
        }


def check_compatibility(report, expected):
    if (
        report.get("binding") != expected
        or report.get("passed") is not True
        or report.get("mode") != "compatibility"
    ):
        raise ValueError("full-size compatibility evidence does not match this training job")
    for key in (
        "official_weights_loaded",
        "gpu_training_verified",
        "real_masks_verified",
        "shared_layers_verified",
        "adapter_save_reload_verified",
        "exact_resume_verified",
        "padded_8k_forward_backward_verified",
        "durable_checkpoints_verified",
    ):
        if report.get(key) is not True:
            raise ValueError(f"missing full-size compatibility check: {key}")


def train(args, preflight, manifest, examples, rows, model_dir, launch, api):
    import torch
    import trackio
    from datasets import Dataset
    from huggingface_hub import hf_hub_download, snapshot_download
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from torch.utils.data import SequentialSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )
    preflight.validate_native(examples, rows, tokenizer)
    environment = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
    }
    binding = {
        "model": MODEL,
        "revision": REVISION,
        "dataset_hash": digest(examples),
        "environment": environment,
        "preflight_hash": args.preflight_sha256,
        "script_hash": file_hash(__file__),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "effective_batch": 16,
        "max_length": 8192,
        "seed": 42,
    }
    if args.mode == "train":
        if not args.compatibility_report or not args.compatibility_sha256:
            raise ValueError("run and approve the full-size compatibility job first")
        if file_hash(args.compatibility_report) != args.compatibility_sha256:
            raise ValueError("compatibility report hash changed")
        check_compatibility(read_json(args.compatibility_report), binding)

    schedule = preflight.balanced_schedule(rows)
    if args.mode == "compatibility":
        # Original admitted rows only; longest first to exercise the real long-row path.
        longest = max(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
        schedule = [{"row": longest, "logical_epoch": 0}] + schedule[:31]
    dataset = Dataset.from_list(
        [
            {key: rows[draw["row"]][key] for key in ("input_ids", "labels", "attention_mask")}
            for draw in schedule
        ]
    )
    identity = {
        **binding,
        "mode": args.mode,
        "run_id": launch["run_id"],
        "bundle_hash": args.manifest_sha256,
        "schedule_hash": digest(schedule),
    }
    output = args.output.resolve()
    if output.exists():
        raise ValueError("preserve prior local run output; use a new job directory")
    output.mkdir(parents=True)
    atomic_json(output / "run-identity.json", identity)
    atomic_json(output / "schedule.json", schedule)
    publisher = Publisher(api, launch["model_repo"], f"runs/{launch['run_id']}", hf_hub_download)
    prefix_files = [
        p for p in api.list_repo_files(launch["model_repo"]) if p.startswith(publisher.prefix + "/")
    ]
    if prefix_files and not args.resume_checkpoint:
        raise ValueError("remote run already exists; resume its immutable checkpoint explicitly")

    def base_model():
        set_seed(42)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        return model

    def adapted_model():
        model = get_peft_model(
            base_model(),
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=TARGETS,
            ),
        )
        modules = [m for m in model.modules() if hasattr(m, "lora_A")]
        core = model.base_model.model.model
        if (
            len(modules) != 154
            or len({id(m) for m in modules}) != 154
            or core._get_num_loops() != 2
            or core._get_layer_execution_order() != [(i, None) for i in range(22)]
            or sum(p.numel() for p in model.parameters() if p.requires_grad) != 23969792
            or any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad)
        ):
            raise ValueError("official shared-layer LoRA attachment changed")
        # Do not serialize the ephemeral Hub cache path as the adapter's base identity.
        model.peft_config["default"].base_model_name_or_path = MODEL
        model.peft_config["default"].revision = REVISION
        return model

    class Metrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            if any(not math.isfinite(v) for v in values.values()):
                raise ValueError("nonfinite training metric")
            values["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            trackio.log(values)

    checkpoint_refs = {}

    class DurableTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def _push_from_checkpoint(self, checkpoint_folder):
            folder = Path(checkpoint_folder)
            checksum = seal_checkpoint(folder, identity, schedule, self.state.global_step)
            validate_checkpoint(folder, checksum, identity, schedule)
            name = f"{Path(self.args.output_dir).name}/{folder.name}"
            checkpoint_refs[name] = publisher.publish(folder, name, "checkpoint-manifest.json")
            atomic_json(output / "durable-checkpoints.json", checkpoint_refs)

    def make_trainer(model, name):
        collator = DataCollatorForLanguageModeling(
            pad_token_id=tokenizer.pad_token_id,
            completion_only_loss=True,
            pad_to_multiple_of=8192 if args.mode == "compatibility" else 8,
        )
        trainer = DurableTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            data_collator=collator,
            callbacks=[Metrics()],
            args=SFTConfig(
                output_dir=str(output / name),
                push_to_hub=True,
                hub_model_id=launch["model_repo"],
                hub_private_repo=True,
                hub_strategy="end",
                # Two logical epochs are already materialized in the explicit schedule.
                num_train_epochs=1,
                max_steps=2 if args.mode == "compatibility" else -1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                learning_rate=1e-4,
                lr_scheduler_type="cosine",
                warmup_ratio=0,
                seed=42,
                data_seed=42,
                bf16=True,
                fp16=False,
                tf32=False,
                optim="adamw_torch",
                dataloader_num_workers=0,
                dataloader_drop_last=False,
                save_strategy="steps",
                save_steps=1,
                save_total_limit=None,
                save_only_model=False,
                logging_steps=1,
                logging_nan_inf_filter=False,
                report_to=[],
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                packing=False,
                padding_free=False,
                max_length=8192,
                completion_only_loss=True,
                dataset_kwargs={"skip_prepare_dataset": True},
                disable_tqdm=True,
            ),
        )
        preflight.validate_native(examples, rows, tokenizer)
        expected = {tuple(r["input_ids"]): r["labels"] for r in rows}
        seen = 0
        for batch in trainer.get_train_dataloader():
            ids, labels, attention = (
                batch[k][0] for k in ("input_ids", "labels", "attention_mask")
            )
            length = int(attention.sum())
            if ids[:length].tolist() != dataset[seen]["input_ids"]:
                raise ValueError("actual training loader reordered the explicit schedule")
            if labels[:length].tolist() != expected[tuple(ids[:length].tolist())]:
                raise ValueError("actual training loader altered completion-only labels")
            if not (labels[length:] == -100).all() or length > 8192:
                raise ValueError("actual training loader truncation or padding supervision")
            if args.mode == "compatibility" and len(ids) != 8192:
                raise ValueError("8K padded compatibility workload not exercised")
            seen += 1
        if seen != len(schedule):
            raise ValueError("actual training loader changed schedule length")
        return trainer

    def adapter_state(model):
        return {n: t.detach().cpu().clone() for n, t in get_peft_model_state_dict(model).items()}

    def equal_adapters(a, b):
        return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)

    started = time.monotonic()
    trackio.init(
        project="arc-nanbeige-symbolic",
        name=launch["run_id"],
        private=True,
        space_id=launch["trackio_space"],
        dataset_id=launch["trackio_dataset"],
        config=binding,
    )
    try:
        model = adapted_model()
        trainer = make_trainer(model, "reference" if args.mode == "compatibility" else "train")
        before = adapter_state(model)
        resume = None
        if args.resume_checkpoint:
            if args.mode != "train" or not args.resume_revision or not args.resume_manifest_sha256:
                raise ValueError("training resume requires an immutable revision and manifest hash")
            if not re.fullmatch(
                re.escape(publisher.prefix) + r"/train/checkpoint-[1-9][0-9]*",
                args.resume_checkpoint,
            ):
                raise ValueError("checkpoint is outside the approved run")
            snapshot = snapshot_download(
                launch["model_repo"],
                revision=args.resume_revision,
                allow_patterns=[args.resume_checkpoint + "/*"],
                local_dir=output / "downloaded-resume",
            )
            resume = Path(snapshot) / args.resume_checkpoint
            cursor = validate_checkpoint(resume, args.resume_manifest_sha256, identity, schedule)
            checkpoint_refs[f"train/checkpoint-{cursor['global_step']}"] = {
                "repo": launch["model_repo"],
                "revision": args.resume_revision,
                "path": args.resume_checkpoint,
                "manifest_sha256": args.resume_manifest_sha256,
            }
            atomic_json(output / "durable-checkpoints.json", checkpoint_refs)
        execution = execute_schedule(
            trainer, identity, schedule, resume, args.resume_manifest_sha256
        )
        final = adapter_state(model)
        if equal_adapters(before, final):
            raise ValueError("training did not change the adapter")
        final_checkpoint = execution["final_checkpoint"]
        final_reference = checkpoint_refs[
            f"{('reference' if args.mode == 'compatibility' else 'train')}/{final_checkpoint.name}"
        ]
        del before, trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        reloaded = PeftModel.from_pretrained(base_model(), final_checkpoint)
        if not equal_adapters(final, adapter_state(reloaded)):
            raise ValueError("saved adapter did not reload exactly")
        del reloaded
        gc.collect()
        torch.cuda.empty_cache()

        if args.mode == "compatibility":
            checkpoint = output / "reference/checkpoint-1"
            validate_checkpoint(
                checkpoint, file_hash(checkpoint / "checkpoint-manifest.json"), identity, schedule
            )
            model = adapted_model()
            trainer = make_trainer(model, "resumed")
            trainer.train(resume_from_checkpoint=str(checkpoint))
            if not equal_adapters(final, adapter_state(model)):
                raise ValueError("GPU optimizer/scheduler/RNG/data-cursor resume is not exact")
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()

        result = {
            "binding": binding,
            "passed": True,
            "mode": args.mode,
            "official_weights_loaded": True,
            "gpu_training_verified": execution["optimizer_updates_this_invocation"] > 0,
            "real_masks_verified": True,
            "shared_layers_verified": True,
            "adapter_save_reload_verified": True,
            "exact_resume_verified": args.mode == "compatibility",
            "padded_8k_forward_backward_verified": args.mode == "compatibility",
            "durable_checkpoints_verified": bool(checkpoint_refs),
            "student_trained": args.mode == "train",
            "promotion_proven": False,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "unique_examples": len(rows),
            "scheduled_draws": len(schedule),
            "checkpoint_refs": checkpoint_refs,
            "final_checkpoint_ref": final_reference,
            "optimizer_updates_this_invocation": execution["optimizer_updates_this_invocation"],
            "recovered_completed_checkpoint": execution["recovered_completed_checkpoint"],
        }
        report_dir = output / "result"
        report_dir.mkdir()
        atomic_json(report_dir / "report.json", result)
        atomic_json(report_dir / "run-identity.json", identity)
        reference = publisher.publish(report_dir, "result", "report.json")
        print(json.dumps({"result": result, "durable_report": reference}, indent=2))
    finally:
        trackio.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compatibility", "train"), required=True)
    parser.add_argument("--bundle-repo", required=True)
    parser.add_argument("--bundle-revision", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("nanbeige-training-output"))
    parser.add_argument("--compatibility-report", type=Path)
    parser.add_argument("--compatibility-sha256")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--resume-revision")
    parser.add_argument("--resume-manifest-sha256")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.bundle_revision):
        raise ValueError("bundle revision must be an immutable Hub commit")
    if args.resume_revision and not re.fullmatch(r"[0-9a-f]{40}", args.resume_revision):
        raise ValueError("resume revision must be an immutable Hub commit")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import torch
    from huggingface_hub import HfApi, snapshot_download

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "require one BF16-capable CUDA GPU; never train the official weights on Mac"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if shutil.disk_usage(Path.cwd()).free < 10 * 1024**3:
        raise RuntimeError("preserve at least 10 GiB free disk")
    api = HfApi()
    require_private(api, args.bundle_repo, "dataset")
    bundle = Path(
        snapshot_download(args.bundle_repo, repo_type="dataset", revision=args.bundle_revision)
    )
    preflight = load_preflight(bundle / "validate_nanbeige_training.py", args.preflight_sha256)
    manifest, examples, rows = preflight.validate_bundle(
        bundle, args.manifest_sha256, require_approval=True
    )
    if args.compatibility_report and not args.compatibility_report.is_absolute():
        args.compatibility_report = bundle / args.compatibility_report
    launch = check_launch(manifest, args.mode, file_hash(__file__))
    for repo, kind in (
        (launch["model_repo"], "model"),
        (launch["trackio_space"], "space"),
        (launch["trackio_dataset"], "dataset"),
    ):
        require_private(api, repo, kind)
    model_dir = Path(snapshot_download(MODEL, revision=REVISION))
    if shutil.disk_usage(Path.cwd()).free < 10 * 1024**3:
        raise RuntimeError("model download left less than the required 10 GiB disk reserve")
    for name, expected in {**preflight.MODEL_CODE, **preflight.TOKENIZER_HASHES, **WEIGHTS}.items():
        if file_hash(model_dir / name) != expected:
            raise ValueError(f"official model artifact changed: {name}")
    train(args, preflight, manifest, examples, rows, model_dir, launch, api)


if __name__ == "__main__":
    main()
