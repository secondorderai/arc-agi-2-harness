"""No GPU, model weights, remote writes or paid jobs occur in these unit tests."""

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "nanbeige_gpu_job", Path(__file__).parents[1] / "scripts/train_nanbeige_symbolic.py"
)
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


def test_import_is_inert():
    assert callable(job.train) and callable(job.main)


def launch_manifest():
    return {
        "gpu_launches": {
            "compatibility": {
                "compute_approval_ref": "synthetic-test-not-real-approval",
                "output_use_permission_ref": "synthetic-test-not-real-permission",
                "model_repo": "fixture/private-model",
                "trackio_space": "fixture/private-space",
                "trackio_dataset": "fixture/private-metrics",
                "run_id": "fixture-01",
                "hardware": "fixture-gpu",
                "script_sha256": "fixture-hash",
                "timeout_seconds": 7200,
            }
        }
    }


@pytest.mark.parametrize("field", list(launch_manifest()["gpu_launches"]["compatibility"]))
def test_missing_launch_field_rejected(field):
    manifest = launch_manifest()
    assert job.check_launch(manifest, "compatibility", "fixture-hash")["run_id"] == "fixture-01"
    del manifest["gpu_launches"]["compatibility"][field]
    with pytest.raises(ValueError):
        job.check_launch(manifest, "compatibility", "fixture-hash")


@pytest.mark.parametrize(
    "key,value",
    [
        ("run_id", "../overwrite"),
        ("model_repo", "/broad/path"),
        ("timeout_seconds", -1),
        ("script_sha256", "changed"),
    ],
)
def test_changed_launch_rejected(key, value):
    manifest = launch_manifest()
    manifest["gpu_launches"]["compatibility"][key] = value
    with pytest.raises(ValueError):
        job.check_launch(manifest, "compatibility", "fixture-hash")


@pytest.fixture
def checkpoint(tmp_path):
    folder = tmp_path / "checkpoint-1"
    folder.mkdir()
    for name in job.CHECKPOINT_REQUIRED - {"run-identity.json", "data-cursor.json"}:
        (folder / name).write_text("synthetic fixture, not real model state")
    (folder / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    (folder / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": job.MODEL,
                "revision": job.REVISION,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "target_modules": job.TARGETS,
                "bias": "none",
                "modules_to_save": None,
            }
        )
    )
    identity = {"model": job.MODEL, "fixture_only": True}
    schedule = [{"row": i, "logical_epoch": 0} for i in range(32)]
    checksum = job.seal_checkpoint(folder, identity, schedule, 1)
    return folder, checksum, identity, schedule


def test_checkpoint_cursor_hashes_and_resume(checkpoint):
    folder, checksum, identity, schedule = checkpoint
    cursor = job.validate_checkpoint(folder, checksum, identity, schedule)
    assert cursor["next_draw"] == 16 and cursor["total_draws"] == 32
    with pytest.raises(ValueError, match="identity"):
        job.validate_checkpoint(folder, checksum, {**identity, "changed": True}, schedule)
    with pytest.raises(ValueError, match="cursor"):
        job.validate_checkpoint(folder, checksum, identity, list(reversed(schedule)))


@pytest.mark.parametrize(
    "name", ["optimizer.pt", "scheduler.pt", "rng_state.pth", "data-cursor.json"]
)
def test_checkpoint_tamper_rejected(checkpoint, name):
    folder, checksum, identity, schedule = checkpoint
    (folder / name).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        job.validate_checkpoint(folder, checksum, identity, schedule)


def test_seal_requires_complete_optimizer_state(tmp_path):
    (tmp_path / "adapter_model.safetensors").write_text("fixture")
    with pytest.raises(ValueError, match="incomplete"):
        job.seal_checkpoint(tmp_path, {}, [{"row": 0}] * 32, 1)


def test_checkpoint_path_traversal_rejected(checkpoint):
    folder, _, identity, schedule = checkpoint
    manifest = job.read_json(folder / "checkpoint-manifest.json")
    manifest["files"]["../outside"] = "fixture"
    (folder / "checkpoint-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="member"):
        job.validate_checkpoint(
            folder, job.file_hash(folder / "checkpoint-manifest.json"), identity, schedule
        )


def compatibility_report():
    report = {"passed": True, "binding": {"fixture": True}, "mode": "compatibility"}
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
        report[key] = True
    return report


@pytest.mark.parametrize("field", [k for k in compatibility_report() if k != "binding"])
def test_incomplete_gpu_evidence_rejected(field):
    report = compatibility_report()
    job.check_compatibility(report, {"fixture": True})
    report[field] = False
    with pytest.raises(ValueError):
        job.check_compatibility(report, {"fixture": True})


def test_gpu_evidence_bound_to_actual_training_protocol():
    with pytest.raises(ValueError, match="match"):
        job.check_compatibility(compatibility_report(), {"different_dataset_or_gpu": True})


def test_wrong_base_adapter_rejected(checkpoint):
    folder, _, _, _ = checkpoint
    config = job.read_json(folder / "adapter_config.json")
    job.validate_adapter_config(config)
    config["base_model_name_or_path"] = "another-model"
    with pytest.raises(ValueError, match="pinned Nanbeige"):
        job.validate_adapter_config(config)


@pytest.fixture
def finished_checkpoint(checkpoint, tmp_path):
    source, _, identity, schedule = checkpoint
    folder = tmp_path / "saved/checkpoint-2"
    shutil.copytree(source, folder, ignore=shutil.ignore_patterns("checkpoint-manifest.json"))
    (folder / "trainer_state.json").write_text(json.dumps({"global_step": 2}))
    return folder, job.seal_checkpoint(folder, identity, schedule, 2), identity, schedule


class FixtureTrainerState(SimpleNamespace):
    @classmethod
    def load_from_json(cls, path):
        return cls(**job.read_json(path))


def fixture_trainer(output):
    def unexpected(*args, **kwargs):
        pytest.fail("unexpected training or adapter loading")

    return SimpleNamespace(
        state=FixtureTrainerState(global_step=0),
        args=SimpleNamespace(output_dir=str(output)),
        model=SimpleNamespace(load_adapter=unexpected),
        train=unexpected,
    )


def test_finished_resume_never_retrains_or_creates_new_checkpoint(finished_checkpoint, tmp_path):
    folder, checksum, identity, schedule = finished_checkpoint
    trainer = fixture_trainer(tmp_path / "new-run")
    loads = []

    def load(path, **kwargs):
        loads.append((path, kwargs))
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    trainer.model.load_adapter = load
    before = {p.name: job.file_hash(p) for p in folder.iterdir()}
    result = job.execute_schedule(trainer, identity, schedule, folder, checksum)
    assert result == {
        "final_checkpoint": folder,
        "optimizer_updates_this_invocation": 0,
        "recovered_completed_checkpoint": True,
    }
    assert trainer.state.global_step == 2
    assert loads == [
        (
            str(folder),
            {"adapter_name": "default", "is_trainable": True, "local_files_only": True},
        )
    ]
    assert before == {p.name: job.file_hash(p) for p in folder.iterdir()}
    assert not Path(trainer.args.output_dir).exists()


@pytest.mark.parametrize("resuming", [False, True])
def test_unfinished_schedule_uses_trainer_and_new_durable_checkpoint(
    checkpoint, finished_checkpoint, tmp_path, resuming
):
    source, source_hash, identity, schedule = checkpoint
    final, _, _, _ = finished_checkpoint
    trainer = fixture_trainer(tmp_path / "new-run")
    calls = []

    def train(**kwargs):
        calls.append(kwargs)
        trainer.state.global_step = 2
        shutil.copytree(final, Path(trainer.args.output_dir) / final.name)

    trainer.train = train
    result = job.execute_schedule(
        trainer, identity, schedule, source if resuming else None, source_hash if resuming else None
    )
    assert calls == [{"resume_from_checkpoint": str(source) if resuming else None}]
    assert result["optimizer_updates_this_invocation"] == (1 if resuming else 2)
    assert result["recovered_completed_checkpoint"] is False
    assert result["final_checkpoint"] == Path(trainer.args.output_dir) / "checkpoint-2"


@pytest.mark.parametrize("fault", ["missing_keys", "unexpected_keys", "step"])
def test_finished_adapter_restore_requires_exact_state(finished_checkpoint, tmp_path, fault):
    folder, _, _, _ = finished_checkpoint
    trainer = fixture_trainer(tmp_path / "new-run")
    if fault != "step":
        result = {"missing_keys": [], "unexpected_keys": []}
        result[fault] = ["lora_bad_key"]
        trainer.model.load_adapter = lambda *args, **kwargs: SimpleNamespace(**result)
    with pytest.raises(ValueError, match="step|completely"):
        job.restore_completed_adapter(trainer, folder, 3 if fault == "step" else 2)
    assert trainer.state.global_step == 0


def test_finished_resume_checks_optimizer_hash_before_loading(finished_checkpoint, tmp_path):
    folder, checksum, identity, schedule = finished_checkpoint
    (folder / "optimizer.pt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        job.execute_schedule(
            fixture_trainer(tmp_path / "new-run"), identity, schedule, folder, checksum
        )


def test_finished_resume_path_must_match_cursor(finished_checkpoint, tmp_path):
    folder, checksum, identity, schedule = finished_checkpoint
    renamed = folder.with_name("checkpoint-3")
    folder.rename(renamed)
    with pytest.raises(ValueError, match="path.*cursor"):
        job.execute_schedule(
            fixture_trainer(tmp_path / "new-run"), identity, schedule, renamed, checksum
        )


@pytest.mark.parametrize("final_step", [1, 2])
def test_finished_state_without_durable_output_is_not_success(checkpoint, tmp_path, final_step):
    _, _, identity, schedule = checkpoint
    trainer = fixture_trainer(tmp_path / "new-run")
    trainer.train = lambda **kwargs: setattr(trainer.state, "global_step", final_step)
    with pytest.raises(ValueError if final_step == 1 else FileNotFoundError):
        job.execute_schedule(trainer, identity, schedule)


def test_empty_schedule_cannot_complete(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        job.execute_schedule(fixture_trainer(tmp_path / "new-run"), {}, [])


def test_publisher_requires_private_append_only_commits(tmp_path, checkpoint):
    folder, _, _, _ = checkpoint

    class Api:
        private = True
        uploaded = 0
        files = []

        def repo_info(self, *args, **kwargs):
            return SimpleNamespace(private=self.private, sha="old-immutable-commit")

        def list_repo_files(self, *args, **kwargs):
            assert kwargs["revision"] == "old-immutable-commit"
            return self.files

        def upload_folder(self, **kwargs):
            assert kwargs["parent_commit"] == "old-immutable-commit"
            assert kwargs["path_in_repo"] == "runs/fixture/checkpoint-1"
            self.uploaded += 1
            shutil.copytree(kwargs["folder_path"], tmp_path / "remote")
            self.files = ["runs/fixture/checkpoint-1/checkpoint-manifest.json"]
            return SimpleNamespace(oid="new-immutable-commit")

    def download(**kwargs):
        assert kwargs["revision"] == "new-immutable-commit"
        assert kwargs["force_download"] is True
        return tmp_path / "remote/checkpoint-manifest.json"

    api = Api()
    publisher = job.Publisher(api, "fixture/private", "runs/fixture", download)
    api.private = False
    with pytest.raises(ValueError, match="private"):
        publisher.publish(folder, "checkpoint-1", "checkpoint-manifest.json")
    assert api.uploaded == 0
    api.private = True
    ref = publisher.publish(folder, "checkpoint-1", "checkpoint-manifest.json")
    assert ref["revision"] == "new-immutable-commit" and api.uploaded == 1
    with pytest.raises(ValueError, match="overwrite"):
        publisher.publish(folder, "checkpoint-1", "checkpoint-manifest.json")
    assert api.uploaded == 1


def test_mac_rejected_before_hub_access(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )

    def unexpected(*args, **kwargs):
        pytest.fail("non-CUDA execution must not access Hub or download weights")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=unexpected, snapshot_download=unexpected),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "compatibility",
            "--bundle-repo",
            "fixture/private",
            "--bundle-revision",
            "a" * 40,
            "--manifest-sha256",
            "b" * 64,
            "--preflight-sha256",
            "c" * 64,
        ],
    )
    with pytest.raises(RuntimeError, match="never train.*Mac"):
        job.main()
