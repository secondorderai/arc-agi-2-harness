"""A compact object-centric world model and symbolic executor interface.

This is deliberately a small, dependency-light experiment rather than a claim to
reproduce Loop-OWM.  It uses NumPy so that the data path and CPU smoke tests work in
the project environment, which currently has no PyTorch and no usable Metal device
in headless runs.  The model has the same high-level decomposition we want to test:

``grid encoder -> object slots -> demonstration summary -> recurrent transitions
-> grid decoder + symbolic operation head``

The encoder and recurrent core are frozen during the tiny built-in trainer; the
trainer optimizes the decoder, input colour skip map and auxiliary operation head.
That makes the experiment fast and honest.  A future MLX/Torch backend can reuse
the numerical batch contract in :mod:`arc_agent.world_model_data` without changing
the evaluator or checkpoint format.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from arc_agent.models import Grid
from arc_agent.world_model_data import (
    OBJECT_FEATURE_DIM,
    WorldModelBatch,
    WorldModelExample,
    collate_world_model_examples,
    iter_world_model_batches,
)

PROGRAM_OPS = (
    "identity",
    "recolor",
    "rotate",
    "reflect",
    "crop",
    "tile",
    "translate",
    "object_transform",
    "relation_follow",
    "unknown",
)

try:  # MLX is optional at import time and requires Apple Silicon at runtime.
    import mlx.core as _mx
    import mlx.nn as _mlx_nn
    import mlx.optimizers as _mlx_optimizers
except (ImportError, RuntimeError):  # pragma: no cover - depends on installation
    _mx = None
    _mlx_nn = None
    _mlx_optimizers = None


def mlx_package_available() -> bool:
    """Return whether the optional MLX Python package imported successfully."""
    return _mx is not None and _mlx_nn is not None and _mlx_optimizers is not None


def mlx_runtime_available() -> bool:
    """Probe whether MLX can actually allocate/evaluate on its current device.

    Importing MLX is not sufficient: in headless macOS sessions the package can
    import and report a GPU while the first allocation fails with ``No Metal device
    available``.  The probe deliberately performs one tiny evaluation and returns
    ``False`` for that case instead of leaking a backend-specific exception.
    """
    if not mlx_package_available():
        return False
    try:
        probe = _mx.array([0], dtype=_mx.int32)
        _mx.eval(probe)
    except (RuntimeError, ValueError, TypeError):
        return False
    return True


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), 1e-9)


def _xavier(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) < 2:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)
    scale = np.sqrt(2.0 / (shape[0] + shape[-1]))
    return (rng.standard_normal(shape) * scale).astype(np.float32)


@dataclass(frozen=True)
class WorldModelConfig:
    """Capacity and reproducibility settings for :class:`HybridWorldModel`."""

    max_grid_size: int = 30
    hidden_size: int = 48
    num_slots: int = 8
    recurrent_steps: int = 4
    max_objects: int = 16
    num_colors: int = 10
    num_program_args: int = 9
    seed: int = 0
    program_loss_weight: float = 0.2
    ignore_unknown_program: bool = True
    balance_program_classes: bool = True

    def __post_init__(self) -> None:
        if self.max_grid_size < 1 or self.max_grid_size > 30:
            raise ValueError("max_grid_size must be in [1, 30]")
        if self.hidden_size < 4:
            raise ValueError("hidden_size must be at least 4")
        if self.num_slots < 1 or self.recurrent_steps < 1 or self.max_objects < 1:
            raise ValueError("num_slots, recurrent_steps and max_objects must be positive")
        if self.num_colors != 10:
            raise ValueError("ARC colour vocabulary must contain exactly 10 colours")
        if self.num_program_args < 1:
            raise ValueError("num_program_args must be positive")
        if not 0.0 <= self.program_loss_weight <= 1.0:
            raise ValueError("program_loss_weight must be in [0, 1]")


def _program_sample_weights(targets: np.ndarray, config: WorldModelConfig) -> np.ndarray:
    """Return normalized inverse-frequency weights for confident program labels."""

    values = np.asarray(targets, dtype=np.int64).reshape(-1)
    weights = np.ones(len(values), dtype=np.float32)
    if config.ignore_unknown_program:
        weights[values == PROGRAM_OPS.index("unknown")] = 0.0
    valid = weights > 0
    if config.balance_program_classes and np.any(valid):
        counts = np.bincount(values[valid], minlength=len(PROGRAM_OPS))
        weights[valid] = np.asarray(
            [1.0 / max(int(counts[value]), 1) for value in values[valid]],
            dtype=np.float32,
        )
    total = float(weights.sum())
    return weights / total if total > 0 else weights


def _program_metrics(predictions: Sequence[int], targets: Sequence[int]) -> dict[str, float]:
    predicted = np.asarray(predictions, dtype=np.int64)
    wanted = np.asarray(targets, dtype=np.int64)
    accuracy = float(np.mean(predicted == wanted)) if len(wanted) else 0.0
    known = wanted != PROGRAM_OPS.index("unknown")
    known_accuracy = float(np.mean(predicted[known] == wanted[known])) if np.any(known) else 0.0
    labels = sorted(set(wanted[known].tolist()))
    macro = (
        float(np.mean([np.mean(predicted[wanted == label] == label) for label in labels]))
        if labels
        else 0.0
    )
    return {
        "program_accuracy": accuracy,
        "known_program_accuracy": known_accuracy,
        "program_macro_accuracy": macro,
    }


@dataclass(frozen=True)
class ProgramCandidate:
    op: str
    argument: int
    probability: float


@dataclass
class WorldModelOutput:
    """Raw logits from a batched forward pass."""

    grid_logits: np.ndarray  # [batch, height, width, 10]
    program_logits: np.ndarray  # [batch, num_ops]
    argument_logits: np.ndarray  # [batch, num_args]
    hidden: np.ndarray


@dataclass
class WorldModelPrediction:
    """Decoded single-example prediction while retaining all model logits."""

    grid: Grid
    grid_logits: np.ndarray
    program_logits: np.ndarray
    argument_logits: np.ndarray
    program_candidates: tuple[ProgramCandidate, ...]


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    grid_loss: float
    program_loss: float
    cell_accuracy: float
    exact_accuracy: float


@dataclass
class TrainingHistory:
    epochs: list[EpochMetrics]

    @property
    def final_loss(self) -> float | None:
        return self.epochs[-1].loss if self.epochs else None


class HybridWorldModel:
    """Small object-slot/recurrent model with a deterministic NumPy backend.

    The model does not hide a language model or call a remote service.  Its
    ``predict`` method is safe to call from a verifier: it only returns a grid and
    symbolic hypotheses, and all dimensions are validated before computation.
    """

    format_version = 1

    def __init__(self, config: WorldModelConfig | None = None) -> None:
        self.config = config or WorldModelConfig()
        rng = np.random.default_rng(self.config.seed)
        d = self.config.hidden_size
        self.params: dict[str, np.ndarray] = {
            # Colour and relative position are the low-level visual representation.
            "color_embedding": _xavier(rng, (10, d)),
            "row_embedding": _xavier(rng, (self.config.max_grid_size, d)),
            "column_embedding": _xavier(rng, (self.config.max_grid_size, d)),
            # Object slot attention uses learned prototypes as slot queries.
            "object_projection": _xavier(rng, (OBJECT_FEATURE_DIM, d)),
            "object_bias": np.zeros(d, dtype=np.float32),
            "slot_queries": _xavier(rng, (self.config.num_slots, d)),
            # Query and demonstration paths use separate projections but shared grid
            # and object encoders, which keeps the task summary comparable.
            "query_projection": _xavier(rng, (2 * d, d)),
            "query_bias": np.zeros(d, dtype=np.float32),
            "demo_projection": _xavier(rng, (3 * d, d)),
            "demo_bias": np.zeros(d, dtype=np.float32),
            "initial_projection": _xavier(rng, (2 * d, d)),
            "initial_bias": np.zeros(d, dtype=np.float32),
            # Shared looped transition core.
            "transition_input": _xavier(rng, (d, d)),
            "transition_hidden": _xavier(rng, (d, d)),
            "transition_bias": np.zeros(d, dtype=np.float32),
            # Grid decoder and a colour-preserving residual path.
            "decoder_state": _xavier(rng, (d, d)),
            "decoder_bias": np.zeros(d, dtype=np.float32),
            "decoder_color": _xavier(rng, (d, 10)),
            "decoder_color_bias": np.zeros(10, dtype=np.float32),
            "skip_logits": (3.0 * np.eye(10, dtype=np.float32)),
            # Auxiliary symbolic operation/argument heads.
            "program_head": _xavier(rng, (d, len(PROGRAM_OPS))),
            "program_bias": np.zeros(len(PROGRAM_OPS), dtype=np.float32),
            "argument_head": _xavier(rng, (d, self.config.num_program_args)),
            "argument_bias": np.zeros(self.config.num_program_args, dtype=np.float32),
        }
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        d = self.config.hidden_size
        required_shapes = {
            "color_embedding": (10, d),
            "row_embedding": (self.config.max_grid_size, d),
            "column_embedding": (self.config.max_grid_size, d),
            "object_projection": (OBJECT_FEATURE_DIM, d),
            "slot_queries": (self.config.num_slots, d),
            "query_projection": (2 * d, d),
            "demo_projection": (3 * d, d),
            "initial_projection": (2 * d, d),
            "transition_input": (d, d),
            "transition_hidden": (d, d),
            "decoder_state": (d, d),
            "decoder_color": (d, 10),
            "skip_logits": (10, 10),
            "program_head": (d, len(PROGRAM_OPS)),
            "argument_head": (d, self.config.num_program_args),
        }
        for name, shape in required_shapes.items():
            if name not in self.params or self.params[name].shape != shape:
                raise ValueError(f"invalid parameter shape for {name}: expected {shape}")

    def _check_grid_size(self, height: int, width: int) -> None:
        if height > self.config.max_grid_size or width > self.config.max_grid_size:
            raise ValueError(
                f"grid {height}x{width} exceeds max_grid_size={self.config.max_grid_size}"
            )

    def _grid_pool(self, grids: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if grids.ndim != 3 or masks.shape != grids.shape:
            raise ValueError("grids must be [batch,height,width] and masks must match")
        batch, height, width = grids.shape
        self._check_grid_size(height, width)
        colours = np.clip(grids, 0, 9).astype(np.int64)
        features = self.params["color_embedding"][colours]
        features = features + self.params["row_embedding"][None, :height, None, :]
        features = features + self.params["column_embedding"][None, None, :width, :]
        weights = masks.astype(np.float32)[..., None]
        return (features * weights).sum(axis=(1, 2)) / np.maximum(weights.sum(axis=(1, 2)), 1.0)

    def _object_pool(self, objects: np.ndarray, masks: np.ndarray) -> np.ndarray:
        if objects.ndim != 3 or masks.shape != objects.shape[:2]:
            raise ValueError("objects must be [batch,objects,features] and masks must match")
        projected = np.tanh(objects @ self.params["object_projection"] + self.params["object_bias"])
        valid = masks.astype(np.float32)
        scores = np.einsum("bod,sd->bos", projected, self.params["slot_queries"])
        scores = np.where(valid[..., None] > 0, scores, -1e9)
        weights = _softmax(scores, axis=1)
        slots = np.einsum("bos,bod->bsd", weights, projected)
        has_object = (valid.sum(axis=1) > 0).astype(np.float32)[:, None]
        return slots.mean(axis=1) * has_object

    def _encode_context(
        self,
        grids: np.ndarray,
        masks: np.ndarray,
        objects: np.ndarray,
        object_masks: np.ndarray,
    ) -> np.ndarray:
        grid_pool = self._grid_pool(grids, masks)
        object_pool = self._object_pool(objects, object_masks)
        return np.tanh(
            np.concatenate([grid_pool, object_pool], axis=-1) @ self.params["query_projection"]
            + self.params["query_bias"]
        )

    def _demo_summary(self, batch: WorldModelBatch) -> np.ndarray:
        batch_size, demo_count, height, width = batch.demo_inputs.shape
        flat_count = batch_size * demo_count
        input_masks = batch.demo_input_masks
        output_masks = batch.demo_output_masks
        if input_masks is None or output_masks is None:
            input_masks = output_masks = batch.demo_masks
        input_object_masks = batch.demo_object_mask * np.any(
            np.abs(batch.demo_input_objects) > 0, axis=-1
        )
        output_object_masks = batch.demo_object_mask * np.any(
            np.abs(batch.demo_output_objects) > 0, axis=-1
        )
        input_context = self._encode_context(
            batch.demo_inputs.reshape(flat_count, height, width),
            input_masks.reshape(flat_count, height, width),
            batch.demo_input_objects.reshape(flat_count, batch.demo_input_objects.shape[2], -1),
            input_object_masks.reshape(flat_count, input_object_masks.shape[2]),
        )
        output_context = self._encode_context(
            batch.demo_outputs.reshape(flat_count, height, width),
            output_masks.reshape(flat_count, height, width),
            batch.demo_output_objects.reshape(flat_count, batch.demo_output_objects.shape[2], -1),
            output_object_masks.reshape(flat_count, output_object_masks.shape[2]),
        )
        pair_context = np.concatenate(
            [input_context, output_context, output_context - input_context], axis=-1
        )
        pair_context = np.tanh(
            pair_context @ self.params["demo_projection"] + self.params["demo_bias"]
        ).reshape(batch_size, demo_count, -1)
        weights = batch.demo_pair_mask.astype(np.float32)[..., None]
        return (pair_context * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)

    def _transition(self, query_context: np.ndarray, demo_context: np.ndarray) -> np.ndarray:
        context = np.concatenate([query_context, demo_context], axis=-1)
        state = np.tanh(
            context @ self.params["initial_projection"] + self.params["initial_bias"]
        )
        for _ in range(self.config.recurrent_steps):
            state = np.tanh(
                context[:, : self.config.hidden_size] @ self.params["transition_input"]
                + state @ self.params["transition_hidden"]
                + self.params["transition_bias"]
            )
        return state

    def forward_batch(self, batch: WorldModelBatch) -> WorldModelOutput:
        """Run all object, demonstration, recurrent and decoder stages."""
        query_context = self._encode_context(
            batch.query_inputs,
            batch.query_mask,
            batch.query_objects,
            batch.query_object_mask,
        )
        demo_context = self._demo_summary(batch)
        state = self._transition(query_context, demo_context)
        batch_size, height, width = batch.query_inputs.shape
        state_features = np.tanh(
            state @ self.params["decoder_state"] + self.params["decoder_bias"]
        )
        positions = self.params["row_embedding"][None, :height, None, :] + self.params[
            "column_embedding"
        ][None, None, :width, :]
        cells = state_features[:, None, None, :] + positions
        logits = np.einsum("bhwd,dc->bhwc", cells, self.params["decoder_color"])
        logits += self.params["decoder_color_bias"]
        query_one_hot = np.eye(10, dtype=np.float32)[np.clip(batch.query_inputs, 0, 9)]
        logits += np.einsum("bhwk,kc->bhwc", query_one_hot, self.params["skip_logits"])
        return WorldModelOutput(
            grid_logits=logits.astype(np.float32),
            program_logits=(
                state @ self.params["program_head"] + self.params["program_bias"]
            ).astype(np.float32),
            argument_logits=(
                state @ self.params["argument_head"] + self.params["argument_bias"]
            ).astype(np.float32),
            hidden=state.astype(np.float32),
        )

    @staticmethod
    def _grid_loss(
        logits: np.ndarray, targets: np.ndarray, mask: np.ndarray
    ) -> tuple[float, np.ndarray]:
        probabilities = _softmax(logits, axis=-1)
        safe_targets = np.clip(targets, 0, 9).astype(np.int64)
        target_probabilities = np.take_along_axis(
            probabilities, safe_targets[..., None], axis=-1
        )[..., 0]
        valid = mask.astype(np.float32)
        count = max(float(valid.sum()), 1.0)
        loss = float((-np.log(np.maximum(target_probabilities, 1e-8)) * valid).sum() / count)
        gradient = probabilities
        gradient -= np.eye(10, dtype=np.float32)[safe_targets]
        gradient *= valid[..., None] / count
        return loss, gradient.astype(np.float32)

    @staticmethod
    def _classification_loss(
        logits: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
        probabilities = _softmax(logits, axis=-1)
        safe_targets = np.clip(targets, 0, logits.shape[-1] - 1).astype(np.int64)
        weights = (
            np.full(len(targets), 1.0 / max(len(targets), 1), dtype=np.float32)
            if sample_weights is None
            else np.asarray(sample_weights, dtype=np.float32)
        )
        loss = float(
            (
                -np.log(np.maximum(probabilities[np.arange(len(targets)), safe_targets], 1e-8))
                * weights
            ).sum()
        )
        gradient = probabilities
        gradient[np.arange(len(targets)), safe_targets] -= 1.0
        return loss, (gradient * weights[:, None]).astype(np.float32)

    def train_step(
        self, batch: WorldModelBatch, *, learning_rate: float = 1e-2
    ) -> dict[str, float]:
        """Perform one deterministic decoder/head update.

        The object encoder and transition core remain fixed in this baseline.  This
        is intentional: it isolates whether a compact semantic representation can
        be useful before introducing expensive end-to-end MLX training.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        output = self.forward_batch(batch)
        grid_loss, grid_gradient = self._grid_loss(
            output.grid_logits, batch.targets, batch.target_mask
        )
        program_loss, program_gradient = self._classification_loss(
            output.program_logits,
            batch.program_targets,
            _program_sample_weights(batch.program_targets, self.config),
        )
        valid_count = max(float(batch.target_mask.sum()), 1.0)
        # Recompute the decoder features needed for the exact analytic gradients.
        state_features = np.tanh(
            output.hidden @ self.params["decoder_state"] + self.params["decoder_bias"]
        )
        height, width = batch.query_inputs.shape[1:]
        positions = self.params["row_embedding"][None, :height, None, :] + self.params[
            "column_embedding"
        ][None, None, :width, :]
        cells = state_features[:, None, None, :] + positions
        query_one_hot = np.eye(10, dtype=np.float32)[np.clip(batch.query_inputs, 0, 9)]
        self.params["decoder_color"] -= learning_rate * np.einsum(
            "bhwd,bhwc->dc", cells, grid_gradient
        )
        self.params["decoder_color_bias"] -= learning_rate * grid_gradient.sum(axis=(0, 1, 2))
        self.params["skip_logits"] -= learning_rate * np.einsum(
            "bhwk,bhwc->kc", query_one_hot, grid_gradient
        )
        d_state_features = np.einsum("bhwc,dc->bhwd", grid_gradient, self.params["decoder_color"])
        d_state_features = d_state_features.sum(axis=(1, 2))
        d_state_features *= 1.0 - state_features**2
        self.params["decoder_state"] -= learning_rate * (output.hidden.T @ d_state_features)
        self.params["decoder_bias"] -= learning_rate * d_state_features.sum(axis=0)
        self.params["program_head"] -= learning_rate * self.config.program_loss_weight * (
            output.hidden.T @ program_gradient
        )
        self.params["program_bias"] -= (
            learning_rate
            * self.config.program_loss_weight
            * program_gradient.sum(axis=0)
        )
        predictions = np.argmax(output.grid_logits, axis=-1)
        cell_accuracy = float(
            ((predictions == batch.targets) * batch.target_mask).sum() / valid_count
        )
        exact = []
        for index, shape in enumerate(batch.target_shapes):
            height_i, width_i = shape
            exact.append(
                np.array_equal(
                    predictions[index, :height_i, :width_i],
                    batch.targets[index, :height_i, :width_i],
                )
            )
        return {
            "loss": grid_loss + self.config.program_loss_weight * program_loss,
            "grid_loss": grid_loss,
            "program_loss": program_loss,
            "cell_accuracy": cell_accuracy,
            "exact_accuracy": float(np.mean(exact)),
        }

    def predict(
        self,
        query_grid: Grid,
        demonstrations: Sequence[tuple[Grid, Grid]],
        *,
        output_shape: tuple[int, int] | None = None,
        top_k_programs: int = 3,
    ) -> WorldModelPrediction:
        """Predict one query and return both grid and symbolic hypotheses."""
        if not demonstrations:
            raise ValueError("at least one labelled demonstration is required")
        if output_shape is None:
            output_shape = (len(query_grid), len(query_grid[0]))
        if len(output_shape) != 2 or not all(
            1 <= value <= self.config.max_grid_size for value in output_shape
        ):
            raise ValueError("output_shape must fit max_grid_size")
        placeholder = [[0] * output_shape[1] for _ in range(output_shape[0])]
        example = WorldModelExample(
            task_id="inference",
            query_input=query_grid,
            target_output=placeholder,
            demonstrations=tuple(demonstrations),
            program_label=0,
        )
        batch = collate_world_model_examples([example], max_objects=self.config.max_objects)
        output = self.forward_batch(batch)
        height, width = output_shape
        grid_logits = output.grid_logits[0, :height, :width]
        grid = np.argmax(grid_logits, axis=-1).astype(int).tolist()
        program_logits = output.program_logits[0]
        argument_logits = output.argument_logits[0]
        candidates = program_candidates_from_logits(
            program_logits, argument_logits, top_k=top_k_programs
        )
        return WorldModelPrediction(
            grid=grid,
            grid_logits=grid_logits,
            program_logits=program_logits,
            argument_logits=argument_logits,
            program_candidates=candidates,
        )

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Save a deterministic, portable NumPy checkpoint."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.format_version,
            "config": asdict(self.config),
            "metadata": metadata or {},
            "parameter_names": sorted(self.params),
        }
        arrays = {f"param::{name}": self.params[name] for name in sorted(self.params)}
        # Writing through a file object preserves the caller's requested extension.
        with target.open("wb") as handle:
            np.savez(handle, __metadata__=np.asarray(json.dumps(payload)), **arrays)
        return target

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> HybridWorldModel:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as archive:
            try:
                metadata = json.loads(str(archive["__metadata__"].item()))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid world-model checkpoint metadata") from exc
            if metadata.get("format_version") != cls.format_version:
                raise ValueError("unsupported world-model checkpoint format")
            model = cls(WorldModelConfig(**metadata["config"]))
            for name in metadata.get("parameter_names", []):
                key = f"param::{name}"
                if key not in archive:
                    raise ValueError(f"checkpoint is missing parameter {name}")
                model.params[name] = np.asarray(archive[key], dtype=np.float32)
            model._validate_parameters()
        return model


@dataclass
class MlxWorldModelOutput:
    """Raw MLX arrays returned by :class:`MlxHybridWorldModel`."""

    grid_logits: Any
    program_logits: Any
    argument_logits: Any
    hidden: Any


if _mlx_nn is not None:  # pragma: no cover - Metal is unavailable in CI/headless runs

    class _MlxNetwork(_mlx_nn.Module):
        """Trainable MLX counterpart to the NumPy reference architecture."""

        def __init__(self, config: WorldModelConfig) -> None:
            super().__init__()
            self.config = config
            d = config.hidden_size
            self.color_embedding = _mlx_nn.Embedding(10, d)
            self.row_embedding = _mx.random.normal((config.max_grid_size, d), scale=0.04)
            self.column_embedding = _mx.random.normal((config.max_grid_size, d), scale=0.04)
            self.object_projection = _mlx_nn.Linear(OBJECT_FEATURE_DIM, d)
            self.slot_queries = _mx.random.normal((config.num_slots, d), scale=0.04)
            self.query_projection = _mlx_nn.Linear(2 * d, d)
            self.demo_projection = _mlx_nn.Linear(3 * d, d)
            self.initial_projection = _mlx_nn.Linear(2 * d, d)
            self.transition = _mlx_nn.GRU(2 * d, d)
            self.decoder_state = _mlx_nn.Linear(d, d)
            self.decoder = _mlx_nn.Linear(d, 10)
            self.skip_logits = 3.0 * _mx.eye(10)
            self.program_head = _mlx_nn.Linear(d, len(PROGRAM_OPS))
            self.argument_head = _mlx_nn.Linear(d, config.num_program_args)

        def _grid_pool(self, grids: Any, masks: Any) -> Any:
            _, height, width = grids.shape
            features = self.color_embedding(grids)
            features = features + self.row_embedding[None, :height, None, :]
            features = features + self.column_embedding[None, None, :width, :]
            weights = masks[..., None]
            return _mx.sum(features * weights, axis=(1, 2)) / _mx.maximum(
                _mx.sum(weights, axis=(1, 2)), 1.0
            )

        def _object_pool(self, objects: Any, masks: Any) -> Any:
            projected = _mx.tanh(self.object_projection(objects))
            scores = _mx.matmul(projected, self.slot_queries.T)
            scores = _mx.where(masks[..., None] > 0, scores, -1e9)
            weights = _mx.softmax(scores, axis=1)
            slots = _mx.einsum("bos,bod->bsd", weights, projected)
            return _mx.mean(slots, axis=1) * (_mx.sum(masks, axis=1) > 0)[..., None]

        def _encode_context(
            self, grids: Any, masks: Any, objects: Any, object_masks: Any
        ) -> Any:
            return _mx.tanh(
                self.query_projection(
                    _mx.concatenate(
                        [self._grid_pool(grids, masks), self._object_pool(objects, object_masks)],
                        axis=-1,
                    )
                )
            )

        def __call__(
            self,
            query_inputs: Any,
            query_masks: Any,
            query_objects: Any,
            query_object_masks: Any,
            demo_inputs: Any,
            demo_outputs: Any,
            demo_masks: Any,
            demo_input_masks: Any,
            demo_output_masks: Any,
            demo_input_objects: Any,
            demo_output_objects: Any,
            demo_object_masks: Any,
            demo_pair_mask: Any,
        ) -> tuple[Any, Any, Any, Any]:
            batch_size, demo_count, height, width = demo_inputs.shape
            flattened = batch_size * demo_count
            input_context = self._encode_context(
                demo_inputs.reshape(flattened, height, width),
                demo_input_masks.reshape(flattened, height, width),
                demo_input_objects.reshape(flattened, demo_input_objects.shape[2], -1),
                (
                    demo_object_masks
                    * (_mx.sum(_mx.abs(demo_input_objects), axis=-1) > 0)
                ).reshape(flattened, demo_object_masks.shape[2]),
            )
            output_context = self._encode_context(
                demo_outputs.reshape(flattened, height, width),
                demo_output_masks.reshape(flattened, height, width),
                demo_output_objects.reshape(flattened, demo_output_objects.shape[2], -1),
                (
                    demo_object_masks
                    * (_mx.sum(_mx.abs(demo_output_objects), axis=-1) > 0)
                ).reshape(flattened, demo_object_masks.shape[2]),
            )
            pair_context = _mx.concatenate(
                [input_context, output_context, output_context - input_context], axis=-1
            )
            pair_context = _mx.tanh(self.demo_projection(pair_context)).reshape(
                batch_size, demo_count, -1
            )
            pair_weights = demo_pair_mask[..., None]
            demo_summary = _mx.sum(pair_context * pair_weights, axis=1) / _mx.maximum(
                _mx.sum(pair_weights, axis=1), 1.0
            )
            query_context = self._encode_context(
                query_inputs, query_masks, query_objects, query_object_masks
            )
            context = _mx.concatenate([query_context, demo_summary], axis=-1)
            state = _mx.tanh(self.initial_projection(context))
            transition_input = _mx.broadcast_to(
                context[:, None, :], (batch_size, self.config.recurrent_steps, context.shape[-1])
            )
            state = self.transition(transition_input, state)[:, -1, :]
            state_features = _mx.tanh(self.decoder_state(state))
            positions = self.row_embedding[None, :height, None, :] + self.column_embedding[
                None, None, :width, :
            ]
            cells = state_features[:, None, None, :] + positions
            logits = self.decoder(cells)
            one_hot = _mx.eye(10)[_mx.clip(query_inputs, 0, 9)]
            logits = logits + _mx.einsum("bhwk,kc->bhwc", one_hot, self.skip_logits)
            return logits, self.program_head(state), self.argument_head(state), state

else:  # pragma: no cover - exercised only when MLX is not installed

    class _MlxNetwork:  # type: ignore[no-redef]
        def __init__(self, config: WorldModelConfig) -> None:
            del config
            raise RuntimeError(
                "MLX backend is unavailable. Install mlx on Apple Silicon or use "
                "HybridWorldModel's NumPy reference backend."
            )


class MlxHybridWorldModel:
    """End-to-end trainable compact model for Apple Silicon MLX.

    Unlike the NumPy reference trainer, this path differentiates through the
    object slots, demonstration summary and all recurrent transition loops.  It is
    intentionally opt-in because MLX requires a working Apple GPU runtime; import
    and dataset construction remain safe on CPU-only machines.
    """

    def __init__(
        self,
        config: WorldModelConfig | None = None,
        *,
        learning_rate: float = 1e-3,
    ) -> None:
        if _mx is None or _mlx_nn is None or _mlx_optimizers is None:
            raise RuntimeError(
                "MLX world-model training requires mlx. Install the arc-ttt extra "
                "on Apple Silicon, or use HybridWorldModel for CPU smoke tests."
            )
        if not mlx_runtime_available():
            raise RuntimeError(
                "MLX is installed but its current device cannot be evaluated. "
                "Run this backend on Apple Silicon with a usable Metal/MLX device, "
                "or use HybridWorldModel for CPU smoke tests."
            )
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        _mx.random.seed((config or WorldModelConfig()).seed)
        self.config = config or WorldModelConfig()
        self.network = _MlxNetwork(self.config)
        self.optimizer = _mlx_optimizers.Adam(learning_rate=learning_rate)
        self.learning_rate = learning_rate
        _mx.eval(self.network.parameters())

    @staticmethod
    def _arrays(batch: WorldModelBatch) -> tuple[Any, ...]:
        input_masks = (
            batch.demo_input_masks
            if batch.demo_input_masks is not None
            else batch.demo_masks
        )
        output_masks = (
            batch.demo_output_masks
            if batch.demo_output_masks is not None
            else batch.demo_masks
        )
        return (
            _mx.array(batch.query_inputs.astype(np.int32)),
            _mx.array(batch.query_mask.astype(np.float32)),
            _mx.array(batch.query_objects.astype(np.float32)),
            _mx.array(batch.query_object_mask.astype(np.float32)),
            _mx.array(batch.demo_inputs.astype(np.int32)),
            _mx.array(batch.demo_outputs.astype(np.int32)),
            _mx.array(batch.demo_masks.astype(np.float32)),
            _mx.array(input_masks.astype(np.float32)),
            _mx.array(output_masks.astype(np.float32)),
            _mx.array(batch.demo_input_objects.astype(np.float32)),
            _mx.array(batch.demo_output_objects.astype(np.float32)),
            _mx.array(batch.demo_object_mask.astype(np.float32)),
            _mx.array(batch.demo_pair_mask.astype(np.float32)),
        )

    def forward_batch(self, batch: WorldModelBatch) -> MlxWorldModelOutput:
        output = self.network(*self._arrays(batch))
        return MlxWorldModelOutput(*output)

    def train_step(self, batch: WorldModelBatch) -> dict[str, float]:
        arrays = self._arrays(batch)
        targets = _mx.array(batch.targets.astype(np.int32)).reshape(-1)
        target_mask = _mx.array(batch.target_mask.astype(np.float32)).reshape(-1)
        program_targets = _mx.array(batch.program_targets.astype(np.int32))
        program_weights = _mx.array(
            _program_sample_weights(batch.program_targets, self.config).astype(np.float32)
        )

        def loss_fn(network: Any) -> Any:
            logits, program_logits, _, _ = network(*arrays)
            grid_loss = _mlx_nn.losses.cross_entropy(
                logits.reshape(-1, 10), targets, reduction="none"
            )
            grid_loss = _mx.sum(grid_loss * target_mask) / _mx.maximum(target_mask.sum(), 1.0)
            program_loss = _mlx_nn.losses.cross_entropy(
                program_logits, program_targets, reduction="none"
            )
            program_loss = _mx.sum(program_loss * program_weights)
            return grid_loss + self.config.program_loss_weight * program_loss

        loss, gradients = _mlx_nn.value_and_grad(self.network, loss_fn)(self.network)
        self.optimizer.update(self.network, gradients)
        _mx.eval(self.network.parameters(), self.optimizer.state, loss)
        output = self.forward_batch(batch)
        grid_prediction = np.asarray(output.grid_logits).argmax(axis=-1)
        cell_accuracy = float(
            ((grid_prediction == batch.targets) * batch.target_mask).sum()
            / max(float(batch.target_mask.sum()), 1.0)
        )
        return {"loss": float(loss.item()), "cell_accuracy": cell_accuracy}

    def predict(
        self,
        query_grid: Grid,
        demonstrations: Sequence[tuple[Grid, Grid]],
        *,
        output_shape: tuple[int, int] | None = None,
        top_k_programs: int = 3,
    ) -> WorldModelPrediction:
        if not demonstrations:
            raise ValueError("at least one labelled demonstration is required")
        if output_shape is None:
            output_shape = (len(query_grid), len(query_grid[0]))
        placeholder = [[0] * output_shape[1] for _ in range(output_shape[0])]
        example = WorldModelExample(
            task_id="inference",
            query_input=query_grid,
            target_output=placeholder,
            demonstrations=tuple(demonstrations),
            program_label=0,
        )
        batch = collate_world_model_examples([example], max_objects=self.config.max_objects)
        output = self.forward_batch(batch)
        _mx.eval(output.grid_logits, output.program_logits, output.argument_logits)
        height, width = output_shape
        grid_logits = np.asarray(output.grid_logits)[0, :height, :width]
        grid = np.argmax(grid_logits, axis=-1).astype(int).tolist()
        program_logits = np.asarray(output.program_logits)[0]
        argument_logits = np.asarray(output.argument_logits)[0]
        return WorldModelPrediction(
            grid=grid,
            grid_logits=grid_logits,
            program_logits=program_logits,
            argument_logits=argument_logits,
            program_candidates=program_candidates_from_logits(
                program_logits, argument_logits, top_k=top_k_programs
            ),
        )

    def save_checkpoint(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix not in {".npz", ".safetensors"}:
            target = target.with_suffix(".safetensors")
        self.network.save_weights(str(target))
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "config": asdict(self.config),
                    "learning_rate": self.learning_rate,
                    "metadata": metadata or {},
                },
                indent=2,
            )
            + "\n"
        )
        return target

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> MlxHybridWorldModel:
        target = Path(path)
        metadata_path = target.with_suffix(target.suffix + ".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("format_version") != 1:
            raise ValueError("unsupported MLX world-model checkpoint format")
        model = cls(
            WorldModelConfig(**metadata["config"]),
            learning_rate=float(metadata.get("learning_rate", 1e-3)),
        )
        model.network.load_weights(str(target))
        _mx.eval(model.network.parameters())
        return model


def train_mlx_world_model(
    model: MlxHybridWorldModel,
    examples: Sequence[WorldModelExample],
    *,
    epochs: int = 5,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Run end-to-end MLX training and return auditable epoch summaries."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not examples:
        raise ValueError("examples must not be empty")
    summaries: list[dict[str, float]] = []
    for epoch in range(epochs):
        rows = [
            model.train_step(batch)
            for batch in iter_world_model_batches(
                examples,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed + epoch,
                max_objects=model.config.max_objects,
            )
        ]
        summaries.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean([row["loss"] for row in rows])),
                "cell_accuracy": float(np.mean([row["cell_accuracy"] for row in rows])),
            }
        )
    return summaries


def evaluate_mlx_world_model(
    model: MlxHybridWorldModel,
    examples: Sequence[WorldModelExample],
    *,
    batch_size: int = 8,
) -> dict[str, float]:
    """Evaluate MLX grid and symbolic-head accuracy without updating weights."""
    if not examples:
        raise ValueError("examples must not be empty")
    cell_hits = 0.0
    cell_count = 0.0
    program_predictions: list[int] = []
    program_targets: list[int] = []
    for batch in iter_world_model_batches(
        examples, batch_size=batch_size, shuffle=False, max_objects=model.config.max_objects
    ):
        output = model.forward_batch(batch)
        _mx.eval(output.grid_logits, output.program_logits)
        predictions = np.asarray(output.grid_logits).argmax(axis=-1)
        cell_hits += float(((predictions == batch.targets) * batch.target_mask).sum())
        cell_count += float(batch.target_mask.sum())
        program_predictions.extend(np.asarray(output.program_logits).argmax(axis=-1).tolist())
        program_targets.extend(batch.program_targets.tolist())
    exact = 0
    for example in examples:
        prediction = model.predict(
            example.query_input,
            example.demonstrations,
            output_shape=(len(example.target_output), len(example.target_output[0])),
            top_k_programs=1,
        )
        exact += int(prediction.grid == example.target_output)
    return {
        "cell_accuracy": cell_hits / max(cell_count, 1.0),
        "exact_accuracy": exact / len(examples),
        **_program_metrics(program_predictions, program_targets),
    }


def program_candidates_from_logits(
    program_logits: np.ndarray,
    argument_logits: np.ndarray,
    *,
    top_k: int = 3,
) -> tuple[ProgramCandidate, ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    program = np.asarray(program_logits, dtype=np.float32).reshape(-1)
    arguments = np.asarray(argument_logits, dtype=np.float32).reshape(-1)
    probabilities = _softmax(program)
    order = np.argsort(-probabilities)[: min(top_k, len(PROGRAM_OPS))]
    argument = int(np.argmax(arguments)) if len(arguments) else 0
    return tuple(
        ProgramCandidate(
            op=PROGRAM_OPS[int(index)],
            argument=argument,
            probability=float(probabilities[int(index)]),
        )
        for index in order
    )


def train_world_model(
    model: HybridWorldModel,
    examples: Sequence[WorldModelExample],
    *,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-2,
    shuffle: bool = True,
    seed: int = 0,
) -> TrainingHistory:
    """Train the lightweight decoder/head and return per-epoch evidence."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if not examples:
        raise ValueError("examples must not be empty")
    metrics: list[EpochMetrics] = []
    for epoch in range(epochs):
        rows = [
            model.train_step(batch, learning_rate=learning_rate)
            for batch in iter_world_model_batches(
                examples,
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed + epoch,
                max_objects=model.config.max_objects,
            )
        ]
        metrics.append(
            EpochMetrics(
                epoch=epoch + 1,
                loss=float(np.mean([row["loss"] for row in rows])),
                grid_loss=float(np.mean([row["grid_loss"] for row in rows])),
                program_loss=float(np.mean([row["program_loss"] for row in rows])),
                cell_accuracy=float(np.mean([row["cell_accuracy"] for row in rows])),
                exact_accuracy=float(np.mean([row["exact_accuracy"] for row in rows])),
            )
        )
    return TrainingHistory(epochs=metrics)


def evaluate_world_model(
    model: HybridWorldModel, examples: Sequence[WorldModelExample], *, batch_size: int = 8
) -> dict[str, float]:
    """Evaluate exact grid, cell and auxiliary-program accuracy."""
    if not examples:
        raise ValueError("examples must not be empty")
    rows: list[dict[str, float]] = []
    program_predictions: list[int] = []
    program_targets: list[int] = []
    for batch in iter_world_model_batches(
        examples, batch_size=batch_size, shuffle=False, max_objects=model.config.max_objects
    ):
        output = model.forward_batch(batch)
        grid_loss, _ = model._grid_loss(output.grid_logits, batch.targets, batch.target_mask)
        predictions = np.argmax(output.grid_logits, axis=-1)
        valid = max(float(batch.target_mask.sum()), 1.0)
        rows.append(
            {
                "grid_loss": grid_loss,
                "cell_accuracy": float(
                    ((predictions == batch.targets) * batch.target_mask).sum() / valid
                ),
            }
        )
        program_predictions.extend(np.argmax(output.program_logits, axis=-1).tolist())
        program_targets.extend(batch.program_targets.tolist())
    exact = 0
    for example in examples:
        prediction = model.predict(
            example.query_input,
            example.demonstrations,
            output_shape=(len(example.target_output), len(example.target_output[0])),
            top_k_programs=1,
        )
        exact += int(prediction.grid == example.target_output)
    return {
        "grid_loss": float(np.mean([row["grid_loss"] for row in rows])),
        "cell_accuracy": float(np.mean([row["cell_accuracy"] for row in rows])),
        "exact_accuracy": exact / len(examples),
        **_program_metrics(program_predictions, program_targets),
    }


__all__ = [
    "PROGRAM_OPS",
    "EpochMetrics",
    "HybridWorldModel",
    "MlxHybridWorldModel",
    "MlxWorldModelOutput",
    "ProgramCandidate",
    "TrainingHistory",
    "WorldModelConfig",
    "WorldModelOutput",
    "WorldModelPrediction",
    "evaluate_world_model",
    "evaluate_mlx_world_model",
    "mlx_package_available",
    "mlx_runtime_available",
    "program_candidates_from_logits",
    "train_world_model",
    "train_mlx_world_model",
]
