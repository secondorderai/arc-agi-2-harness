from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from arc_agent.llm import GeneratedCandidate, Generation
from arc_agent.models import ArcTask, Grid, Usage, validate_grid
from arc_agent.skills import SkillCard

USER = 11
ASSISTANT = 12
EOS = 15
NEWLINE = 10

GEOMETRIES = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "transpose",
    "anti_transpose",
)

LORA_LAYER_KEYS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass(frozen=True)
class Augmentation:
    geometry: str = "identity"
    color_map: tuple[int, ...] = tuple(range(10))

    @property
    def inverse_color_map(self) -> tuple[int, ...]:
        inverse = [0] * 10
        for source, target in enumerate(self.color_map):
            inverse[target] = source
        return tuple(inverse)


def _rotate(grid: Grid, turns: int) -> Grid:
    result = [list(row) for row in grid]
    for _ in range(turns % 4):
        result = [list(row) for row in zip(*result[::-1], strict=True)]
    return result


def transform_grid(grid: Grid, augmentation: Augmentation) -> Grid:
    geometry = augmentation.geometry
    if geometry == "identity":
        transformed = [list(row) for row in grid]
    elif geometry == "rotate_90":
        transformed = _rotate(grid, 1)
    elif geometry == "rotate_180":
        transformed = _rotate(grid, 2)
    elif geometry == "rotate_270":
        transformed = _rotate(grid, 3)
    elif geometry == "transpose":
        transformed = [list(row) for row in zip(*grid, strict=True)]
    elif geometry == "anti_transpose":
        transformed = _rotate(
            [list(row) for row in zip(*grid, strict=True)],
            2,
        )
    else:
        raise ValueError(f"unknown geometry: {geometry}")
    return [[augmentation.color_map[value] for value in row] for row in transformed]


def inverse_transform_grid(grid: Grid, augmentation: Augmentation) -> Grid:
    recolored = [[augmentation.inverse_color_map[value] for value in row] for row in grid]
    inverse_geometry = {
        "identity": "identity",
        "rotate_90": "rotate_270",
        "rotate_180": "rotate_180",
        "rotate_270": "rotate_90",
        "transpose": "transpose",
        "anti_transpose": "anti_transpose",
    }[augmentation.geometry]
    return transform_grid(recolored, Augmentation(geometry=inverse_geometry))


def task_augmentations(task_id: str, *, seed: int, count: int) -> list[Augmentation]:
    augmentations = [Augmentation()]
    augmentations.extend(Augmentation(geometry=geometry) for geometry in GEOMETRIES[1:])
    digest = int(hashlib.sha256(task_id.encode()).hexdigest()[:16], 16)
    generator = random.Random(seed ^ digest)
    for _ in range(4):
        colors = list(range(1, 10))
        generator.shuffle(colors)
        augmentations.append(Augmentation(color_map=tuple([0, *colors])))
    return augmentations[:count]


def grid_to_tokens(grid: Grid) -> list[int]:
    tokens: list[int] = []
    for index, row in enumerate(grid):
        tokens.extend(row)
        if index < len(grid) - 1:
            tokens.append(NEWLINE)
    return tokens


def build_sequence(pairs: list[tuple[Grid, Grid]]) -> list[int]:
    tokens: list[int] = []
    for input_grid, output_grid in pairs:
        tokens.extend([USER, NEWLINE, *grid_to_tokens(input_grid), EOS])
        tokens.extend([ASSISTANT, NEWLINE, *grid_to_tokens(output_grid), EOS])
    return tokens


def assistant_labels(tokens: list[int]) -> list[int]:
    labels = [-100] * len(tokens)
    index = 0
    while index < len(tokens):
        if tokens[index] == ASSISTANT:
            index += 2
            while index < len(tokens) and tokens[index] != EOS:
                labels[index] = tokens[index]
                index += 1
            if index < len(tokens):
                labels[index] = EOS
        index += 1
    return labels


def build_prompt(pairs: list[tuple[Grid, Grid]], test_input: Grid) -> list[int]:
    return [
        *build_sequence(pairs),
        USER,
        NEWLINE,
        *grid_to_tokens(test_input),
        EOS,
        ASSISTANT,
        NEWLINE,
    ]


def parse_grid_text(
    text: str, *, expected_shape: tuple[int, int] | None = None
) -> Grid | None:
    rows: Grid = []
    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(character not in "0123456789" for character in line):
            break
        rows.append([int(character) for character in line])
    try:
        grid = validate_grid(rows, label="generated grid")
        if expected_shape is not None and (len(grid), len(grid[0])) != expected_shape:
            return None
        return grid
    except ValueError:
        return None


def grid_grammar_allowed_tokens(
    shape: tuple[int, int], generated_index: int
) -> tuple[int, ...]:
    """Return valid next tokens for a fixed-shape ARC grid serialization."""
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("grid grammar needs a positive shape")
    if generated_index < 0:
        raise ValueError("generated token index cannot be negative")
    grid_token_count = height * width + height - 1
    if generated_index >= grid_token_count:
        return (EOS,)
    if generated_index % (width + 1) == width:
        return (NEWLINE,)
    return tuple(range(10))


def make_grid_logits_processor(mx: Any, shape: tuple[int, int]):
    """Force generation to be a rectangular ARC grid followed by EOS."""

    def processor(tokens: Any, logits: Any):
        # MLX's processor history starts with the final prompt token, so the
        # generated position is one less than the history length.
        generated_index = int(tokens.size) - 1
        allowed = grid_grammar_allowed_tokens(shape, generated_index)
        vocabulary = mx.arange(logits.shape[-1])
        mask = vocabulary < 10 if len(allowed) == 10 else vocabulary == allowed[0]
        return mx.where(mask[None, :], logits, -float("inf"))

    return processor


def training_converged(
    epoch_losses: list[float],
    *,
    completed_epochs: int,
    min_epochs: int,
    median_threshold: float | None,
    max_threshold: float,
) -> bool:
    if median_threshold is None or completed_epochs < min_epochs or not epoch_losses:
        return False
    return median(epoch_losses) <= median_threshold and max(epoch_losses) <= max_threshold


def rank_candidate_keys(
    votes: Counter[str],
    generation_nlls: dict[str, list[float]],
    augmentation_nlls: dict[str, float] | None = None,
) -> list[str]:
    """Rank candidates by augmented consistency, then votes and generation NLL."""
    augmentation_nlls = augmentation_nlls or {}
    use_augmented_scores = bool(augmentation_nlls)

    def key(candidate: str) -> tuple[float, int, float, str]:
        augmented = augmentation_nlls.get(candidate, float("inf"))
        generated = min(generation_nlls.get(candidate, [float("inf")]))
        return (
            augmented if use_augmented_scores else 0.0,
            -votes[candidate],
            generated,
            candidate,
        )

    return sorted(votes, key=key)


def _task_side(task: ArcTask) -> int:
    grids = [pair.input for pair in task.train + task.test]
    grids.extend(pair.output for pair in task.train if pair.output is not None)
    return max(max(len(grid), len(grid[0])) for grid in grids if grid is not None)


def infer_output_shape(task: ArcTask, test_index: int) -> tuple[int, int] | None:
    relations = [
        (
            (len(pair.input), len(pair.input[0])),
            (len(pair.output or []), len((pair.output or [[]])[0])),
        )
        for pair in task.train
    ]
    output_shapes = {output_shape for _, output_shape in relations}
    if len(output_shapes) == 1:
        return next(iter(output_shapes))
    if all(input_shape == output_shape for input_shape, output_shape in relations):
        test_input = task.test[test_index].input
        return len(test_input), len(test_input[0])
    ratios = [
        (
            output_shape[0] // input_shape[0],
            output_shape[1] // input_shape[1],
        )
        for input_shape, output_shape in relations
        if output_shape[0] % input_shape[0] == 0
        and output_shape[1] % input_shape[1] == 0
    ]
    if len(ratios) == len(relations) and len(set(ratios)) == 1:
        row_ratio, column_ratio = ratios[0]
        test_input = task.test[test_index].input
        shape = len(test_input) * row_ratio, len(test_input[0]) * column_ratio
        if shape[0] <= 30 and shape[1] <= 30:
            return shape
    return None


class MlxArcTTTAdapter:
    """ARC-specialized QLoRA test-time training for Apple Silicon."""

    def __init__(
        self,
        *,
        model_path: str,
        rank: int = 32,
        scale: float = 10.0,
        learning_rate: float = 5e-5,
        epochs: int = 5,
        min_epochs: int = 2,
        early_stop_median_loss: float | None = None,
        early_stop_max_loss: float = 0.05,
        train_token_layers: bool = False,
        augmentation_count: int = 10,
        inference_augmentations: int = 1,
        scoring_augmentations: int = 0,
        samples: int = 3,
        include_greedy: bool = True,
        temperature: float = 0.5,
        top_p: float = 0.0,
        dfs_min_probability: float | None = None,
        dfs_max_candidates: int = 32,
        dfs_max_nodes: int = 2048,
        max_tokens: int = 1024,
        max_grid_side: int = 30,
        seed: int = 42,
    ) -> None:
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            import mlx_lm
            from mlx.utils import tree_unflatten
            from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.tuner.lora import LoRAEmbedding
            from mlx_lm.tuner.utils import linear_to_lora_layers, remove_lora_layers
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "MLX ARC TTT needs Apple Silicon GPU access and the arc-ttt extra"
            ) from exc

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"MLX ARC model not found: {path}")
        self.mx = mx
        self.nn = nn
        self.optim = optim
        self.mlx_lm = mlx_lm
        self.make_sampler = make_sampler
        self.make_prompt_cache = make_prompt_cache
        self.trim_prompt_cache = trim_prompt_cache
        self.linear_to_lora_layers = linear_to_lora_layers
        self.remove_lora_layers = remove_lora_layers
        self.LoRAEmbedding = LoRAEmbedding
        self.tree_unflatten = tree_unflatten
        self.model, self.tokenizer = mlx_lm.load(str(path))
        self.layer_count = len(self.model.model.layers)
        self.rank = rank
        self.scale = scale
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.early_stop_median_loss = early_stop_median_loss
        self.early_stop_max_loss = early_stop_max_loss
        self.train_token_layers = train_token_layers
        self.augmentation_count = augmentation_count
        self.inference_augmentations = inference_augmentations
        self.scoring_augmentations = scoring_augmentations
        self.samples = samples
        self.include_greedy = include_greedy
        self.temperature = temperature
        self.top_p = top_p
        self.dfs_min_probability = dfs_min_probability
        self.dfs_max_candidates = dfs_max_candidates
        self.dfs_max_nodes = dfs_max_nodes
        self.max_tokens = max_tokens
        self.max_grid_side = max_grid_side
        self.seed = seed

    def _attach_lora(self) -> None:
        # MLX modules are trainable by default. Freeze the quantized base before
        # inserting LoRA modules; otherwise optimizer steps corrupt base scales.
        self.model.freeze()
        config: dict[str, Any] = {
            "rank": self.rank,
            "dropout": 0.0,
            "scale": self.scale,
        }
        if self.train_token_layers:
            config["keys"] = [*LORA_LAYER_KEYS, "model.embed_tokens", "lm_head"]
        self.linear_to_lora_layers(
            self.model,
            num_layers=self.layer_count,
            config=config,
        )
        self.mx.eval(self.model.parameters())

    def _detach_lora(self) -> None:
        embedding_modules = [
            (name, module.embedding)
            for name, module in self.model.named_modules()
            if isinstance(module, self.LoRAEmbedding)
        ]
        self.remove_lora_layers(self.model)
        if embedding_modules:
            self.model.update_modules(self.tree_unflatten(embedding_modules))

    def _training_items(
        self, task: ArcTask, augmentations: list[Augmentation]
    ) -> list[tuple[list[int], list[int]]]:
        items: list[tuple[list[int], list[int]]] = []
        for augmentation_index, augmentation in enumerate(augmentations):
            pairs = [
                (
                    transform_grid(pair.input, augmentation),
                    transform_grid(pair.output or [[]], augmentation),
                )
                for pair in task.train
            ]
            # Make every demonstration appear in the query-like final position across
            # augmentations. A fixed order otherwise teaches position-specific shortcuts.
            offset = augmentation_index % len(pairs)
            pairs = pairs[offset:] + pairs[:offset]
            tokens = build_sequence(pairs)
            items.append((tokens, assistant_labels(tokens)))
        return items

    def _train(self, task: ArcTask, augmentations: list[Augmentation]) -> list[float]:
        mx, nn = self.mx, self.nn
        optimizer = self.optim.Adam(learning_rate=self.learning_rate)
        losses: list[float] = []
        items = self._training_items(task, augmentations)
        self.model.train()
        for epoch_index in range(self.epochs):
            epoch_started = time.monotonic()
            epoch_losses: list[float] = []
            for tokens, labels in items:
                inputs = mx.array([tokens[:-1]])
                targets = mx.array([labels[1:]])

                def loss_fn(
                    model: Any,
                    batch_inputs: Any = inputs,
                    batch_targets: Any = targets,
                ):
                    output = model(batch_inputs)
                    if isinstance(output, tuple):
                        output = output[0]
                    logits = output.reshape(-1, output.shape[-1])
                    target = batch_targets.reshape(-1)
                    mask = target != -100
                    safe_target = mx.where(mask, target, mx.zeros_like(target))
                    cross_entropy = nn.losses.cross_entropy(
                        logits, safe_target, reduction="none"
                    )
                    return (cross_entropy * mask).sum() / mask.sum()

                loss, gradients = nn.value_and_grad(self.model, loss_fn)(self.model)
                optimizer.update(self.model, gradients)
                mx.eval(self.model.parameters(), optimizer.state)
                measured_loss = float(loss.item())
                losses.append(measured_loss)
                epoch_losses.append(measured_loss)
            converged = training_converged(
                epoch_losses,
                completed_epochs=epoch_index + 1,
                min_epochs=self.min_epochs,
                median_threshold=self.early_stop_median_loss,
                max_threshold=self.early_stop_max_loss,
            )
            print(
                f"[{task.task_id}] TTT epoch {epoch_index + 1}/{self.epochs}: "
                f"median_loss={median(epoch_losses):.4f} "
                f"max_loss={max(epoch_losses):.4f} "
                f"seconds={time.monotonic() - epoch_started:.1f}",
                flush=True,
            )
            if converged:
                break
        self.model.eval()
        return losses

    def _generate_response(
        self,
        *,
        prompt: list[int],
        generation_limit: int,
        temperature: float,
        expected_shape: tuple[int, int] | None,
    ) -> tuple[str, float]:
        sampler = self.make_sampler(temp=temperature, top_p=self.top_p)
        logits_processors = (
            [make_grid_logits_processor(self.mx, expected_shape)]
            if expected_shape is not None
            else None
        )
        fragments: list[str] = []
        negative_log_likelihood = 0.0
        for response in self.mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=generation_limit,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            fragments.append(response.text)
            token_log_probability = response.logprobs[response.token]
            self.mx.eval(token_log_probability)
            negative_log_likelihood -= float(token_log_probability.item())
        return "".join(fragments), negative_log_likelihood

    def _dfs_grid_responses(
        self,
        *,
        prompt: list[int],
        expected_shape: tuple[int, int],
    ) -> tuple[list[tuple[str, float]], int]:
        """Enumerate high-probability fixed-grid replies with bounded DFS."""
        if self.dfs_min_probability is None:
            return [], 0

        mx = self.mx
        max_nll = -math.log(self.dfs_min_probability)
        prompt_cache = self.make_prompt_cache(self.model)
        output = self.model(mx.array([prompt]), cache=prompt_cache)
        if isinstance(output, tuple):
            output = output[0]
        next_logits = output[0, -1, :]
        mx.eval(next_logits, [cache.state for cache in prompt_cache])

        results: list[tuple[str, float]] = []
        nodes = 0

        def explore(logits: Any, path: list[int], nll: float) -> None:
            nonlocal nodes
            if len(results) >= self.dfs_max_candidates or nodes >= self.dfs_max_nodes:
                return
            allowed = grid_grammar_allowed_tokens(expected_shape, len(path))
            log_probabilities = logits - mx.logsumexp(logits)
            allowed_nlls = [
                -float(log_probabilities[token].item()) for token in allowed
            ]
            ranked_tokens = sorted(
                zip(allowed, allowed_nlls, strict=True),
                key=lambda item: item[1],
            )
            for token, token_nll in ranked_tokens:
                next_nll = nll + token_nll
                if next_nll > max_nll:
                    continue
                nodes += 1
                if token == EOS:
                    results.append((self.tokenizer.decode(path), next_nll))
                    if len(results) >= self.dfs_max_candidates:
                        return
                    continue
                branch_output = self.model(mx.array([[token]]), cache=prompt_cache)
                if isinstance(branch_output, tuple):
                    branch_output = branch_output[0]
                branch_logits = branch_output[0, -1, :]
                mx.eval(branch_logits, [cache.state for cache in prompt_cache])
                explore(branch_logits, [*path, token], next_nll)
                trimmed = self.trim_prompt_cache(prompt_cache, 1)
                if trimmed != 1:
                    raise RuntimeError("MLX prompt cache could not be rewound during DFS")
                if len(results) >= self.dfs_max_candidates or nodes >= self.dfs_max_nodes:
                    return

        explore(next_logits, [], 0.0)
        return results, nodes

    def _score_candidate(
        self,
        task: ArcTask,
        test_index: int,
        candidate: Grid,
        augmentations: list[Augmentation],
    ) -> float:
        total = 0.0
        for augmentation in augmentations[: self.scoring_augmentations]:
            pairs = [
                (
                    transform_grid(pair.input, augmentation),
                    transform_grid(pair.output or [[]], augmentation),
                )
                for pair in task.train
            ]
            test_input = transform_grid(task.test[test_index].input, augmentation)
            prompt = build_prompt(pairs, test_input)
            reply = [*grid_to_tokens(transform_grid(candidate, augmentation)), EOS]
            sequence = prompt + reply
            output = self.model(self.mx.array([sequence[:-1]]))
            if isinstance(output, tuple):
                output = output[0]
            logits = output[0, len(prompt) - 1 :, :]
            token_losses = self.nn.losses.cross_entropy(
                logits,
                self.mx.array(reply),
                reduction="none",
            )
            self.mx.eval(token_losses)
            total += float(token_losses.sum().item())
        return total

    def _predict_one(
        self,
        task: ArcTask,
        test_index: int,
        augmentations: list[Augmentation],
    ) -> tuple[
        list[tuple[Grid, int, float]],
        int,
        list[str],
        list[float],
        dict[str, float],
        int,
    ]:
        votes: Counter[str] = Counter()
        grids: dict[str, Grid] = {}
        generation_nlls: dict[str, list[float]] = {}
        raw: list[str] = []
        raw_nlls: list[float] = []
        attempts = 0
        dfs_nodes = 0
        expected_shape = infer_output_shape(task, test_index)
        generation_limit = self.max_tokens
        if expected_shape is not None:
            height, width = expected_shape
            generation_limit = min(generation_limit, height * width + height)
        for augmentation_index, augmentation in enumerate(
            augmentations[: self.inference_augmentations]
        ):
            pairs = [
                (
                    transform_grid(pair.input, augmentation),
                    transform_grid(pair.output or [[]], augmentation),
                )
                for pair in task.train
            ]
            test_input = transform_grid(task.test[test_index].input, augmentation)
            prompt = build_prompt(pairs, test_input)
            temperatures = [
                *([0.0] if self.include_greedy else []),
                *([self.temperature] * self.samples),
            ]
            for sample_index, temperature in enumerate(temperatures):
                attempts += 1
                self.mx.random.seed(
                    self.seed
                    + test_index * 10_000
                    + augmentation_index * 100
                    + sample_index
                )
                response, response_nll = self._generate_response(
                    prompt=prompt,
                    generation_limit=generation_limit,
                    temperature=temperature,
                    expected_shape=expected_shape,
                )
                raw.append(response)
                raw_nlls.append(response_nll)
                predicted = parse_grid_text(response, expected_shape=expected_shape)
                if predicted is None:
                    continue
                try:
                    canonical = inverse_transform_grid(predicted, augmentation)
                    validate_grid(canonical, label="canonical prediction")
                except ValueError:
                    continue
                key = json.dumps(canonical, separators=(",", ":"))
                votes[key] += 1
                grids[key] = canonical
                generation_nlls.setdefault(key, []).append(response_nll)
            if expected_shape is not None and self.dfs_min_probability is not None:
                dfs_responses, explored_nodes = self._dfs_grid_responses(
                    prompt=prompt,
                    expected_shape=expected_shape,
                )
                dfs_nodes += explored_nodes
                for response, response_nll in dfs_responses:
                    attempts += 1
                    raw.append(response)
                    raw_nlls.append(response_nll)
                    predicted = parse_grid_text(response, expected_shape=expected_shape)
                    if predicted is None:
                        continue
                    try:
                        canonical = inverse_transform_grid(predicted, augmentation)
                        validate_grid(canonical, label="canonical DFS prediction")
                    except ValueError:
                        continue
                    key = json.dumps(canonical, separators=(",", ":"))
                    votes[key] += 1
                    grids[key] = canonical
                    generation_nlls.setdefault(key, []).append(response_nll)
        augmentation_nlls = (
            {
                key: self._score_candidate(
                    task,
                    test_index,
                    grid,
                    augmentations,
                )
                for key, grid in grids.items()
            }
            if self.scoring_augmentations
            else {}
        )
        ranked_keys = rank_candidate_keys(votes, generation_nlls, augmentation_nlls)
        ranked = [
            (
                grids[key],
                votes[key],
                augmentation_nlls.get(key, min(generation_nlls[key])),
            )
            for key in ranked_keys[:2]
        ]
        return ranked, attempts, raw, raw_nlls, augmentation_nlls, dfs_nodes

    def generate(
        self,
        task: ArcTask,
        skills: list[SkillCard],
        *,
        level: int,
        feedback: list[str] | None = None,
        search_mode: str | None = None,
    ) -> Generation:
        del skills, level, feedback, search_mode
        started = time.monotonic()
        if _task_side(task) > self.max_grid_side:
            return Generation(
                raw_text=json.dumps(
                    {"skipped": "grid_side_limit", "limit": self.max_grid_side}
                ),
                usage=Usage(calls=1, latency_seconds=time.monotonic() - started),
            )

        augmentations = task_augmentations(
            task.task_id,
            seed=self.seed,
            count=self.augmentation_count,
        )
        training_seed = self.seed ^ int(
            hashlib.sha256(task.task_id.encode()).hexdigest()[:8],
            16,
        )
        self.mx.random.seed(training_seed)
        self._attach_lora()
        try:
            losses = self._train(task, augmentations)
            per_test: list[list[tuple[Grid, int, float]]] = []
            total_attempts = 0
            raw: list[list[str]] = []
            raw_nlls: list[list[float]] = []
            augmented_nlls: list[dict[str, float]] = []
            dfs_nodes: list[int] = []
            for test_index in range(len(task.test)):
                (
                    ranked,
                    attempts,
                    responses,
                    response_nlls,
                    candidate_nlls,
                    explored_nodes,
                ) = self._predict_one(task, test_index, augmentations)
                per_test.append(ranked)
                total_attempts += attempts
                raw.append(responses)
                raw_nlls.append(response_nlls)
                augmented_nlls.append(candidate_nlls)
                dfs_nodes.append(explored_nodes)
                print(
                    f"[{task.task_id}] test {test_index + 1}/{len(task.test)}: "
                    f"attempts={attempts} candidates={len(ranked)} "
                    f"dfs_nodes={explored_nodes}",
                    flush=True,
                )

            candidates: list[GeneratedCandidate] = []
            for rank in range(2):
                if any(len(ranked) <= rank for ranked in per_test):
                    continue
                predictions = [ranked[rank][0] for ranked in per_test]
                vote_confidence = sum(ranked[rank][1] for ranked in per_test) / max(
                    1, total_attempts
                )
                confidence = min(1.0, vote_confidence + 0.02 - rank * 0.01)
                candidates.append(
                    GeneratedCandidate(
                        hypothesis=(
                            f"MLX ARC per-puzzle QLoRA vote rank {rank + 1}; "
                            f"final_loss={losses[-1]:.4f}; "
                            f"candidate_nll={sum(item[rank][2] for item in per_test):.3f}"
                        ),
                        predictions=predictions,
                        source="mlx_ttt",
                        confidence=confidence,
                    )
                )
            metadata = {
                "augmentations": len(augmentations),
                "epochs_configured": self.epochs,
                "epochs_completed": len(losses) // len(augmentations),
                "training_seed": training_seed,
                "losses": losses,
                "generation_attempts": total_attempts,
                "responses": raw,
                "response_nlls": raw_nlls,
                "candidate_augmented_nlls": augmented_nlls,
                "dfs_nodes": dfs_nodes,
            }
            completion_tokens = sum(len(response) for group in raw for response in group)
            return Generation(
                candidates=candidates,
                usage=Usage(
                    prompt_tokens=sum(
                        len(tokens)
                        for tokens, _ in self._training_items(task, augmentations)
                    ),
                    completion_tokens=completion_tokens,
                    calls=1,
                    latency_seconds=time.monotonic() - started,
                ),
                raw_text=json.dumps(metadata, separators=(",", ":")),
            )
        finally:
            self._detach_lora()
            self.mx.clear_cache()
