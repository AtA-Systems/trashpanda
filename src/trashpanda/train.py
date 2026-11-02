"""Token-budgeted training with accumulation, finite checks, signals, and resume."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
import tempfile
import time
import random
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .checkpoint import TrainingState, checkpoint_path, load_checkpoint, restore_rng_state, save_checkpoint
from .config import Config, TrainingConfig
from .data import MMapTokenDataset, PCG64Sampler
from .model import build_model, causal_loss
from .provenance import evaluate_policy, make_record, record_identity, sha256_file
from .tokenizer import DeterministicTokenizer, tokenizer_snapshot


@dataclass(frozen=True)
class TokenBudgetSchedule:
    context_length: int
    effective_batch_size: int
    token_budget: int

    @classmethod
    def from_config(cls, config: Config) -> "TokenBudgetSchedule":
        return cls(
            context_length=config.model.context_length,
            effective_batch_size=config.training.effective_batch_size,
            token_budget=config.training.token_budget,
        )

    @property
    def tokens_per_sequence(self) -> int:
        return self.context_length - 1

    @property
    def tokens_per_update(self) -> int:
        return self.effective_batch_size * self.tokens_per_sequence

    @property
    def updates(self) -> int:
        return math.ceil(self.token_budget / self.tokens_per_update)

    @property
    def accounted_tokens(self) -> int:
        return self.updates * self.tokens_per_update


@dataclass
class TrainResult:
    state: TrainingState
    checkpoint: Path | None
    device: str
    schedule: TokenBudgetSchedule
    model: Any | None = None
    optimizer: Any | None = None
    sampler: PCG64Sampler | None = None
    tokenizer: Any | None = None


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("training requires PyTorch")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def resolve_device(training: TrainingConfig) -> Any:
    _require_torch()
    if training.device == "cpu":
        return torch.device("cpu")
    if training.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("recipe requests CUDA but no CUDA device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimizer_param_groups(model: Any, weight_decay: float) -> list[dict[str, Any]]:
    """Separate matrix parameters (decay) from vector parameters (no decay)."""
    _require_torch()
    decay: list[Any] = []
    no_decay: list[Any] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _finite_tensor(value: Any) -> bool:
    return bool(torch.isfinite(value).all().item())


class _BoundarySignal:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def handler(self, signum: int, _frame: Any) -> None:
        # Do not interrupt a backward pass.  The loop observes this after the
        # current optimizer boundary and writes a resumable checkpoint.
        self.requested = True
        self.signal_number = signum

    def __enter__(self) -> "_BoundarySignal":
        self.previous: dict[int, Any] = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self.handler)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)


def _latest_checkpoint(path: str | Path) -> Path:
    target = Path(path)
    if target.is_file():
        return target
    candidates = sorted(target.glob("checkpoint-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found in {target}")
    return candidates[-1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _software_identity() -> dict[str, Any]:
    """Capture an allowlisted environment summary, never the process environment."""
    result: dict[str, Any] = {"python": sys.version.split()[0]}
    if torch is not None:
        result["torch"] = getattr(torch, "__version__", "unknown")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2
        ).stdout.strip()
        result["git_commit"] = commit
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False, timeout=2).returncode != 0
        result["git_dirty"] = dirty
    except (OSError, subprocess.SubprocessError):
        result["git_commit"] = "unknown"
        result["git_dirty"] = None
    return result


def _input_records(dataset: MMapTokenDataset) -> list[dict[str, Any]]:
    values = dataset.manifest.get("provenance", {}).get("records", [])
    return [dict(value) for value in values if isinstance(value, Mapping) and value.get("type") in {"input", "preparation"}]


def _previous_segments(value: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    return [dict(item) for item in value.get("segments", []) if isinstance(item, Mapping)]


def _provenance_snapshot(
    *,
    run_id: str,
    session_id: str,
    start_time: str,
    end_time: str | None,
    status: str,
    elapsed_seconds: float,
    start_state: TrainingState,
    state: TrainingState,
    parent_checkpoint_sha256: str | None,
    input_records: list[dict[str, Any]],
    previous_segments: list[dict[str, Any]],
    config: Config,
    policy_decision: Mapping[str, Any] | None = None,
    source_target_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    segment = make_record(
        "execution_segment",
        {
            "run_id": run_id,
            "session_id": session_id,
            "status": status,
            "start_time_utc": start_time,
            "end_time_utc": end_time,
            "elapsed_seconds": elapsed_seconds,
            "start_counters": start_state.to_dict(),
            "end_counters": state.to_dict(),
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "input_records": [record["id"] for record in input_records],
            "software": _software_identity(),
            "config_sha256": record_identity(config.to_dict()),
            "evidence_state": "observed",
            "policy_decision": dict(policy_decision) if policy_decision is not None else {"decision": "not_evaluated"},
        },
    )
    segments = previous_segments + [segment]
    run = make_record(
        "run",
        {
            "run_id": run_id,
            "status": status,
            "segments": [item["id"] for item in segments],
            "parents": [item["parent_checkpoint_sha256"] for item in segments if item.get("parent_checkpoint_sha256")],
            "cumulative_counters": state.to_dict(),
            "actual_target_accounting": {
                "committed_targets": state.consumed_tokens,
                "source_target_counts": {key: int(value) for key, value in (source_target_counts or {}).items()},
            },
            "requested_budget": config.training.token_budget,
            "requested_budget_is_evidence": False,
            "input_records": [record["id"] for record in input_records],
            "software": _software_identity(),
            "config_sha256": record_identity(config.to_dict()),
            "evidence_state": "observed",
            "policy_decision": dict(policy_decision) if policy_decision is not None else {"decision": "not_evaluated"},
        },
    )
    deduped: dict[str, dict[str, Any]] = {}
    for record in input_records + segments + [run]:
        deduped[str(record["id"])] = record
    return {
        "run_id": run_id,
        "session_id": session_id,
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "status": status,
        "segments": segments,
        "run_record": run,
        "records": list(deduped.values()),
    }


def train(
    config: Config,
    dataset: MMapTokenDataset,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    resume: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    max_steps: int | None = None,
    policy: Mapping[str, Any] | None = None,
) -> TrainResult:
    """Run updates until the shifted-token budget or max_steps is reached."""
    _require_torch()
    if len(dataset) == 0:
        raise ValueError("training dataset is empty")
    if dataset.context_length != config.model.context_length:
        raise ValueError("prepared dataset context_length does not match model context_length")
    schedule = TokenBudgetSchedule(
        context_length=dataset.context_length,
        effective_batch_size=config.training.effective_batch_size,
        token_budget=config.training.token_budget,
    )
    if schedule.tokens_per_update <= 0:
        raise ValueError("invalid token schedule")
    input_records = _input_records(dataset)
    policy_decision: Mapping[str, Any] | None = None
    if policy is not None:
        policy_decision = evaluate_policy(input_records, policy)
        if policy_decision.get("decision") == "deny":
            reasons = "; ".join(reason.get("message", reason.get("code", "unknown")) for reason in policy_decision.get("reasons", []))
            raise PermissionError(f"provenance policy denied training: {reasons}")
    if model is None:
        seed_everything(config.training.seed, deterministic=config.runtime.deterministic)
        model = build_model(config.model, seed=config.training.seed)
    device = resolve_device(config.training)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model, config.training.weight_decay),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
    )
    total_updates = max(1, schedule.updates)
    def lr_factor(step: int) -> float:
        if config.training.warmup_steps and step < config.training.warmup_steps:
            return max(1e-12, (step + 1) / config.training.warmup_steps)
        progress = min(1.0, max(0.0, (step - config.training.warmup_steps) / max(1, total_updates - config.training.warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.training.min_lr_ratio + (1.0 - config.training.min_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    sampler = PCG64Sampler(len(dataset), config.training.seed, repeat_data=config.training.repeat_data)
    state = TrainingState()
    resume_path: Path | None = None
    resume_payload: dict[str, Any] | None = None
    prior_provenance: Mapping[str, Any] | None = None
    if resume:
        resume_path = _latest_checkpoint(resume)
        # Inspect the immutable configuration before applying any saved model,
        # optimizer, scheduler, or RNG state. Legacy checkpoints omitted the
        # architecture field and are defined to mean Llama.
        saved_metadata = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(saved_metadata, Mapping):
            raise ValueError("checkpoint metadata is not a mapping")
        saved_config = saved_metadata.get("config", {})
        saved_model = dict(saved_config.get("model", {})) if isinstance(saved_config, Mapping) else {}
        saved_model.setdefault("architecture", "llama")
        expected_model = config.to_dict()["model"]
        if saved_model != expected_model:
            raise ValueError("checkpoint model configuration does not match current recipe")
        del saved_metadata
        payload = load_checkpoint(resume_path, model=model, optimizer=optimizer, map_location=str(device))
        resume_payload = payload
        state = TrainingState.from_dict(payload["state"])
        sampler.load_state_dict(payload["sampler"])
        prior_provenance = payload.get("extra", {}).get("provenance") if isinstance(payload.get("extra"), Mapping) else None
        expected_identity = dataset.manifest.get("identity")
        if payload.get("data_identity") and expected_identity and payload["data_identity"] != expected_identity:
            raise ValueError("checkpoint data identity does not match prepared dataset")
        if payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        if payload.get("rng"):
            restore_rng_state(payload["rng"])
        saved_config = payload.get("config", {})
        normalized_saved_model = dict(saved_config.get("model", {})) if isinstance(saved_config, Mapping) else {}
        normalized_saved_model.setdefault("architecture", "llama")
        if normalized_saved_model != config.to_dict()["model"] or saved_config.get("training") != config.to_dict()["training"]:
            raise ValueError("checkpoint model/training configuration does not match current recipe")

    destination = Path(checkpoint_dir or config.runtime.checkpoint_dir or config.runtime.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.jsonl"
    run_started = time.perf_counter()
    effective_max_steps = max_steps or config.runtime.max_steps
    remaining_tokens = max(0, config.training.token_budget - state.consumed_tokens)
    if not config.training.repeat_data and schedule.updates * config.training.effective_batch_size > len(dataset):
        raise ValueError("--no-repeat-data cannot satisfy the rounded token budget with available sequences")
    updates_remaining = math.ceil(remaining_tokens / schedule.tokens_per_update) if remaining_tokens else 0
    if effective_max_steps is not None:
        updates_remaining = min(updates_remaining, max(0, effective_max_steps - state.update_step))
    if not config.training.repeat_data:
        available = len(dataset) - (sampler.position if sampler._permutation else 0)
        updates_remaining = min(updates_remaining, max(0, available // config.training.batch_size))
    last_checkpoint = resume_path
    if updates_remaining == 0 and resume_path is not None:
        return TrainResult(
            state=state,
            checkpoint=resume_path,
            device=str(device),
            schedule=schedule,
            model=model,
            optimizer=optimizer,
            sampler=sampler,
            tokenizer=tokenizer,
        )
    run_id = str(prior_provenance.get("run_id")) if prior_provenance and prior_provenance.get("run_id") else str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    session_start = _utc_now()
    start_state = TrainingState.from_dict(state.to_dict())
    parent_digest = sha256_file(resume_path) if resume_path is not None else None
    prior_segments = _previous_segments(prior_provenance)
    source_target_totals: dict[str, int] = {}
    if prior_provenance and isinstance(prior_provenance.get("run_record"), Mapping):
        prior_counts = prior_provenance["run_record"].get("actual_target_accounting", {}).get("source_target_counts", {})
        if isinstance(prior_counts, Mapping):
            source_target_totals.update({str(key): int(value) for key, value in prior_counts.items()})
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.precision == "fp16" and device.type == "cuda")
    autocast_enabled = config.training.precision in {"bf16", "fp16"} and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if config.training.precision == "bf16" else torch.float16

    with _BoundarySignal() as boundary:
        for _ in range(updates_remaining):
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for _micro in range(config.training.grad_accumulation_steps):
                indices = sampler.next_indices(config.training.microbatch_size)
                inputs, targets = dataset.batch(indices)
                target_attribution = getattr(dataset, "target_source_counts", lambda _indices: {})
                for source_id, count in target_attribution(indices).items():
                    source_target_totals[source_id] = source_target_totals.get(source_id, 0) + int(count)
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    loss = causal_loss(model, inputs, targets)
                if not _finite_tensor(loss):
                    raise FloatingPointError("non-finite loss encountered")
                update_loss += float(loss.detach().float().item())
                scaled_loss = loss / config.training.grad_accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                state.micro_steps += 1
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None and not _finite_tensor(parameter.grad):
                    raise FloatingPointError("non-finite gradient encountered")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip, error_if_nonfinite=True)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            state.update_step += 1
            state.consumed_tokens += schedule.tokens_per_update
            should_save = state.update_step % config.training.checkpoint_interval == 0
            should_stop = boundary.requested
            if should_save or should_stop:
                provenance = _provenance_snapshot(
                    run_id=run_id,
                    session_id=session_id,
                    start_time=session_start,
                    end_time=None,
                    status="interrupted" if should_stop else "running",
                    elapsed_seconds=time.perf_counter() - run_started,
                    start_state=start_state,
                    state=state,
                    parent_checkpoint_sha256=parent_digest,
                    input_records=input_records,
                    previous_segments=prior_segments,
                    config=config,
                    policy_decision=policy_decision,
                    source_target_counts=source_target_totals,
                )
                last_checkpoint = _save_training_checkpoint(
                    destination, model, optimizer, scheduler, state, config, sampler, tokenizer, dataset, provenance=provenance
                )
            with metrics_path.open("a", encoding="utf-8") as metrics:
                metrics.write(json.dumps({
                    "step": state.update_step,
                    "targets": state.consumed_tokens,
                    "train_loss": update_loss / config.training.grad_accumulation_steps,
                    "learning_rate": config.training.learning_rate,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }, sort_keys=True) + "\n")
            if should_stop:
                break

    # Always leave a checkpoint at the actual boundary, including a tiny run
    # shorter than the configured interval.
    if last_checkpoint is None or state.update_step > 0:
        provenance = _provenance_snapshot(
            run_id=run_id,
            session_id=session_id,
            start_time=session_start,
            end_time=_utc_now(),
            status="completed" if state.consumed_tokens >= config.training.token_budget else "stopped",
            elapsed_seconds=time.perf_counter() - run_started,
            start_state=start_state,
            state=state,
            parent_checkpoint_sha256=parent_digest,
            input_records=input_records,
            previous_segments=prior_segments,
            config=config,
            policy_decision=policy_decision,
            source_target_counts=source_target_totals,
        )
        last_checkpoint = _save_training_checkpoint(destination, model, optimizer, scheduler, state, config, sampler, tokenizer, dataset, provenance=provenance)
    return TrainResult(
        state=state,
        checkpoint=last_checkpoint,
        device=str(device),
        schedule=schedule,
        model=model,
        optimizer=optimizer,
        sampler=sampler,
        tokenizer=tokenizer,
    )


def _save_training_checkpoint(
    destination: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    state: TrainingState,
    config: Config,
    sampler: PCG64Sampler,
    tokenizer: Any | None,
    dataset: MMapTokenDataset,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    if tokenizer is None:
        if config.model.vocab_size != 32:
            raise ValueError("production training requires its pinned tokenizer object for checkpoint provenance")
        tokenizer = DeterministicTokenizer()
    tokenizer_files: dict[str, bytes] = {}
    tokenizer_dir = Path(dataset.manifest_path).parent / "tokenizer"
    if tokenizer_dir.is_dir():
        tokenizer_files = {file.relative_to(tokenizer_dir).as_posix(): file.read_bytes() for file in tokenizer_dir.rglob("*") if file.is_file()}
    elif hasattr(tokenizer, "save_pretrained"):
        with tempfile.TemporaryDirectory(prefix="trashpanda-tokenizer-") as temp:
            tokenizer.save_pretrained(temp)
            tokenizer_files = {file.relative_to(temp).as_posix(): file.read_bytes() for file in Path(temp).rglob("*") if file.is_file()}
    return save_checkpoint(
        checkpoint_path(destination, state.update_step),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config=config.to_dict(),
        sampler_state=sampler.state_dict(),
        tokenizer_snapshot=tokenizer_snapshot(tokenizer).to_dict(),
        tokenizer_files=tokenizer_files,
        data_identity=dataset.manifest.get("identity"),
        extra={"provenance": dict(provenance)} if provenance is not None else None,
    )


__all__ = [
    "TokenBudgetSchedule",
    "TrainResult",
    "optimizer_param_groups",
    "resolve_device",
    "seed_everything",
    "train",
]
