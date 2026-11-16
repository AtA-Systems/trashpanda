"""Single-process multi-device gradient accumulation.

One optimizer update already consists of several microbatches whose gradients
are summed.  With more than one device, those microbatches run concurrently on
per-device model replicas and the replica gradients are added to the primary
model before clipping and the optimizer step.

The sampler order, batch membership, token accounting, optimizer, scheduler,
and checkpoint contents are those of a single-device run, so a checkpoint
written with several devices resumes on one device and vice versa.  Results
match a single-device run up to floating-point summation order.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from typing import Any, Sequence

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .config import TrainingConfig
from .model import causal_loss


def resolve_devices(training: TrainingConfig, gpus: int | None) -> list[Any] | None:
    """Map a ``--gpus`` request to explicit CUDA devices.

    ``None`` or ``1`` keeps the single-device path and returns ``None``.
    """
    if gpus is None or gpus == 1:
        return None
    if torch is None:
        raise RuntimeError("training requires PyTorch")
    if type(gpus) is not int or gpus < 1:
        raise ValueError("--gpus must be a positive integer")
    if training.device == "cpu":
        raise ValueError("--gpus needs a CUDA recipe; this recipe trains on cpu")
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpus > available:
        raise RuntimeError(f"--gpus {gpus} requested but {available} CUDA device(s) are visible")
    return [torch.device("cuda", index) for index in range(gpus)]


def validate_devices(devices: Sequence[Any], training: TrainingConfig) -> list[Any]:
    """Normalize an explicit device list and reject unsupported combinations."""
    if torch is None:
        raise RuntimeError("training requires PyTorch")
    resolved = [torch.device(device) for device in devices]
    if not resolved:
        raise ValueError("devices must name at least one device")
    if len(resolved) > 1 and training.precision == "fp16":
        # GradScaler keeps its scale on one device; FP16 across replicas needs
        # its own design.  FP32 and BF16 need no scaler.
        raise ValueError("fp16 is not supported with more than one device; use fp32 or bf16")
    if len(resolved) > training.grad_accumulation_steps:
        raise ValueError(
            f"{len(resolved)} devices need at least {len(resolved)} microbatches per update, "
            f"but batch_size / microbatch_size is {training.grad_accumulation_steps}; lower microbatch_size"
        )
    return resolved


class ReplicaSet:
    """Model replicas, one per device, that accumulate gradients for one update.

    ``devices[0]`` hosts the primary model (the one the optimizer owns).  The
    other devices hold copies whose parameters are refreshed from the primary
    at the start of each update and whose gradients are added back at the end.
    """

    def __init__(self, model: Any, devices: Sequence[Any], *, autocast_enabled: bool, autocast_dtype: Any) -> None:
        if torch is None:
            raise RuntimeError("training requires PyTorch")
        self.devices = [torch.device(device) for device in devices]
        self.primary = model
        self.replicas = [model]
        for device in self.devices[1:]:
            replica = copy.deepcopy(model)
            replica.to(device)
            replica.train(model.training)
            self.replicas.append(replica)
        primary_names = [name for name, _ in model.named_parameters()]
        for replica in self.replicas[1:]:
            if [name for name, _ in replica.named_parameters()] != primary_names:
                raise RuntimeError("replica parameters do not line up with the primary model")
        self._autocast_enabled = autocast_enabled
        self._autocast_dtype = autocast_dtype
        self._pool = ThreadPoolExecutor(max_workers=len(self.devices), thread_name_prefix="trashpanda-replica")

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "ReplicaSet":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def _sync(self) -> None:
        """Copy current primary parameters to every other replica and clear their gradients."""
        with torch.no_grad():
            for replica in self.replicas[1:]:
                replica.zero_grad(set_to_none=True)
                for target, source in zip(replica.parameters(), self.primary.parameters()):
                    target.copy_(source)

    def _run(self, slot: int, microbatches: Sequence[tuple[Any, Any]], accumulation_steps: int) -> list[tuple[int, float]]:
        model = self.replicas[slot]
        device = self.devices[slot]
        losses: list[tuple[int, float]] = []
        # Autocast state is thread-local, so it is entered inside the worker.
        for index in range(slot, len(microbatches), len(self.devices)):
            inputs, targets = microbatches[index]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=self._autocast_dtype, enabled=self._autocast_enabled and device.type == "cuda"):
                loss = causal_loss(model, inputs, targets)
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError("non-finite loss encountered")
            losses.append((index, float(loss.detach().float().item())))
            (loss / accumulation_steps).backward()
        return losses

    def accumulate(self, microbatches: Sequence[tuple[Any, Any]], accumulation_steps: int) -> list[float]:
        """Backpropagate every microbatch of one update; leave summed gradients on the primary.

        Microbatch ``i`` runs on device ``i % len(devices)``.  The caller zeroes
        the primary gradients first and owns clipping and the optimizer step.
        Returns the per-microbatch losses in sampler order.
        """
        self._sync()
        futures = [self._pool.submit(self._run, slot, microbatches, accumulation_steps) for slot in range(len(self.devices))]
        # Collect every worker before raising so no thread is still writing
        # gradients when an error propagates.
        outcomes: list[Any] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except BaseException as exc:  # re-raised below
                outcomes.append(exc)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        with torch.no_grad():
            for replica in self.replicas[1:]:
                for target, source in zip(self.primary.parameters(), replica.parameters()):
                    if source.grad is None:
                        continue
                    moved = source.grad.to(target.device)
                    if target.grad is None:
                        target.grad = moved.clone()
                    else:
                        target.grad.add_(moved)
        ordered = sorted(pair for outcome in outcomes for pair in outcome)
        return [value for _, value in ordered]


__all__ = ["ReplicaSet", "resolve_devices", "validate_devices"]
