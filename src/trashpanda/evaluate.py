"""Offline evaluation and deterministic autoregressive sampling."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from .data import MMapTokenDataset
from .provenance import make_record


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    perplexity: float | None
    tokens: int
    batches: int
    perplexity_overflow: bool = False


def evaluation_record(
    result: EvaluationResult,
    *,
    split: str,
    settings: dict[str, Any],
    input_identity: str | None = None,
    model_identity: str | None = None,
) -> dict[str, Any]:
    """Turn observed evaluation output into a portable, explicit record."""
    return make_record(
        "evaluation",
        {
            "split": split,
            "settings": dict(settings),
            "counts": {"tokens": result.tokens, "batches": result.batches},
            "metrics": {"loss": result.loss, "perplexity": result.perplexity, "perplexity_overflow": result.perplexity_overflow},
            "input_identity": input_identity,
            "model_identity": model_identity,
            "evidence_state": "observed",
        },
    )


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("evaluation requires PyTorch")


def evaluate(
    model: Any,
    dataset: MMapTokenDataset,
    *,
    batch_size: int = 1,
    max_batches: int | None = None,
    device: str | Any = "cpu",
) -> EvaluationResult:
    _require_torch()
    if len(dataset) == 0:
        raise ValueError("evaluation dataset is empty")
    device = torch.device(device)
    model.to(device)
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            if max_batches is not None and batches >= max_batches:
                break
            indices = list(range(start, min(start + batch_size, len(dataset))))
            inputs, targets = dataset.batch(indices)
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(input_ids=inputs).logits
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
            if not bool(torch.isfinite(nll).item()):
                raise FloatingPointError("non-finite evaluation loss")
            total_nll += float(nll.item())
            total_tokens += int(targets.numel())
            batches += 1
    if total_tokens == 0:
        raise ValueError("evaluation produced no target tokens")
    loss = total_nll / total_tokens
    try:
        perplexity = math.exp(loss)
        overflow = False
    except OverflowError:
        perplexity = None
        overflow = True
    return EvaluationResult(loss=loss, perplexity=perplexity, tokens=total_tokens, batches=batches, perplexity_overflow=overflow)


def sample(
    model: Any,
    prompt: Sequence[int],
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    device: str | Any = "cpu",
) -> list[int]:
    """Sample IDs with an isolated generator, so global RNG state is unchanged."""
    _require_torch()
    if max_new_tokens < 0 or temperature <= 0:
        raise ValueError("invalid sampling parameters")
    device = torch.device(device)
    model.to(device)
    model.eval()
    generated = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    context_length = int(getattr(model.config, "context_length", model.config.max_position_embeddings))
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[:, -context_length :]
            logits = model(input_ids=context).logits[:, -1, :] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive")
                k = min(top_k, logits.shape[-1])
                values, indices = torch.topk(logits, k)
                filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
                filtered.scatter_(1, indices, values)
                logits = filtered
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            generated = torch.cat((generated, next_token), dim=1)
    return [int(value) for value in generated[0].tolist()]


def sample_text(model: Any, tokenizer: Any, prompt: str, **kwargs: Any) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.decode(sample(model, ids, **kwargs))


__all__ = ["EvaluationResult", "evaluate", "evaluation_record", "sample", "sample_text"]
