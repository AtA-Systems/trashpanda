"""Native Hugging Face model export without custom code or remote execution."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
from typing import Any, Mapping

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from .checkpoint import load_checkpoint
from .config import Config, config_from_dict
from .model import build_model
from .provenance import build_bundle_manifest, make_record, sha256_file
from .tokenizer import TinyWordLevelTokenizer, load_tokenizer


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("export requires PyTorch")


def _config_from_dict(value: Mapping[str, Any]) -> Config:
    return config_from_dict(value)


def export_hf(
    model: Any,
    output_dir: str | Path,
    *,
    config: Config,
    tokenizer: Any | None = None,
    provenance: Mapping[str, Any] | None = None,
    write_manifest: bool = True,
) -> Path:
    """Call the real HF ``save_pretrained`` path for the selected adapter."""
    _require_torch()
    expected_class = {"llama": "LlamaForCausalLM", "gpt2": "GPT2LMHeadModel"}[config.model.architecture]
    if not hasattr(model, "save_pretrained") or model.__class__.__name__ != expected_class:
        raise TypeError(f"export_hf requires the selected {config.model.architecture} model ({expected_class})")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"export output {output} already exists; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=False)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    if tokenizer is None:
        tokenizer = TinyWordLevelTokenizer() if config.model.vocab_size == 32 else None
    if tokenizer is None or not hasattr(tokenizer, "save_pretrained"):
        raise ValueError("production export requires the locally cached Mistral tokenizer")
    tokenizer.save_pretrained(output)
    if write_manifest:
        records = [dict(record) for record in (provenance or {}).get("records", []) if isinstance(record, Mapping)]
        records.append(
            make_record(
                "artifact",
                {
                    "artifact_type": "hf-export",
                    "initialization": "unknown",
                    "config_sha256": sha256_file(output / "config.json"),
                    "evidence_state": "incomplete-direct-export",
                },
            )
        )
        build_bundle_manifest(output, records=records, artifact_type="hf-export")
    return output


def _safe_load(path: str | Path) -> dict[str, Any]:
    _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not a mapping")
    return payload


def export_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    tokenizer: Any | None = None,
    tokenizer_name: str | None = None,
    local_files_only: bool = True,
) -> Path:
    raw = _safe_load(checkpoint)
    config = config_from_dict(raw["config"])
    model = build_model(config.model, seed=config.training.seed)
    load_checkpoint(checkpoint, model=model, map_location="cpu")
    if tokenizer is None and tokenizer_name:
        tokenizer = load_tokenizer(tokenizer_name, revision=config.data.tokenizer_revision, vocab_size=config.model.vocab_size, local_files_only=local_files_only)
    if tokenizer is None and raw.get("tokenizer_files"):
        # A normal checkpoint is self-contained: reconstruct the tokenizer from
        # its captured bytes, so export does not need Hub access.
        with tempfile.TemporaryDirectory(prefix="trashpanda-checkpoint-tokenizer-") as temp:
            snapshot_dir = Path(temp)
            for name, content in raw["tokenizer_files"].items():
                target = snapshot_dir / str(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            tokenizer = load_tokenizer(str(snapshot_dir), revision=config.data.tokenizer_revision, vocab_size=config.model.vocab_size, local_files_only=True)
    provenance_records = []
    if isinstance(raw.get("extra"), Mapping):
        provenance_records.extend(
            record for record in raw["extra"].get("provenance", {}).get("records", [])
            if isinstance(record, Mapping)
        )
    provenance_records.append(
        make_record(
            "artifact",
            {
                "artifact_type": "hf-export",
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "derived_by": "export",
                "evidence_state": "observed",
            },
        )
    )
    exported = export_hf(
        model,
        output_dir,
        config=config,
        tokenizer=tokenizer,
        provenance={"records": provenance_records},
        write_manifest=False,
    )
    provenance = {
        "format": "trashpanda.export.provenance.v1",
        "checkpoint": str(Path(checkpoint).name),
        "data_identity": raw.get("data_identity"),
        "config": raw.get("config"),
        "tokenizer": raw.get("tokenizer"),
        "checkpoint_format": raw.get("format"),
        "source_checkpoint_sha256": sha256_file(checkpoint),
    }
    (exported / "trashpanda-provenance.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    build_bundle_manifest(
        exported,
        records=provenance_records,
        artifact_type="hf-export",
        source_checkpoint_sha256=sha256_file(checkpoint),
    )
    return exported


def _dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """transformers renamed ``torch_dtype`` to ``dtype`` in 4.56."""
    import transformers

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def compare_export(
    checkpoint: str | Path,
    export_dir: str | Path,
    *,
    prompt: str = "The water cycle",
    device: str = "cpu",
    tolerance: float = 1e-4,
    new_tokens: int = 12,
) -> dict[str, Any]:
    """Check that an export is the model in ``checkpoint``, as a Hugging Face
    consumer would load it: ``AutoModelForCausalLM`` + ``AutoTokenizer`` from
    the export folder only. Compares logits on ``prompt`` and greedy
    generation token for token. Returns a report; ``report["ok"]`` is the
    verdict and ``report["problems"]`` says what failed."""
    _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    export_dir = Path(export_dir)
    problems: list[str] = []
    tokenizer = AutoTokenizer.from_pretrained(export_dir, local_files_only=True)
    exported = AutoModelForCausalLM.from_pretrained(export_dir, local_files_only=True, **_dtype_kwargs(torch.float32)).to(device).eval()
    architectures = list(getattr(exported.config, "architectures", None) or [])
    raw = _safe_load(checkpoint)
    config = config_from_dict(raw["config"])
    original = build_model(config.model, seed=config.training.seed)
    load_checkpoint(checkpoint, model=original, map_location="cpu")
    original = original.to(device).eval()
    if architectures != [original.__class__.__name__]:
        problems.append(f"export architectures {architectures} != trained {original.__class__.__name__}")
    tied = exported.get_input_embeddings().weight.data_ptr() == exported.get_output_embeddings().weight.data_ptr()
    if config.model.tie_embeddings and not tied:
        problems.append("export lost its tied embeddings")

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        max_diff = float((original(input_ids=ids).logits.float() - exported(input_ids=ids).logits.float()).abs().max())
        greedy_trained = original.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=pad)
        greedy_export = exported.generate(ids, max_new_tokens=new_tokens, do_sample=False, pad_token_id=pad)
    if max_diff > tolerance:
        problems.append(f"logits differ by {max_diff:g} (> {tolerance:g})")
    greedy_match = bool(torch.equal(greedy_trained, greedy_export))
    if not greedy_match:
        problems.append("greedy generation differs")
    return {
        "ok": not problems,
        "problems": problems,
        "architectures": architectures,
        "tied_embeddings": tied,
        "parameters": sum(p.numel() for p in exported.parameters()),
        "context_length": getattr(exported.config, "max_position_embeddings", None),
        "device": device,
        "prompt": prompt,
        "max_abs_logit_diff": max_diff,
        "greedy_match": greedy_match,
        "greedy_text": tokenizer.decode(greedy_export[0], skip_special_tokens=True),
    }


__all__ = ["compare_export", "export_checkpoint", "export_hf"]
