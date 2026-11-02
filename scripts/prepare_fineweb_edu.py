#!/usr/bin/env python3
"""Stream a bounded FineWeb-Edu slice into trashpanda's local JSONL contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trashpanda.config import MISTRAL_TOKENIZER_NAME, MISTRAL_TOKENIZER_REVISION, get_recipe
from trashpanda.prepare import prepare_splits
from trashpanda.tokenizer import load_tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="prepared dataset output directory")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", help="immutable dataset revision; omitted means the adapter records unknown")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--train-documents", type=int, default=30_000)
    parser.add_argument("--validation-documents", type=int, default=1_000)
    parser.add_argument("--tokenizer", default=MISTRAL_TOKENIZER_NAME)
    parser.add_argument("--tokenizer-revision", default=MISTRAL_TOKENIZER_REVISION)
    parser.add_argument("--tokenizer-cache", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    if args.train_documents <= 0 or args.validation_documents <= 0:
        parser.error("document counts must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("install the optional datasets dependency for FineWeb-Edu preparation") from exc

    dataset = load_dataset(args.dataset, name=args.config, split=args.split, revision=args.revision, streaming=True)
    resolved_revision = args.revision or getattr(dataset, "_fingerprint", None) or "unknown"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fineweb-jsonl-", dir=args.output.parent) as temp:
        train_path = Path(temp) / "train.jsonl"
        validation_path = Path(temp) / "validation.jsonl"
        train_count = validation_count = 0
        with train_path.open("w", encoding="utf-8") as train, validation_path.open("w", encoding="utf-8") as valid:
            for record_number, record in enumerate(dataset, 1):
                text = record.get(args.text_field) if isinstance(record, dict) else None
                if not isinstance(text, str) or text == "":
                    raise ValueError(f"FineWeb-Edu record {record_number} lacks a nonempty string {args.text_field!r}")
                if train_count < args.train_documents:
                    train.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    train_count += 1
                elif validation_count < args.validation_documents:
                    valid.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    validation_count += 1
                if validation_count >= args.validation_documents:
                    break
        if train_count < args.train_documents or validation_count < args.validation_documents:
            raise RuntimeError("FineWeb-Edu stream ended before both requested splits were filled")
        config = get_recipe("300m-a100")
        tokenizer_name = str(args.tokenizer_cache) if args.tokenizer_cache else args.tokenizer
        tokenizer = load_tokenizer(
            tokenizer_name,
            revision=args.tokenizer_revision,
            vocab_size=config.model.vocab_size,
            local_files_only=not args.allow_download,
        )
        prepared = prepare_splits(
            args.output,
            train_path=train_path,
            validation_path=validation_path,
            tokenizer=tokenizer,
            context_length=config.model.context_length,
            vocab_size=config.model.vocab_size,
            text_field="text",
            source_descriptors=[
                {
                    "source_id": f"hf:{args.dataset}:{args.config}:{resolved_revision}:train",
                    "source": args.dataset,
                    "source_revision": {"requested": args.revision, "resolved": resolved_revision},
                    "selection": {
                        "split": args.split,
                        "train_documents": args.train_documents,
                        "validation_documents": args.validation_documents,
                        "text_field": args.text_field,
                    },
                    "evidence_state": "observed" if args.revision else "revision-unavailable",
                },
                {
                    "source_id": f"hf:{args.dataset}:{args.config}:{resolved_revision}:validation",
                    "source": args.dataset,
                    "source_revision": {"requested": args.revision, "resolved": resolved_revision},
                    "selection": {
                        "split": args.split,
                        "train_documents": args.train_documents,
                        "validation_documents": args.validation_documents,
                        "text_field": args.text_field,
                    },
                    "evidence_state": "observed" if args.revision else "revision-unavailable",
                },
            ],
        )
    print(json.dumps({"dataset": args.dataset, "requested_revision": args.revision, "resolved_revision": resolved_revision, "train_documents": train_count, "validation_documents": validation_count, **{key: value.to_dict() for key, value in prepared.items()}}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
