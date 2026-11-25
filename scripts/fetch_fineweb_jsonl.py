#!/usr/bin/env python3
"""Stream a bounded FineWeb-Edu slice to JSONL plus source descriptors.

Unlike prepare_fineweb_edu.py, this stops at JSONL so FineWeb text can be mixed
with other sources (for example collect_apache_repos.py output) in one
`trashpanda prepare` call, each input keeping its own provenance record.

Output: OUT/train.jsonl, OUT/validation.jsonl,
        OUT/descriptor-train.json, OUT/descriptor-validation.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterable


def write_splits(records: Iterable, out: Path, *, train_documents: int, validation_documents: int, text_field: str = "text") -> tuple[int, int]:
    """Write the first N records to train and the next M to validation."""
    out.mkdir(parents=True, exist_ok=False)
    train_count = validation_count = 0
    with (out / "train.jsonl").open("w", encoding="utf-8") as train, (out / "validation.jsonl").open("w", encoding="utf-8") as valid:
        for number, record in enumerate(records, 1):
            text = record.get(text_field) if isinstance(record, dict) else None
            if not isinstance(text, str) or not text:
                raise ValueError(f"record {number} lacks a nonempty string {text_field!r}")
            if train_count < train_documents:
                train.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                train_count += 1
            elif validation_count < validation_documents:
                valid.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                validation_count += 1
            if validation_count >= validation_documents:
                break
    if train_count < train_documents or validation_count < validation_documents:
        raise RuntimeError("stream ended before both splits were filled")
    return train_count, validation_count


def descriptor(dataset: str, config: str, split: str, revision: str, *, kind: str, start: int, count: int) -> dict:
    return {
        "source_id": f"hf:{dataset}/{config}@{revision}:{kind}",
        "source": f"https://huggingface.co/datasets/{dataset}",
        "source_revision": revision,
        "selection": {"kind": "stream-prefix", "config": config, "split": split, "first_record": start, "records": count},
        "terms": "ODC-By-1.0 (FineWeb-Edu); see the dataset card",
        "evidence_state": "observed" if revision != "unknown" else "incomplete",
        "acquired_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", help="immutable dataset revision; omitted means the stream's fingerprint or 'unknown'")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--train-documents", type=int, default=30_000)
    parser.add_argument("--validation-documents", type=int, default=1_000)
    args = parser.parse_args(argv)
    if args.train_documents <= 0 or args.validation_documents <= 0:
        parser.error("document counts must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("install the optional datasets dependency (pip install '.[datasets]')") from exc
    stream = load_dataset(args.dataset, name=args.config, split=args.split, revision=args.revision, streaming=True)
    revision = args.revision or getattr(stream, "_fingerprint", None) or "unknown"
    n_train, n_valid = write_splits(stream, args.output, train_documents=args.train_documents,
                                    validation_documents=args.validation_documents, text_field=args.text_field)
    for kind, start, count in (("train", 0, n_train), ("validation", n_train, n_valid)):
        path = args.output / f"descriptor-{kind}.json"
        path.write_text(json.dumps(descriptor(args.dataset, args.config, args.split, revision, kind=kind, start=start, count=count), indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "train_documents": n_train, "validation_documents": n_valid, "revision": revision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
