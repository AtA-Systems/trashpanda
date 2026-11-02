#!/usr/bin/env python3
"""Train the 100-parameter project-owned provenance acceptance fixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trashpanda.provenance import build_bundle_manifest, make_record, sha256_file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory containing only the portable synthetic bundle")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("install PyTorch to run the 100-parameter provenance fixture") from exc

    with tempfile.TemporaryDirectory(prefix="trashpanda-provenance-demo-") as temporary:
        work = Path(temporary)
        sources = []
        for name, values in (("A", [0, 1, 2, 3, 0, 1]), ("B", [3, 2, 1, 0, 3, 2])):
            path = work / f"source-{name}.jsonl"
            path.write_text("".join(json.dumps({"token": value}) + "\n" for value in values), encoding="utf-8")
            sources.append(
                make_record(
                    "input",
                    {
                        "source_id": f"synthetic:{name}",
                        "source": "project-owned-fixture",
                        "content_sha256": sha256_file(path),
                        "source_revision": "fixture-1",
                        "selection": {"records": len(values)},
                        "evidence_state": "observed",
                    },
                )
            )

        preparation = make_record(
            "preparation",
            {
                "split": "train",
                "sources": [record["id"] for record in sources],
                "tokenizer": {"name": "synthetic-integer", "vocab_size": 8, "sha256": "fixture"},
                "settings": {"context_length": 4, "target_rule": "(token + 1) mod 4"},
                "evidence_state": "observed",
            },
        )

        class Tiny100(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(8, 8)
                self.output = nn.Linear(8, 4)

            def forward(self, values: torch.Tensor) -> torch.Tensor:
                return self.output(self.embedding(values))

        torch.manual_seed(7)
        model = Tiny100()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert parameter_count == 100, parameter_count
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        inputs = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=torch.long)
        targets = (inputs % 4 + 1) % 4
        run_id = "synthetic-run-100"
        segments = []
        start_time = _now()
        start = time.perf_counter()
        committed = 0
        parent_checkpoint_digest = None
        for segment_number in (1, 2):
            segment_start = committed
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(inputs).reshape(-1, 4), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                committed += int(targets.numel())
            checkpoint = work / "resume.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "committed": committed}, checkpoint)
            segments.append(
                make_record(
                    "execution_segment",
                    {
                        "run_id": run_id,
                        "session_id": f"synthetic-session-{segment_number}",
                        "status": "completed",
                        "start_time_utc": start_time,
                        "end_time_utc": _now(),
                        "elapsed_seconds": time.perf_counter() - start,
                        "start_counters": {"committed_targets": segment_start},
                        "end_counters": {"committed_targets": committed},
                        "parent_checkpoint_sha256": parent_checkpoint_digest,
                        "input_records": [record["id"] for record in sources] + [preparation["id"]],
                        "software": {"python": "fixture", "torch": torch.__version__},
                        "config_sha256": "synthetic-config",
                        "evidence_state": "observed",
                    },
                )
            )
            parent_checkpoint_digest = sha256_file(checkpoint)

        model_path = args.output / "model.pt"
        args.output.mkdir(parents=True)
        torch.save(model.state_dict(), model_path)
        (args.output / "config.json").write_text(json.dumps({"parameter_count": parameter_count, "vocab_size": 8}, indent=2) + "\n", encoding="utf-8")
        run = make_record(
            "run",
            {
                "run_id": run_id,
                "status": "completed",
                "segments": [segment["id"] for segment in segments],
                "parents": [],
                "cumulative_counters": {"committed_targets": committed},
                "actual_target_accounting": {
                    "committed_targets": committed,
                    "source_target_counts": {"synthetic:A": committed // 2, "synthetic:B": committed // 2},
                },
                "requested_budget": 64,
                "requested_budget_is_evidence": False,
                "input_records": [record["id"] for record in sources] + [preparation["id"]],
                "software": {"python": "fixture", "torch": torch.__version__},
                "config_sha256": "synthetic-config",
                "evidence_state": "observed",
            },
        )
        artifact = make_record(
            "artifact",
            {
                "artifact_type": "synthetic-model",
                "model_sha256": sha256_file(model_path),
                "parameter_count": parameter_count,
                "records": [record["id"] for record in sources] + [preparation["id"], run["id"]],
                "evidence_state": "observed",
            },
        )
        build_bundle_manifest(args.output, records=[*sources, preparation, *segments, run, artifact], artifact_type="synthetic-model")
    print(json.dumps({"output": str(args.output), "parameter_count": 100, "committed_targets": committed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
