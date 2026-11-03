#!/usr/bin/env python3
"""Bounded smoke test for an installed Trashpanda distribution.

The script intentionally runs outside the source checkout in release CI. It
uses only generated text and validates both model adapters plus federation.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(*args: str, cwd: Path) -> str:
    result = subprocess.run([sys.executable, "-m", "trashpanda", *args], cwd=cwd, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise SystemExit(f"failed: {' '.join(args)}\nstdout={result.stdout}\nstderr={result.stderr}")
    return result.stdout


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="trashpanda-release-smoke-") as temporary:
        root = Path(temporary)
        train = root / "train.jsonl"
        validation = root / "validation.jsonl"
        train.write_text("".join(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(256))}) + "\n" for _ in range(8)), encoding="utf-8")
        validation.write_text(json.dumps({"text": " ".join(f"w{i % 29}" for i in range(128))}) + "\n", encoding="utf-8")
        for architecture in ("llama", "gpt2"):
            prepared = root / f"prepared-{architecture}"
            run("prepare", "--model", architecture, "--recipe", "tiny", "--train", str(train), "--validation", str(validation), "--output", str(prepared), cwd=root)
            run_dir = root / f"run-{architecture}"
            run("train", "--model", architecture, "--recipe", "tiny", "--data", str(prepared), "--output", str(run_dir), "--tokens", "744", cwd=root)
            checkpoint = run_dir / "checkpoint-00000006.pt"
            run("evaluate", "--checkpoint", str(checkpoint), "--data", str(prepared), "--max-batches", "1", cwd=root)
            exported = root / f"export-{architecture}"
            run("export", "--checkpoint", str(checkpoint), "--output", str(exported), cwd=root)
            run("provenance", "verify", str(exported), "--json", cwd=root)
        federation = root / "federation"
        run("federate", "demo", "--output", str(federation), cwd=root)
        run("federate", "export", "--run", str(federation / "run"), "--output", str(root / "federated-export"), cwd=root)
        run("provenance", "verify", str(root / "federated-export"), "--json", cwd=root)
    print("release smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
