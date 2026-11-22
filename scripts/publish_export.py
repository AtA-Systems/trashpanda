#!/usr/bin/env python3
"""Publish a trashpanda export to a Hugging Face model repository.

  python scripts/publish_export.py /data/runs/40m-titanx-export \\
      --repo ata-systems/trashpanda-40m-titanx --dataset HuggingFaceFW/fineweb-edu --dry-run
  python scripts/publish_export.py /data/runs/40m-titanx-export \\
      --repo ata-systems/trashpanda-40m-titanx --dataset HuggingFaceFW/fineweb-edu

  python scripts/publish_export.py --verify-repo ata-systems/trashpanda-40m-titanx

What it does:
  1. Verifies the export's provenance bundle (`trashpanda provenance verify`).
     Publishing stops if verification fails.
  2. Copies the export to a staging folder and adds a model card (README.md)
     built from trashpanda-provenance.json. The export itself is not modified,
     so its manifest still verifies afterwards.
  3. Creates the repo if needed (PRIVATE unless --public) and uploads the
     folder in one commit.
  4. Downloads the repo back and verifies the provenance bundle again
     (--verify-repo does only this step, for any published repo).

The verifier rejects files the bundle doesn't list, and the Hub adds
README.md and .gitattributes, so verification of a download runs on a copy
without those two files.

Authentication: the token from `hf auth login` (or HF_TOKEN). It needs write
access to the target organization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def verify(export: Path) -> dict:
    result = subprocess.run([sys.executable, "-m", "trashpanda", "provenance", "verify", str(export), "--json"],
                            text=True, capture_output=True)
    if result.returncode:
        raise SystemExit(f"provenance verification failed; not publishing.\n{result.stdout}\n{result.stderr}")
    return json.loads(result.stdout)


def verify_repo(repo: str, revision: str | None = None) -> None:
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo, repo_type="model", revision=revision))
    with tempfile.TemporaryDirectory(prefix="tp-verify-") as temp:
        copy = Path(temp) / "bundle"
        shutil.copytree(snapshot, copy, ignore=shutil.ignore_patterns("README.md", ".gitattributes"))
        verify(copy)
    print(f"verified: {repo} matches its provenance bundle")


def _records(export: Path) -> dict[str, dict]:
    """Provenance records in the export, keyed by type (last one wins)."""
    found: dict[str, dict] = {}
    for path in sorted((export / "provenance" / "records").glob("*.json")):
        record = json.loads(path.read_text())
        found[str(record.get("type"))] = record
    return found


def model_card(export: Path, repo: str, license_id: str, datasets: list[str], evaluation: dict | None, recipe: str | None) -> str:
    hf = json.loads((export / "config.json").read_text())
    records = _records(export)
    manifest = json.loads((export / "provenance-manifest.json").read_text()) if (export / "provenance-manifest.json").exists() else {}
    from safetensors import safe_open

    parameters = 0
    with safe_open(str(export / "model.safetensors"), "pt") as handle:
        for key in handle.keys():
            count = 1
            for dim in handle.get_slice(key).get_shape():
                count *= dim
            parameters += count
    budget = (records.get("run") or {}).get("requested_budget")
    checkpoint_sha = (records.get("artifact") or {}).get("source_checkpoint_sha256", "?")
    front = ["---", f"license: {license_id}", "library_name: transformers", "pipeline_tag: text-generation",
             "tags:", "- trashpanda", f"- {hf.get('model_type', 'llama')}", "- base-model"]
    if datasets:
        front += ["datasets:"] + [f"- {d}" for d in datasets]
    front += ["---", ""]
    rows = []
    if recipe:
        rows.append(("Recipe", f"`{recipe}`"))
    rows += [
        ("Architecture", ", ".join(hf.get("architectures", [])) or "unknown"),
        ("Parameters", f"{parameters:,}"),
        ("Layers / hidden size / heads", f"{hf.get('num_hidden_layers', '?')} / {hf.get('hidden_size', '?')} / {hf.get('num_attention_heads', '?')}"),
        ("Context length", f"{hf.get('max_position_embeddings', '?')} tokens"),
        ("Vocabulary", f"{hf.get('vocab_size', '?')} tokens"),
        ("Weights", str(hf.get("torch_dtype") or hf.get("dtype") or "float32")),
    ]
    if isinstance(budget, (int, float)):
        rows.append(("Training token budget", f"{int(budget):,}"))
    elif isinstance(budget, dict) and budget:
        rows.append(("Training budget", ", ".join(f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}" for k, v in budget.items())))
    rows.append(("Source checkpoint SHA-256", f"`{checkpoint_sha}`"))
    if manifest.get("artifact_id"):
        rows.append(("Provenance artifact id", f"`{manifest['artifact_id']}`"))
    if evaluation:
        rows.append(("Validation loss", f"{evaluation.get('loss', float('nan')):.3f} (perplexity {evaluation.get('perplexity', float('nan')):.1f})"))
    table = "| | |\n| --- | --- |\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
    body = f"""# {repo.split('/')[-1]}

A small base language model trained from scratch with [trashpanda](https://github.com/AtA-Systems/trashpanda).
It continues text; it is not instruction- or chat-tuned, and it makes up facts.

{table}

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForCausalLM.from_pretrained("{repo}")
ids = tok("The water cycle", return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=64, do_sample=True, temperature=0.7, top_k=40)[0]))
```

Serve with vLLM (compute capability 7.0+; completions only, no chat template):

```bash
vllm serve {repo} --max-model-len {hf.get('max_position_embeddings', 512)}
```

## Provenance

This repository includes trashpanda's provenance bundle (`provenance-manifest.json`,
`trashpanda-provenance.json` and `provenance/records/`), which links these weights to
the training checkpoint, run configuration, prepared data and tokenizer. Verify what
you downloaded with `python scripts/publish_export.py --verify-repo {repo}` from a
trashpanda checkout; it checks the files against the bundle, leaving out this model
card and the Hub's `.gitattributes`, which are not part of it.
These records are evidence, not a certification of data rights or model quality.
"""
    return "\n".join(front) + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, nargs="?", help="folder written by `trashpanda export`")
    parser.add_argument("--repo", help="e.g. ata-systems/trashpanda-40m-titanx")
    parser.add_argument("--verify-repo", metavar="REPO", help="only download REPO and verify its provenance")
    parser.add_argument("--license", default="apache-2.0", help="model card license id (default apache-2.0)")
    parser.add_argument("--dataset", action="append", default=[], help="Hub dataset id used for training (repeatable)")
    parser.add_argument("--recipe", help="trashpanda recipe name to show in the card, e.g. 40m-titanx")
    parser.add_argument("--evaluation", type=Path, help="JSON from `trashpanda evaluate` to show in the card")
    parser.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    parser.add_argument("--message", default=None, help="commit message")
    parser.add_argument("--dry-run", action="store_true", help="verify and write the staged folder, upload nothing")
    args = parser.parse_args()
    if args.verify_repo:
        verify_repo(args.verify_repo)
        return 0
    if not args.export or not args.repo:
        parser.error("give an export folder and --repo (or --verify-repo REPO)")

    export = args.export.resolve()
    for required in ("config.json", "model.safetensors"):
        if not (export / required).exists():
            raise SystemExit(f"{export} is not a trashpanda export (missing {required})")
    verify(export)
    print("provenance: verified")
    evaluation = json.loads(args.evaluation.read_text()) if args.evaluation else None

    stage_root = Path(tempfile.mkdtemp(prefix="tp-publish-"))
    stage = stage_root / export.name
    shutil.copytree(export, stage)
    (stage / "README.md").write_text(model_card(export, args.repo, args.license, args.dataset, evaluation, args.recipe), encoding="utf-8")
    files = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file())
    size = sum((stage / f).stat().st_size for f in files)
    print(f"staged {len(files)} files, {size / 1e6:.1f} MB, in {stage}")
    for f in files:
        print("  ", f)
    if args.dry_run:
        print(f"dry run: would create {'public' if args.public else 'PRIVATE'} repo {args.repo} and upload; README.md preview:\n")
        print((stage / "README.md").read_text())
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()
    print(f"authenticated as {user.get('name')}; orgs: {', '.join(o.get('name', '?') for o in user.get('orgs', [])) or 'none'}")
    url = api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(stage),
                               commit_message=args.message or f"Upload {export.name} from trashpanda")
    print(f"published: {url}\ncommit: {getattr(commit, 'commit_url', commit)}")
    shutil.rmtree(stage_root, ignore_errors=True)
    verify_repo(args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
