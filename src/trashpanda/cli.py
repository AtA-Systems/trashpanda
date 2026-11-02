"""Command-line entry points for the issue contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from dataclasses import replace

from .config import MODEL_ARCHITECTURES, load_config, model_recipe, resolve_training_config
from .data import SequenceReader
from .prepare import prepare_splits
from .provenance import inspect_bundle, load_policy, verify_bundle
from .tokenizer import TinyWordLevelTokenizer, load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trashpanda", description="Deterministic CPU/GPU causal-LM training")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare train/validation JSONL into uint16 shards")
    prepare.add_argument("--train", type=Path, action="append", required=True)
    prepare.add_argument("--validation", type=Path, action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--recipe", choices=("tiny", "300m-a100"), default="tiny")
    prepare.add_argument("--model", choices=MODEL_ARCHITECTURES, default="llama")
    prepare.add_argument("--tokenizer")
    prepare.add_argument("--revision")
    prepare.add_argument("--text-field", default="text")
    prepare.add_argument("--shard-tokens", type=int, default=1 << 20)
    prepare.add_argument("--source-descriptor", type=Path, action="append", help="JSON object, one per --train/--validation input; records source_id, revision, selection, terms, and evidence_state")

    train = commands.add_parser("train", help="token-budgeted training")
    train.add_argument("--data", type=Path)
    train.add_argument("--recipe", choices=("tiny", "300m-a100"), default=None)
    train.add_argument("--model", choices=MODEL_ARCHITECTURES, default=None)
    train.add_argument("--output", type=Path)
    train.add_argument("--tokens", type=int)
    train.add_argument("--microbatch-size", type=int)
    train.add_argument("--no-repeat-data", action="store_true")
    train.add_argument("--resume", type=Path)
    train.add_argument("--config", type=Path, help="strict flat JSON override file")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--policy", type=Path, help="adopter policy JSON to evaluate before optimizer work")

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "validation"), default="validation")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--max-batches", type=int)
    evaluate.add_argument("--device", default="cpu")

    sample = commands.add_parser("sample", help="sample from a checkpoint")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--prompt", default="")
    sample.add_argument("--max-new-tokens", type=int, default=32)
    sample.add_argument("--temperature", type=float, default=1.0)
    sample.add_argument("--top-k", type=int)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--device", default="cpu")

    export = commands.add_parser("export", help="export native Hugging Face model files")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--tokenizer")
    export.add_argument("--allow-download", action="store_true")

    show = commands.add_parser("show-config", help="print a resolved recipe")
    show.add_argument("--recipe", choices=("tiny", "300m-a100"), default="tiny")
    show.add_argument("--model", choices=MODEL_ARCHITECTURES, default="llama")
    show.add_argument("--config", type=Path)

    provenance = commands.add_parser("provenance", help="inspect or verify an offline provenance bundle")
    provenance_commands = provenance.add_subparsers(dest="provenance_command", required=True)
    inspect = provenance_commands.add_parser("inspect", help="read provenance without loading model weights")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--json", action="store_true", help="emit stable JSON")
    verify = provenance_commands.add_parser("verify", help="verify payload hashes and evidence references offline")
    verify.add_argument("path", type=Path)
    verify.add_argument("--policy", type=Path, help="optional adopter policy JSON")
    verify.add_argument("--strict", action="store_true", help="fail incomplete or unevaluated optional evidence")
    verify.add_argument("--json", action="store_true", help="emit stable JSON")
    return parser


def _tokenizer_for(config: Any, explicit: str | None = None, *, allow_download: bool = False) -> Any:
    if explicit:
        return load_tokenizer(explicit, revision=config.data.tokenizer_revision, vocab_size=config.model.vocab_size, local_files_only=not allow_download)
    return load_tokenizer(config.data.tokenizer_name, revision=config.data.tokenizer_revision, vocab_size=config.model.vocab_size, local_files_only=True)


def _checkpoint_config(checkpoint: Path) -> Any:
    import torch
    from .checkpoint import _register_safe_checkpoint_globals
    from .config import config_from_dict

    _register_safe_checkpoint_globals()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return config_from_dict(raw["config"])


def _latest_checkpoint(path: Path) -> Path:
    if path.is_dir():
        from .checkpoint import list_checkpoints

        checkpoints = list_checkpoints(path)
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints found in {path}")
        return checkpoints[-1]
    return path


def _with_model(config: Any, architecture: str) -> Any:
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(f"unsupported model architecture {architecture!r}")
    if architecture == "gpt2" and config.recipe_name != "tiny":
        raise ValueError("gpt2 only supports the tiny recipe")
    return replace(config, model=replace(config.model, architecture=architecture))


def _checkpoint_model(checkpoint: Path) -> tuple[Any, Any, Any]:
    from .checkpoint import load_checkpoint
    from .model import build_model

    config = _checkpoint_config(checkpoint)
    model = build_model(config.model, seed=config.training.seed)
    load_checkpoint(checkpoint, model=model, map_location="cpu")
    try:
        tokenizer = _tokenizer_for(config)
    except RuntimeError:
        tokenizer = TinyWordLevelTokenizer()
    return config, model, tokenizer


def _tokenizer_from_prepared(data: Path, config: Any) -> Any:
    snapshot = data / "tokenizer"
    if snapshot.is_dir() and config.model.vocab_size == 32:
        return TinyWordLevelTokenizer()
    if snapshot.is_dir():
        return load_tokenizer(str(snapshot), revision=config.data.tokenizer_revision, vocab_size=config.model.vocab_size, local_files_only=True)
    return _tokenizer_for(config)


def _reader(path: Path, split: str) -> SequenceReader:
    try:
        return SequenceReader(path, split=split)
    except KeyError:
        if split != "train":
            return SequenceReader(path, split="train")
        raise


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "provenance":
        if args.provenance_command == "inspect":
            result = inspect_bundle(args.path)
            if args.json:
                print(json.dumps(result, sort_keys=True, indent=2))
            else:
                print(f"artifact: {result.get('artifact_id')}")
                print(f"type: {result.get('artifact_type')}")
                print(f"records: {len(result.get('records', []))}")
                for record in result.get("records", []):
                    print(f"- {record.get('type')}: {record.get('id')}")
            return 0
        policy = load_policy(args.policy) if args.policy else None
        result = verify_bundle(args.path, policy=policy)
        output = result.to_dict()
        if args.json:
            print(json.dumps(output, sort_keys=True, indent=2))
        else:
            for name in ("schema", "integrity", "completeness", "signatures", "policy"):
                print(f"{name}: {output[name]['status']}")
                for reason in output[name].get("reasons", []):
                    print(f"  {reason.get('code', 'reason')}: {reason.get('message', reason)}")
        if not result.ok or args.strict and (result.completeness["status"] != "complete" or result.policy.get("decision", "not_evaluated") == "not_evaluated"):
            return 2
        return 0
    if args.command == "show-config":
        config = load_config(args.recipe, config_json=args.config)
        print(json.dumps(_with_model(config, args.model).to_dict(), sort_keys=True, indent=2))
        return 0

    if args.command == "prepare":
        config = model_recipe(args.recipe, args.model)
        tokenizer = _tokenizer_for(config, args.tokenizer)
        result = prepare_splits(
            args.output,
            train_path=args.train,
            validation_path=args.validation,
            tokenizer=tokenizer,
            context_length=config.model.context_length,
            vocab_size=config.model.vocab_size,
            text_field=args.text_field,
            shard_tokens=args.shard_tokens,
            source_descriptors=[json.loads(path.read_text(encoding="utf-8")) for path in args.source_descriptor] if args.source_descriptor else None,
        )
        print(json.dumps({split: info.to_dict() for split, info in result.items()}, sort_keys=True, indent=2))
        return 0

    if args.command == "train":
        if args.resume is None and (args.data is None or args.output is None):
            raise SystemExit("new train requires --data and --output")
        if args.resume is not None:
            saved_config = _checkpoint_config(_latest_checkpoint(args.resume))
            if args.model is not None and saved_config.model.architecture != args.model:
                raise ValueError(
                    f"resume checkpoint uses {saved_config.model.architecture}, but --model selects {args.model}"
                )
            if args.recipe is not None and saved_config.recipe_name != args.recipe:
                raise ValueError(
                    f"resume checkpoint uses recipe {saved_config.recipe_name!r}, but --recipe selects {args.recipe!r}"
                )
            config = saved_config
        else:
            config = load_config(args.recipe or "tiny", config_json=args.config)
            config = _with_model(config, args.model or "llama")
        config = resolve_training_config(
            config,
            token_budget=args.tokens,
            microbatch_size=args.microbatch_size,
            repeat_data=False if args.no_repeat_data else None,
        )
        data_path = args.data
        if data_path is None and args.resume is not None:
            data_path = Path(config.data.train_path) if config.data.train_path else None
        if data_path is None:
            raise SystemExit("resume requires --data unless the resolved config records data.train_path")
        dataset = SequenceReader(data_path, split="train")
        tokenizer = _tokenizer_from_prepared(data_path, config)
        from .train import train as run_training

        result = run_training(config, dataset, tokenizer=tokenizer, resume=args.resume, checkpoint_dir=args.output or args.resume, max_steps=args.max_steps, policy=load_policy(args.policy) if args.policy else None)
        print(json.dumps({"state": result.state.to_dict(), "checkpoint": str(result.checkpoint), "device": result.device}, sort_keys=True))
        return 0

    if args.command == "evaluate":
        _config, model, _tokenizer = _checkpoint_model(args.checkpoint)
        from .evaluate import evaluate as run_evaluation

        result = run_evaluation(model, _reader(args.data, args.split), batch_size=args.batch_size, max_batches=args.max_batches, device=args.device)
        print(json.dumps(result.__dict__, sort_keys=True))
        return 0

    if args.command == "sample":
        _config, model, tokenizer = _checkpoint_model(args.checkpoint)
        from .evaluate import sample_text

        print(sample_text(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, seed=args.seed, device=args.device))
        return 0

    if args.command == "export":
        from .export import export_checkpoint

        config = _checkpoint_config(args.checkpoint)
        tokenizer = _tokenizer_for(config, args.tokenizer, allow_download=args.allow_download) if args.tokenizer else None
        export_checkpoint(args.checkpoint, args.output, tokenizer=tokenizer, local_files_only=not args.allow_download)
        print(json.dumps({"output": str(args.output)}, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except SystemExit:
        raise
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"trashpanda: error: {exc}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "main"]
