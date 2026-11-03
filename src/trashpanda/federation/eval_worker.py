from __future__ import annotations

import argparse
from pathlib import Path

from .bundles import load_round_bundle
from .config import FederationConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", dest="round_path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    config = FederationConfig.from_file(args.config)
    bundle = load_round_bundle(args.round_path)
    model = __import__("trashpanda.model", fromlist=["build_model"]).build_model(config.model_config.model, seed=config.seed)
    model.load_state_dict(bundle.state)
    dataset = __import__("trashpanda.data", fromlist=["SequenceReader"]).SequenceReader(args.data, split="validation")
    result = __import__("trashpanda.evaluate", fromlist=["evaluate"]).evaluate(model, dataset, max_batches=args.max_batches)
    import json
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
