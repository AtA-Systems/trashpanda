"""Private worker entry point used by the local subprocess runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from .bundles import load_round_bundle
from .config import FederationConfig
from .participant import train_local_round


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="trashpanda federation worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--round", dest="round_path", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-after-steps", type=int)
    args = parser.parse_args(argv)
    config = FederationConfig.from_file(args.config)
    participant = next((item for item in config.participants if item.participant_id == args.participant), None)
    if participant is None:
        raise ValueError(f"unknown participant: {args.participant}")
    import torch
    torch.set_num_threads(1)
    path = train_local_round(load_round_bundle(args.round_path), config, participant, args.private, args.output, fail_after_steps=args.fail_after_steps)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
