#!/usr/bin/env python3
"""Load a trashpanda export with Hugging Face transformers and generate text.

An export (``trashpanda export``) is a standard Hugging Face model folder, so
this script needs only torch and transformers, the same as anyone you hand the
model to. ``--check`` also needs trashpanda: it rebuilds the model from the
training checkpoint and confirms the export gives the same logits and the same
greedy tokens (``trashpanda.export.compare_export``).

  python scripts/run_export.py /data/runs/40m-titanx-export
  python scripts/run_export.py /data/runs/40m-titanx-export --prompt "Photosynthesis is" --greedy
  python scripts/run_export.py /data/runs/40m-titanx-export \\
      --check /data/runs/40m-titanx/checkpoint-00012231.pt
"""
import argparse

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = ["The water cycle", "In 1905, Albert Einstein", "Photosynthesis is"]


def dtype_kwargs(dtype):
    # transformers renamed torch_dtype to dtype in 4.56
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="folder written by `trashpanda export`")
    parser.add_argument("--prompt", action="append", help="repeatable; default: three sample prompts")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--greedy", action="store_true", help="always take the most likely token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check", metavar="CHECKPOINT", help="confirm the export matches this trashpanda checkpoint")
    args = parser.parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS

    tokenizer = AutoTokenizer.from_pretrained(args.export, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.export, local_files_only=True, **dtype_kwargs(torch.float32))
    model = model.to(args.device).eval()
    parameters = sum(t.numel() for t in model.parameters())
    print(f"loaded {model.config.architectures[0]}: {parameters / 1e6:.1f}M parameters, "
          f"context {model.config.max_position_embeddings} tokens, on {args.device}")

    if args.check:
        from trashpanda.export import compare_export

        report = compare_export(args.check, args.export, prompt=prompts[0], device=args.device)
        print(f"check: max logit difference {report['max_abs_logit_diff']:.2e}, "
              f"greedy tokens identical: {report['greedy_match']}")
        if not report["ok"]:
            raise SystemExit("export does NOT match the checkpoint: " + "; ".join(report["problems"]))

    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for prompt in prompts:
        torch.manual_seed(args.seed)
        ids = tokenizer(prompt, return_tensors="pt").to(args.device)
        sampling = {} if args.greedy else {"do_sample": True, "temperature": args.temperature, "top_k": args.top_k}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens, pad_token_id=pad,
                                 **({"do_sample": False} if args.greedy else sampling))
        print(f"\n--- {prompt}\n{tokenizer.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
