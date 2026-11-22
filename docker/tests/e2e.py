#!/usr/bin/env python3
"""End-to-end check of a trashpanda install, through Hugging Face transformers.

Stages: environment -> prepare -> train -> evaluate -> sample -> export ->
provenance verify -> transformers (load the export, compare it with
trashpanda's own model, generate). Every stage must pass; a JSON report is
written to <out>/report.json.

Modes:
  tiny  Offline, CPU training on generated text with the project's 32-token
        tokenizer. Proves the plumbing anywhere, including CI with no GPU.
  gpu   A short 40m-titanx run on CUDA from an already prepared dataset
        (--data). Proves the image works on this host's GPUs and produces an
        export that vLLM can serve (see vllm_smoke.sh).

Examples (inside the image):
  python docker/tests/e2e.py --mode tiny --out /work/e2e-tiny
  python docker/tests/e2e.py --mode gpu --data /data/prepared --gpus 2 --out /out/e2e-gpu
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

GPU_PROMPT = "The water cycle"
TINY_PROMPT = "w0 w1 w2 w3"


class StageFailed(RuntimeError):
    pass


def cli(*args: str, timeout: int) -> str:
    cmd = [sys.executable, "-m", "trashpanda", *args]
    print("  $", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise StageFailed(f"exit {result.returncode}\n--- stdout ---\n{result.stdout[-4000:]}\n--- stderr ---\n{result.stderr[-4000:]}")
    return result.stdout


def last_json(text: str) -> dict:
    """The CLI prints one JSON object (possibly over several lines) last."""
    start = text.rfind("\n{")
    start = 0 if start < 0 and text.lstrip().startswith("{") else start + 1
    return json.loads(text[start:])


def stage_environment(args: argparse.Namespace) -> dict:
    import torch
    import transformers

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "flavor": os.environ.get("TP_FLAVOR", "not-in-image"),
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "devices": [],
    }
    if torch.cuda.is_available():
        archs = set(info["arch_list"])
        for index in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(index)
            sm = f"sm_{major}{minor}"
            # A device runs if its exact sm is built, or PTX for an older sm of
            # the same major can be JIT-compiled.
            runnable = sm in archs or any(a.startswith("compute_") and int(a[8:]) <= major * 10 + minor for a in archs)
            info["devices"].append({"index": index, "name": torch.cuda.get_device_name(index), "capability": f"{major}.{minor}", "built_for_it": runnable})
            if not runnable:
                message = f"GPU {index} ({torch.cuda.get_device_name(index)}, {sm}) is not in this torch build {sorted(archs)}: wrong image flavor for this GPU"
                if args.mode == "gpu":
                    raise StageFailed(message)
                info.setdefault("warnings", []).append(message)
                continue
            # Prove a kernel actually runs on the device.
            x = torch.randn(256, 256, device=f"cuda:{index}")
            float((x @ x).sum())
    if args.mode == "gpu":
        if not torch.cuda.is_available():
            raise StageFailed("gpu mode needs CUDA, and torch sees no GPU (missing --gpus, driver or container toolkit?)")
        if torch.cuda.device_count() < args.gpus:
            raise StageFailed(f"--gpus {args.gpus} requested, {torch.cuda.device_count()} visible")
    return info


def stage_prepare(args: argparse.Namespace, out: Path) -> dict:
    if args.mode == "gpu":
        data = Path(args.data)
        if not (data / "manifest.json").exists() and not any(data.glob("train-*.bin")):
            raise StageFailed(f"{data} does not look like a prepared trashpanda dataset")
        return {"data": str(data), "prepared_here": False}
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, documents, words, offset in (("train", 8, 128, 0), ("validation", 4, 96, 13)):
        with (raw / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for document in range(documents):
                text = " ".join(f"w{(i + document + offset) % 29}" for i in range(words))
                handle.write(json.dumps({"text": text}) + "\n")
    prepared = out / "prepared"
    cli("prepare", "--recipe", "tiny", "--train", str(raw / "train.jsonl"), "--validation", str(raw / "validation.jsonl"), "--output", str(prepared), timeout=300)
    args.data = str(prepared)
    return {"data": str(prepared), "prepared_here": True}


def stage_train(args: argparse.Namespace, out: Path) -> dict:
    run = out / "run"
    if args.mode == "tiny":
        cli("train", "--recipe", "tiny", "--data", args.data, "--output", str(run), "--tokens", "744", timeout=600)
    else:
        extra = ["--gpus", str(args.gpus)] if args.gpus > 1 else []
        if args.microbatch_size:
            extra += ["--microbatch-size", str(args.microbatch_size)]
        cli("train", "--recipe", args.recipe, "--data", args.data, "--output", str(run), "--max-steps", str(args.steps), *extra, timeout=3600)
    checkpoints = sorted(run.glob("checkpoint-*.pt"))
    if not checkpoints:
        raise StageFailed(f"no checkpoint written in {run}")
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) < 2:
        raise StageFailed("metrics.jsonl has fewer than two rows")
    half = rows[len(rows) // 2:]
    seconds_per_update = (half[-1]["elapsed_seconds"] - half[0]["elapsed_seconds"]) / max(1, half[-1]["step"] - half[0]["step"])
    targets_per_update = (half[-1]["targets"] - half[0]["targets"]) / max(1, half[-1]["step"] - half[0]["step"])
    args.checkpoint = str(checkpoints[-1])
    return {
        "checkpoint": args.checkpoint,
        "updates": rows[-1]["step"],
        "first_loss": rows[0]["train_loss"],
        "last_loss": rows[-1]["train_loss"],
        "seconds_per_update": round(seconds_per_update, 4),
        "targets_per_second": round(targets_per_update / seconds_per_update) if seconds_per_update > 0 else None,
    }


def device_flag(args: argparse.Namespace) -> list[str]:
    return ["--device", "cuda"] if args.mode == "gpu" else []


def stage_evaluate(args: argparse.Namespace) -> dict:
    result = last_json(cli("evaluate", "--checkpoint", args.checkpoint, "--data", args.data, "--max-batches", "4" if args.mode == "tiny" else "20", *device_flag(args), timeout=900))
    loss = float(result["loss"])
    if not loss == loss or loss <= 0:  # NaN or nonsense
        raise StageFailed(f"bad validation loss {loss}")
    return result


def stage_sample(args: argparse.Namespace) -> dict:
    prompt = TINY_PROMPT if args.mode == "tiny" else GPU_PROMPT
    text = cli("sample", "--checkpoint", args.checkpoint, "--prompt", prompt, "--max-new-tokens", "16", "--seed", "1", *device_flag(args), timeout=600).strip()
    if not text:
        raise StageFailed("sample printed nothing")
    return {"prompt": prompt, "text": text[-500:]}


def stage_export(args: argparse.Namespace, out: Path) -> dict:
    export = out / "export"
    cli("export", "--checkpoint", args.checkpoint, "--output", str(export), timeout=600)
    files = sorted(p.name for p in export.iterdir())
    for required in ("config.json", "model.safetensors"):
        if required not in files:
            raise StageFailed(f"export is missing {required}: {files}")
    args.export = str(export)
    return {"export": str(export), "files": files, "model_bytes": (export / "model.safetensors").stat().st_size}


def stage_provenance(args: argparse.Namespace) -> dict:
    return last_json(cli("provenance", "verify", args.export, "--json", timeout=300))


def stage_transformers(args: argparse.Namespace) -> dict:
    """Load the export the way any Hugging Face consumer would, and check it is
    the same model trashpanda trained (trashpanda.export.compare_export)."""
    import torch
    from transformers import AutoModelForCausalLM
    from trashpanda.export import _dtype_kwargs, compare_export

    device = "cuda" if args.mode == "gpu" else "cpu"
    prompt = TINY_PROMPT if args.mode == "tiny" else GPU_PROMPT
    report = compare_export(args.checkpoint, args.export, prompt=prompt, device=device, tolerance=args.tolerance)
    if not report["ok"]:
        raise StageFailed("; ".join(report["problems"]))
    if device == "cuda":
        # What a serving stack does: half precision. Informational, not a gate.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.export, local_files_only=True)
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        full = AutoModelForCausalLM.from_pretrained(args.export, local_files_only=True, **_dtype_kwargs(torch.float32)).to(device).eval()
        half = AutoModelForCausalLM.from_pretrained(args.export, local_files_only=True, **_dtype_kwargs(torch.float16)).to(device).eval()
        with torch.no_grad():
            a = full(input_ids=ids).logits.float()
            c = half(input_ids=ids).logits.float()
        report["fp16_max_abs_logit_diff"] = float((a - c).abs().max())
        report["fp16_top1_agreement"] = float((a.argmax(-1) == c.argmax(-1)).float().mean())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("tiny", "gpu"), default="tiny")
    parser.add_argument("--data", help="prepared dataset directory (gpu mode)")
    parser.add_argument("--recipe", default="40m-titanx", help="gpu mode recipe (default 40m-titanx)")
    parser.add_argument("--steps", type=int, default=20, help="gpu mode training updates (default 20)")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--tolerance", type=float, default=1e-4, help="max |logit| difference, export vs trained model")
    parser.add_argument("--out", default="e2e-out")
    args = parser.parse_args()
    if args.mode == "gpu" and not args.data:
        parser.error("gpu mode needs --data (a dataset made by prepare_fineweb_edu.py or `trashpanda prepare`)")
    if args.mode == "tiny":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = [
        ("environment", lambda: stage_environment(args)),
        ("prepare", lambda: stage_prepare(args, out)),
        ("train", lambda: stage_train(args, out)),
        ("evaluate", lambda: stage_evaluate(args)),
        ("sample", lambda: stage_sample(args)),
        ("export", lambda: stage_export(args, out)),
        ("provenance", lambda: stage_provenance(args)),
        ("transformers", lambda: stage_transformers(args)),
    ]
    report = {"mode": args.mode, "out": str(out), "stages": {}, "passed": False}
    failed = False
    for name, run in stages:
        start = time.time()
        print(f"[{name}]", flush=True)
        try:
            detail = run()
            report["stages"][name] = {"ok": True, "seconds": round(time.time() - start, 1), "detail": detail}
            print(f"  ok ({time.time() - start:.1f}s)", flush=True)
        except Exception as exc:  # report every failure the same way
            message = str(exc) if isinstance(exc, StageFailed) else traceback.format_exc()
            report["stages"][name] = {"ok": False, "seconds": round(time.time() - start, 1), "error": message}
            print(f"  FAILED: {message}", flush=True)
            failed = True
            break
    report["passed"] = not failed
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"\n{'PASS' if report['passed'] else 'FAIL'}: report at {out / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
