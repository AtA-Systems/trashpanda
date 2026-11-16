"""Multi-device gradient accumulation.

CPU tests emulate several devices with repeated ``cpu`` entries, which exercises
the replica fan-out, gradient gathering, accounting, and resume logic without a
GPU.  The final test needs two visible CUDA devices and skips otherwise.
"""
from dataclasses import replace
import importlib.util

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
    pytest.skip("PyTorch/transformers are not installed in this environment", allow_module_level=True)

import torch

from trashpanda.checkpoint import _register_safe_checkpoint_globals
from trashpanda.cli import _main
from trashpanda.config import get_recipe, model_recipe, resolve_training_config
from trashpanda.data import SequenceReader
from trashpanda.parallel import ReplicaSet, resolve_devices, validate_devices
from trashpanda.train import train

from .helpers import make_prepared_fixture

pytestmark = pytest.mark.acceptance


def _reader(tmp_path):
    return SequenceReader(make_prepared_fixture(tmp_path / "fixture"), split="train")


def _weights(result):
    return {name: value.detach().cpu().clone() for name, value in result.model.state_dict().items()}


def _assert_same_run(left, right, *, atol=1e-6):
    assert left.state.to_dict() == right.state.to_dict()
    assert left.sampler.state_dict() == right.sampler.state_dict()
    left_weights, right_weights = _weights(left), _weights(right)
    assert left_weights.keys() == right_weights.keys()
    for name, value in left_weights.items():
        assert torch.allclose(value, right_weights[name], rtol=0, atol=atol), name


def _last_segment(checkpoint):
    _register_safe_checkpoint_globals()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    return payload["extra"]["provenance"]["segments"][-1]


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_two_devices_match_single_device(tmp_path, architecture):
    config = model_recipe("tiny", architecture)
    dataset = _reader(tmp_path)
    single = train(config, dataset, checkpoint_dir=tmp_path / "single")
    double = train(config, dataset, checkpoint_dir=tmp_path / "double", devices=["cpu", "cpu"])
    assert single.state.update_step == 6 and single.state.consumed_tokens == 744
    assert double.state.micro_steps == 12
    _assert_same_run(single, double)


def test_uneven_microbatch_split_matches_single_device(tmp_path):
    # Four microbatches over three devices: device 0 takes two, the others one.
    config = resolve_training_config(get_recipe("tiny"), microbatch_size=1)
    dataset = _reader(tmp_path)
    single = train(config, dataset, checkpoint_dir=tmp_path / "single")
    triple = train(config, dataset, checkpoint_dir=tmp_path / "triple", devices=["cpu", "cpu", "cpu"])
    _assert_same_run(single, triple)


def test_checkpoints_resume_across_device_counts(tmp_path):
    config = get_recipe("tiny")
    dataset = _reader(tmp_path)
    reference = train(config, dataset, checkpoint_dir=tmp_path / "reference")

    multi_first = train(config, dataset, checkpoint_dir=tmp_path / "a", max_steps=3, devices=["cpu", "cpu"])
    finished_single = train(config, dataset, resume=multi_first.checkpoint, checkpoint_dir=tmp_path / "a")
    _assert_same_run(reference, finished_single)

    single_first = train(config, dataset, checkpoint_dir=tmp_path / "b", max_steps=3)
    finished_multi = train(config, dataset, resume=single_first.checkpoint, checkpoint_dir=tmp_path / "b", devices=["cpu", "cpu"])
    _assert_same_run(reference, finished_multi)


def test_reported_losses_match_single_device(tmp_path):
    import json

    config = get_recipe("tiny")
    dataset = _reader(tmp_path)
    train(config, dataset, checkpoint_dir=tmp_path / "single")
    train(config, dataset, checkpoint_dir=tmp_path / "double", devices=["cpu", "cpu"])
    rows = [
        [json.loads(line) for line in (tmp_path / name / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        for name in ("single", "double")
    ]
    assert [row["targets"] for row in rows[0]] == [row["targets"] for row in rows[1]]
    for left, right in zip(*rows):
        assert left["train_loss"] == pytest.approx(right["train_loss"], abs=1e-6)


def test_devices_recorded_only_for_multi_device_segments(tmp_path):
    config = get_recipe("tiny")
    dataset = _reader(tmp_path)
    single = train(config, dataset, checkpoint_dir=tmp_path / "single", max_steps=1)
    double = train(config, dataset, checkpoint_dir=tmp_path / "double", max_steps=1, devices=["cpu", "cpu"])
    assert "devices" not in _last_segment(single.checkpoint)
    assert _last_segment(double.checkpoint)["devices"] == ["cpu", "cpu"]
    # One explicit device is the ordinary single-device path.
    one = train(config, dataset, checkpoint_dir=tmp_path / "one", max_steps=1, devices=["cpu"])
    assert "devices" not in _last_segment(one.checkpoint)
    _assert_same_run(single, one, atol=0)


def test_replica_gradients_are_summed_onto_primary():
    from trashpanda.model import build_model, causal_loss

    config = get_recipe("tiny")
    model = build_model(config.model, seed=3)
    generator = torch.Generator().manual_seed(0)
    microbatches = [
        (torch.randint(0, 32, (2, 31), generator=generator), torch.randint(0, 32, (2, 31), generator=generator))
        for _ in range(3)
    ]
    for inputs, targets in microbatches:
        (causal_loss(model, inputs, targets) / 3).backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    expected_losses = [float(causal_loss(model, inputs, targets).detach()) for inputs, targets in microbatches]
    model.zero_grad(set_to_none=True)

    with ReplicaSet(model, ["cpu", "cpu"], autocast_enabled=False, autocast_dtype=torch.bfloat16) as replicas:
        assert replicas.replicas[0] is model and replicas.replicas[1] is not model
        losses = replicas.accumulate(microbatches, 3)
        assert losses == pytest.approx(expected_losses, abs=1e-6)
        for parameter, gradient in zip(model.parameters(), expected):
            assert torch.allclose(parameter.grad, gradient, rtol=0, atol=1e-6)
        # Replicas pick up the primary's current parameters at each update.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.5)
        model.zero_grad(set_to_none=True)
        replicas.accumulate(microbatches, 3)
        for primary, copy in zip(model.parameters(), replicas.replicas[1].parameters()):
            assert torch.equal(primary, copy)


def test_worker_errors_propagate(tmp_path):
    config = get_recipe("tiny")
    dataset = _reader(tmp_path)
    from trashpanda.model import build_model

    model = build_model(config.model, seed=config.training.seed)
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(config, dataset, model=model, checkpoint_dir=tmp_path / "bad", devices=["cpu", "cpu"])


def test_device_requests_are_validated():
    tiny = get_recipe("tiny").training
    assert resolve_devices(tiny, None) is None
    assert resolve_devices(tiny, 1) is None
    with pytest.raises(ValueError, match="CUDA recipe"):
        resolve_devices(tiny, 2)
    with pytest.raises(ValueError, match="positive integer"):
        resolve_devices(get_recipe("40m-titanx").training, 0)
    if torch.cuda.device_count() < 64:
        with pytest.raises(RuntimeError, match="CUDA device"):
            resolve_devices(get_recipe("40m-titanx").training, 64)
    with pytest.raises(ValueError, match="fp16"):
        validate_devices(["cpu", "cpu"], replace(tiny, precision="fp16"))
    with pytest.raises(ValueError, match="microbatches per update"):
        validate_devices(["cpu", "cpu", "cpu"], tiny)
    assert [str(device) for device in validate_devices(["cpu"], tiny)] == ["cpu"]


def test_cli_rejects_gpus_for_cpu_recipe(tmp_path):
    prepared = make_prepared_fixture(tmp_path / "fixture")
    with pytest.raises(ValueError, match="CUDA recipe"):
        _main(["train", "--recipe", "tiny", "--data", str(prepared), "--output", str(tmp_path / "run"), "--gpus", "2"])
    assert _main(["train", "--recipe", "tiny", "--data", str(prepared), "--output", str(tmp_path / "run"), "--gpus", "1", "--max-steps", "1"]) == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two visible CUDA devices")
def test_two_cuda_devices_match_one(tmp_path):
    base = get_recipe("tiny")
    config = replace(base, training=replace(base.training, device="cuda"))
    dataset = _reader(tmp_path)
    single = train(config, dataset, checkpoint_dir=tmp_path / "single")
    double = train(config, dataset, checkpoint_dir=tmp_path / "double", devices=resolve_devices(config.training, 2))
    assert _last_segment(double.checkpoint)["devices"] == ["cuda:0", "cuda:1"]
    _assert_same_run(single, double, atol=1e-4)
    resumed = train(config, dataset, resume=train(config, dataset, checkpoint_dir=tmp_path / "half", max_steps=3, devices=resolve_devices(config.training, 2)).checkpoint, checkpoint_dir=tmp_path / "half")
    _assert_same_run(single, resumed, atol=1e-4)
