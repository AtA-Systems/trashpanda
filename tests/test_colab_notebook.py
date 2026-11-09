"""Execute notebook workflow cells with local storage standing in for mounted Drive."""
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "trashpanda_colab.ipynb"
pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("architecture", ["llama", "gpt2"])
def test_colab_restart_and_provenance(tmp_path, architecture):
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    for name, source in cells.items():
        compile(source, f"notebook:{name}", "exec")

    def execute(name, namespace):
        source = cells[name]
        if name == "settings":
            source = source.replace('model_architecture = "llama"', f'model_architecture = "{architecture}"')
            source = source.replace("token_budget = 12400", "token_budget = 744")
            source = source.replace("steps_per_session = 25", "steps_per_session = 3")
            source = source.replace("checkpoint_every = 5", "checkpoint_every = 1")
        exec(compile(source, f"notebook:{name}", "exec"), namespace)

    def runtime():
        namespace = {"Path": Path, "DRIVE_ROOT": tmp_path}
        execute("settings", namespace)
        execute("prepare", namespace)
        execute("stopping", namespace)
        return namespace

    first = runtime()
    execute("train", first)
    assert first["state"]["update_step"] == 3
    assert first["state"]["consumed_tokens"] == 372
    first_checkpoint = first["checkpoint"]
    # Simulate a disconnected runtime and a partial newer checkpoint on Drive.
    corrupt = first_checkpoint.parent / "checkpoint-00000004.pt"
    corrupt.write_bytes(b"incomplete write")
    del first
    second = runtime()
    assert second["RUN"] == tmp_path / "my-first-run"
    execute("train", second)
    assert second["state"]["update_step"] == 6
    assert second["state"]["consumed_tokens"] == 744
    execute("export", second)
    execute("portable", second)
    original_export = second["EXPORT"]
    # Completed runs and exports are safe to rerun without advancing training.
    execute("train", second)
    execute("export", second)
    assert second["EXPORT"] == original_export
    assert first_checkpoint.exists()
    records = second["inspect_bundle"](original_export)
    types = [record["type"] for record in records["records"]]
    assert {"input", "preparation", "run", "artifact"} <= set(types)
    assert types.count("execution_segment") >= 2
    # Settings guard prevents accidentally repurposing an existing run.
    saved = second["settings_path"]
    changed = json.loads(saved.read_text())
    changed["architecture"] = "gpt2" if architecture == "llama" else "llama"
    saved.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="different saved settings"):
        execute("settings", second)
    # Never silently restart from scratch if every persisted checkpoint is broken.
    for path in second["list_checkpoints"](second["CHECKPOINTS"]):
        path.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="No usable checkpoint"):
        second["latest_usable_checkpoint"]()


def test_validation_patience_survives_runtime_restart(tmp_path, monkeypatch):
    from trashpanda.evaluate import EvaluationResult
    import trashpanda.evaluate as evaluation_module

    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    # Constant held-out loss gives an explicit, deterministic plateau.
    monkeypatch.setattr(evaluation_module, "evaluate", lambda *a, **kw: EvaluationResult(2.0, 7.389, 248, 8))

    def runtime():
        namespace = {"Path": Path, "DRIVE_ROOT": tmp_path}
        settings = cells["settings"].replace("validation_patience = 5", "validation_patience = 1")
        settings = settings.replace("steps_per_session = 25", "steps_per_session = 1")
        for source in (settings, cells["prepare"], cells["stopping"], cells["train"]):
            exec(source, namespace)
        return namespace

    first = runtime()
    assert first["state"]["update_step"] == 1
    assert first["status"]["stop_reason"] is None
    second = runtime()
    assert second["state"]["update_step"] == 2
    assert second["status"]["stop_reason"] == "validation_plateau"
    assert len(second["history"]) == 2
    third = runtime()
    assert third["state"]["update_step"] == 2
    assert len(third["history"]) == 2  # No duplicate evaluation or extra training.
    # If the latest checkpoint is lost, its stopping observation must not survive.
    third["checkpoint"].write_bytes(b"incomplete checkpoint")
    fourth = runtime()
    assert fourth["state"]["update_step"] == 2
    assert len(fourth["history"]) == 2
    assert fourth["status"]["stop_reason"] == "validation_plateau"


def test_full_settings_use_production_budget_and_reject_gpt2(tmp_path):
    from trashpanda.config import model_recipe

    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(next(c for c in notebook["cells"] if c["id"] == "settings")["source"])
    source = source.replace("full_run = False", "full_run = True")
    namespace = {"Path": Path, "DRIVE_ROOT": tmp_path}
    exec(source, namespace)
    assert namespace["token_budget"] == 6_000_000_000
    assert namespace["checkpoint_every"] == 500
    assert namespace["recipe_name"] == "300m-a100"
    assert model_recipe(namespace["recipe_name"]).parameter_count == 298_103_808
    assert namespace["settings"]["full_run"] is True
    with pytest.raises(ValueError, match="issues/61"):
        exec(source.replace('model_architecture = "llama"', 'model_architecture = "gpt2"'),
             {"Path": Path, "DRIVE_ROOT": tmp_path})


def test_full_controller_continues_until_budget_on_cpu_fixture(tmp_path, monkeypatch):
    """Exercise full-mode orchestration on tiny weights; this is not GPU validation."""
    from contextlib import nullcontext
    import torch
    import trashpanda.evaluate as evaluation_module

    real_evaluate = evaluation_module.evaluate
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(evaluation_module, "evaluate",
                        lambda *a, **kw: real_evaluate(*a, **{**kw, "device": "cpu"}))
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    namespace = {"Path": Path, "DRIVE_ROOT": tmp_path}
    source = cells["settings"].replace("token_budget = 12400", "token_budget = 744")
    source = source.replace("steps_per_session = 25", "steps_per_session = 2")
    source = source.replace("early_stopping = True", "early_stopping = False")
    for source in (source, cells["prepare"], cells["stopping"]):
        exec(source, namespace)
    namespace["full_run"] = True
    exec(cells["train"], namespace)
    assert namespace["state"]["update_step"] == 6
    assert len(namespace["history"]) == 3
    assert namespace["status"]["stop_reason"] == "token_budget"
    exec(cells["train"], namespace)
    assert namespace["state"]["update_step"] == 6
    assert len(namespace["history"]) == 3


def test_full_preparation_persists_real_data_contract_with_mocked_downloads(tmp_path, monkeypatch):
    """Mock external I/O/GPU and model size, retaining real preparation/file writes."""
    import sys
    from types import SimpleNamespace
    import torch
    import huggingface_hub
    import trashpanda.config as configuration
    import trashpanda.tokenizer as tokenization

    requested = []
    downloads = []
    tiny = configuration.model_recipe("tiny")

    def fake_recipe(name, architecture):
        requested.append((name, architecture))
        return tiny

    def fake_download(dataset, **kwargs):
        downloads.append((dataset, kwargs))
        return iter({"text": " ".join(f"w{(i + d) % 29}" for i in range(128))} for d in range(6))

    monkeypatch.setattr(configuration, "model_recipe", fake_recipe)
    monkeypatch.setattr(tokenization, "load_tokenizer", lambda *a, **kw: tokenization.TinyWordLevelTokenizer())
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_download))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=lambda _: SimpleNamespace(sha="fixed-revision")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock A100")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=80 * 2**30))
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    settings = cells["settings"].replace("full_run = False", "full_run = True")
    settings = settings.replace("full_train_documents = 30000", "full_train_documents = 4")
    settings = settings.replace("full_validation_documents = 1000", "full_validation_documents = 2")
    namespace = {"Path": Path, "DRIVE_ROOT": tmp_path}
    exec(settings, namespace)
    exec(cells["prepare"], namespace)
    assert requested == [("300m-a100", "llama")]
    assert downloads[0][1]["revision"] == "fixed-revision"
    sources = namespace["SOURCES"] / "fineweb"
    assert len((sources / "train.jsonl").read_text().splitlines()) == 4
    assert len((sources / "validation.jsonl").read_text().splitlines()) == 2
    assert (sources / "train.jsonl").read_text() != (sources / "validation.jsonl").read_text()
    assert namespace["config"].training.token_budget == 6_000_000_000
    exec(cells["prepare"], namespace)
    assert len(downloads) == 1


def test_nonfinite_evaluation_is_a_persisted_stop(tmp_path, monkeypatch):
    import trashpanda.evaluate as evaluation_module

    def fail_evaluation(*args, **kwargs):
        raise FloatingPointError("non-finite evaluation loss")

    monkeypatch.setattr(evaluation_module, "evaluate", fail_evaluation)
    notebook = json.loads(NOTEBOOK.read_text())
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
    namespace = {"Path": Path, "DRIVE_ROOT": tmp_path}
    source = cells["settings"].replace("steps_per_session = 25", "steps_per_session = 1")
    for source in (source, cells["prepare"], cells["stopping"], cells["train"]):
        exec(source, namespace)
    assert namespace["status"]["stop_reason"] == "non_finite_validation_loss"
    persisted = json.loads(namespace["STOPPING_FILE"].read_text())
    assert persisted["observations"][0]["loss"] is None
    exec(cells["train"], namespace)
    assert namespace["state"]["update_step"] == 1
