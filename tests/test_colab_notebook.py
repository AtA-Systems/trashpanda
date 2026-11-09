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
