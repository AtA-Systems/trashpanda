import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from trashpanda.checkpoint import list_checkpoints
from trashpanda.config import model_recipe
from trashpanda.data import SequenceReader
from trashpanda.export import export_checkpoint
from trashpanda.provenance import inspect_bundle, verify_bundle
from trashpanda.tokenizer import TinyWordLevelTokenizer
from trashpanda.train import train

from .helpers import make_prepared_fixture


pytestmark = pytest.mark.acceptance


def test_real_training_export_contains_reconciled_lineage_and_detects_each_corruption(tmp_path: Path):
    prepared = make_prepared_fixture(tmp_path / "input")
    config = model_recipe("tiny", "llama")
    result = train(config, SequenceReader(prepared), checkpoint_dir=tmp_path / "run", max_steps=3)
    assert result.checkpoint is not None
    exported = export_checkpoint(result.checkpoint, tmp_path / "export")
    valid = verify_bundle(exported)
    assert valid.ok
    manifest = json.loads((exported / "provenance-manifest.json").read_text(encoding="utf-8"))
    records = [json.loads((exported / "provenance" / "records" / f"{item['id']}.json").read_text(encoding="utf-8")) for item in manifest["records"]]
    kinds = {record["type"] for record in records}
    assert {"input", "preparation", "execution_segment", "run", "artifact"} <= kinds
    run = next(record for record in records if record["type"] == "run")
    assert run["actual_target_accounting"]["committed_targets"] == result.state.consumed_tokens
    assert sum(run["actual_target_accounting"]["source_target_counts"].values()) == result.state.consumed_tokens

    cases = {
        "payload": lambda: (exported / "config.json").write_text('{"tampered":true}\n', encoding="utf-8"),
        "record": lambda: (next((exported / "provenance" / "records").glob("*.json"))).write_text("{}\n", encoding="utf-8"),
        "missing": lambda: (exported / "provenance" / "records" / "missing.json").unlink(missing_ok=True),
    }
    for name, mutate in cases.items():
        fresh = tmp_path / name
        import shutil
        shutil.copytree(exported, fresh)
        if name == "missing":
            manifest = json.loads((fresh / "provenance-manifest.json").read_text(encoding="utf-8"))
            reference = manifest["records"][0]
            (fresh / "provenance" / "records" / f"{reference['id']}.json").unlink()
        else:
            mutate_path = (fresh / "config.json") if name == "payload" else next((fresh / "provenance" / "records").glob("*.json"))
            mutate_path.write_text("{}\n", encoding="utf-8")
        assert not verify_bundle(fresh).ok, name


def test_export_relocates_without_training_files(tmp_path: Path):
    prepared = make_prepared_fixture(tmp_path / "input")
    result = train(model_recipe("tiny", "gpt2"), SequenceReader(prepared), checkpoint_dir=tmp_path / "run", max_steps=3)
    exported = export_checkpoint(result.checkpoint, tmp_path / "export")
    relocated = tmp_path / "relocated"
    import shutil
    shutil.copytree(exported, relocated)
    shutil.rmtree(tmp_path / "input")
    shutil.rmtree(tmp_path / "run")
    assert verify_bundle(relocated).ok
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(relocated, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(relocated, local_files_only=True, trust_remote_code=False)
    assert model.config.model_type == "gpt2"
    assert tokenizer.vocab_size == 32
