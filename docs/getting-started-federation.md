# Local federated training quickstart

After installing the project and its test dependencies, run the complete
synthetic fixture without a GPU, account, download or runtime network access:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FED_DEMO="$(mktemp -d)"
python -m trashpanda federate demo --output "$FED_DEMO"
python -m trashpanda federate evaluate --run "$FED_DEMO/run"
python -m trashpanda federate export --run "$FED_DEMO/run" --output "$FED_DEMO/export"
python -m trashpanda provenance verify "$FED_DEMO/export" --json
```

Add `--compare-baselines` to the demo command to train local-only and pooled
synthetic comparison models from the same initial tensors. The report is written
to `run/baseline-comparison.json`; it is a measurement fixture, not a quality
claim.

The demo creates two distinct synthetic prepared datasets under `fixture/`,
launches separate `alice` and `bob` worker processes for three rounds, and
stores the coordinator ledger under `run/`. A successful default run reports
`current_round: 3` and `target_count: 1488` (496 targets per round).

To resume a run after a worker or coordinator interruption:

```bash
python -m trashpanda federate run --resume "$FED_DEMO/run"
```

The run's configuration is frozen at `run/config.json`. Completed contributions
are identified by digest and are not trained or counted again. The final export
is relocatable: verification and standard Hugging Face loading need only the
export directory, not the private fixture or coordinator run.

The public provenance states that participant source lineage is withheld and
that the coordinator observed individual updates. Local file separation does
not provide confidentiality from a malicious same-user process, and the
prototype must not be treated as secure federated learning.
