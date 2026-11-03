# Testing and CI

Trashpanda's required feature tests run on generated data and tiny CPU recipes. They do not need accounts, downloaded model weights, a GPU, or a runtime network connection. Installation may use the package index; runtime tests set the Hugging Face offline flags and use only project-owned tokenizers and fixtures.

Run the suite locally from a checkout with Python 3.11 or newer:

```bash
python -m pip install '.[test,provenance]'
python -m pytest -m 'unit or acceptance' --require-feature-deps
```

The `unit` group covers pure configuration, data, provenance, model construction and aggregation. The `acceptance` group runs both Llama and GPT-2 lifecycle/export checks and the local tiny-Llama federation workflow. `extended` is reserved for scheduled extra seeds and dependency compatibility runs. Every subprocess helper has a timeout and captures stdout/stderr; failed runs retain those logs in the pytest failure.

The pull-request workflow uses one pinned Linux CPU environment and runs the unit and acceptance groups. The nightly workflow adds extra seeds and a small Python/dependency compatibility matrix. Release validation builds a wheel and source distribution, installs them in a clean environment outside the checkout, and runs the same bounded scenarios. Release validation retains distributions, hashes and test reports but does not publish them.

Numerical equality claims are scoped to the same CPU environment: split training and uninterrupted training must have identical tensors, counters, sampler state and RNG state. Export logits use `atol=1e-6, rtol=1e-5`. Tests assert accounting, recovery, identity and valid artifacts; they do not gate on generated prose or on federation outperforming a baseline.
