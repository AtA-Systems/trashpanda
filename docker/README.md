# trashpanda container images

The full guide (file layout, image configuration, how `docker run` works, testing,
publishing to GHCR and Hugging Face, serving with vLLM) is
[docs/containers.md](../docs/containers.md). The short version:

| Flavor | PyTorch | GPUs | Host driver |
|---|---|---|---|
| `pascal` | 2.14, CUDA 12.6 | Pascal, Volta, up to Hopper | 525+ |
| `modern` | 2.14, CUDA 13.0 | Turing through Blackwell | 580+ |

```bash
# build (from the repository root)
docker build -f docker/Dockerfile --build-arg FLAVOR=pascal -t trashpanda:dev-pascal .
docker build -f docker/Dockerfile --build-arg FLAVOR=modern -t trashpanda:dev-modern .

# test on this host
docker/test-image.sh trashpanda:dev-pascal --data /data/datasets/fw-40m-titanx
docker/test-image.sh trashpanda:dev-modern --data /data/datasets/fw-40m-titanx --vllm

# train
docker run --rm --gpus all --ipc=host --shm-size=8g --user "$(id -u):$(id -g)" -e HOME=/work \
  -v /data/datasets/fw-40m-titanx:/data/prepared:ro -v /data/runs:/runs -v /data/models/huggingface:/hf \
  trashpanda:dev-pascal python -m trashpanda train --recipe 40m-titanx --data /data/prepared --output /runs/40m-titanx --gpus 2
```

Files here:

- `Dockerfile`: both flavors (`--build-arg FLAVOR=pascal|modern`)
- `test-image.sh`: tests an image on a host, stage by stage
- `tests/e2e.py`: train → evaluate → sample → export → provenance → transformers
- `tests/vllm_smoke.sh`: serves an export with vLLM and calls its OpenAI API
