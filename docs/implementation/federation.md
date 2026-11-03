# Local federation implementation

Trashpanda's federation prototype runs entirely on one machine. A coordinator
process launches one worker process per participant. Each worker reads only its
own prepared directory, trains the current tiny Llama checkpoint for a bounded
number of optimizer steps, and writes a validated model update to a local
exchange directory. The coordinator performs deterministic target-weighted
full-model averaging and commits an immutable next-round bundle.

The default fixture has participants `alice` and `bob`, three rounds, two local
steps per round, batch size four, context length 32, and CPU FP32. Each update
contains 248 prediction targets; a complete run therefore commits 1,488 targets.
The coordinator sees individual updates. This is a process and file-layout
boundary for a reproducible experiment, not an OS security sandbox or a privacy
mechanism.

Round bundles contain only protocol metadata and model tensors. Participant
optimizer, sampler, RNG, source records and prepared shards stay in the private
participant state directory. Bundle publication uses a temporary sibling and
rename, and the ledger records accepted update digests so identical retries are
idempotent while conflicting retries fail.

The supported commands are:

```bash
python -m trashpanda federate demo --output /tmp/trashpanda-fed-demo
python -m trashpanda federate evaluate --run /tmp/trashpanda-fed-demo/run
python -m trashpanda federate export --run /tmp/trashpanda-fed-demo/run --output /tmp/trashpanda-fed-export
python -m trashpanda provenance verify /tmp/trashpanda-fed-export --json
```

`federate run --config CONFIG --output RUN` runs an explicitly prepared local
configuration. `federate run --resume RUN` reads the frozen configuration from
the run and retries only missing work. A failed round remains pending; it is not
partially averaged.

The prototype does not provide secure aggregation, differential privacy,
cryptographic participant authentication, poisoning resistance, unlearning,
network transport or a convergence guarantee. Do not use proprietary data until
those questions have a reviewed threat model and an implementation with tested
guarantees.
