"""Memory-mapped uint16 sequence reader and resumable PCG64 sampler."""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import struct
from typing import Any, Iterator, Sequence

from .prepare import read_manifest

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore
try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


class SequenceReader:
    """Read one prepared split as exact, unpadded context-length sequences."""

    def __init__(self, path: str | Path, *, split: str = "train", verify_hash: bool = True):
        source = Path(path)
        manifest_path = source / "manifest.json" if source.is_dir() else source
        self.manifest_path = manifest_path
        self.manifest = read_manifest(manifest_path)
        self.context_length = int(self.manifest["context_length"])
        self.vocab_size = int(self.manifest["vocab_size"])
        if self.manifest.get("format") == "trashpanda.tokens.v1":
            split_info = {"shards": [{"path": self.manifest["token_file"], "sequence_count": self.manifest["sequence_count"], "token_count": self.manifest["token_count"]}], "sequence_count": self.manifest["sequence_count"], "token_count": self.manifest["token_count"], "dropped_tokens": 0}
        else:
            if split not in self.manifest.get("splits", {}):
                raise KeyError(f"prepared split {split!r} is unavailable")
            split_info = self.manifest["splits"][split]
        self.split = split
        self.split_info = split_info
        self.sequence_count = int(split_info["sequence_count"])
        self.token_count = int(split_info["token_count"])
        self.dropped_tokens = int(split_info.get("dropped_tokens", 0))
        self.attribution = split_info.get("attribution", {})
        self.source_target_counts = dict(split_info.get("source_target_counts", {}))
        self._verify_tokenizer_files()
        self._shards: list[Any] = []
        self._offsets: list[int] = []
        offset = 0
        for shard in split_info["shards"]:
            path = manifest_path.parent / shard["path"]
            shard_sequences = int(shard["sequence_count"])
            expected_bytes = shard_sequences * self.context_length * 2
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"shard size mismatch: {path}")
            if verify_hash:
                import hashlib
                if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
                    raise ValueError(f"shard hash mismatch: {path}")
            self._offsets.append(offset)
            offset += shard_sequences
            if np is not None:
                self._shards.append(np.memmap(path, mode="r", dtype="<u2", shape=(shard_sequences, self.context_length)))
            else:  # pragma: no cover
                self._shards.append((path, shard_sequences))
        if offset != self.sequence_count:
            raise ValueError("manifest sequence count does not match shards")

    def _verify_tokenizer_files(self) -> None:
        expected = self.manifest.get("tokenizer_file_hashes", {})
        if not expected:
            return
        tokenizer_dir = self.manifest_path.parent / "tokenizer"
        for name, digest in expected.items():
            path = tokenizer_dir / str(name)
            if not path.is_file():
                raise ValueError(f"tokenizer file is missing: {name}")
            import hashlib

            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"tokenizer hash mismatch: {name}")

    def __len__(self) -> int:
        return self.sequence_count

    @property
    def training_tokens_per_sequence(self) -> int:
        return self.context_length - 1

    def _locate(self, index: int) -> tuple[Any, int]:
        index = int(index)
        if not 0 <= index < self.sequence_count:
            raise IndexError(index)
        for shard_index in range(len(self._offsets) - 1, -1, -1):
            if index >= self._offsets[shard_index]:
                return self._shards[shard_index], index - self._offsets[shard_index]
        raise IndexError(index)

    def tokens(self, index: int) -> Any:
        shard, row = self._locate(index)
        if np is not None:
            return np.asarray(shard[row], dtype=np.int64).copy()
        path, _count = shard  # pragma: no cover
        with path.open("rb") as handle:
            handle.seek(row * self.context_length * 2)
            return struct.unpack(f"<{self.context_length}H", handle.read(self.context_length * 2))

    def __iter__(self) -> Iterator[Any]:
        for index in range(self.sequence_count):
            yield self.tokens(index)

    def __getitem__(self, index: int) -> Any:
        """Return one complete, unpadded L-token sequence as int64."""
        row = self.tokens(index)
        if torch is not None:
            return torch.as_tensor(row, dtype=torch.long)
        return row

    def batch(self, indices: Sequence[int]) -> tuple[Any, Any]:
        rows = [self.tokens(index) for index in indices]
        if np is not None:
            values = np.asarray(rows, dtype=np.int64)
            inputs, targets = values[:, :-1], values[:, 1:]
        else:  # pragma: no cover
            inputs = [row[:-1] for row in rows]
            targets = [row[1:] for row in rows]
        if torch is not None:
            return torch.as_tensor(inputs, dtype=torch.long), torch.as_tensor(targets, dtype=torch.long)
        return inputs, targets

    def iter_batches(self, indices: Iterator[Sequence[int]]) -> Iterator[tuple[Any, Any]]:
        for batch_indices in indices:
            yield self.batch(batch_indices)

    def target_source_counts(self, indices: Sequence[int]) -> dict[str, int]:
        """Return source ownership of actual prediction targets in sampled rows."""
        ranges = self.attribution.get("ranges", [])
        per_sequence = int(self.attribution.get("sequence_target_count", self.training_tokens_per_sequence))
        counts: dict[str, int] = defaultdict(int)
        if not ranges:
            return dict(counts)
        for index in indices:
            start = int(index) * per_sequence
            end = start + per_sequence
            for item in ranges:
                range_start = max(start, int(item["start"]))
                range_end = min(end, int(item["end"]))
                if range_start < range_end:
                    counts[str(item["source_id"])] += range_end - range_start
                if int(item["start"]) >= end:
                    break
        return dict(counts)

    # Short alias for callers that use the language from the issue contract.
    attribution_for_indices = target_source_counts


MMapTokenDataset = SequenceReader
MMapReader = SequenceReader


class PCG64Sampler:
    """A deterministic permutation sampler whose repeat policy is explicit."""

    def __init__(self, size: int, seed: int = 0, *, repeat_data: bool = True):
        if np is None:
            raise RuntimeError("numpy is required for PCG64Sampler")
        if size <= 0:
            raise ValueError("sampler size must be positive")
        self.size = int(size)
        self.seed = int(seed)
        self.repeat_data = bool(repeat_data)
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self.epoch = 0
        self.position = 0
        self._permutation: list[int] = []

    def _start_epoch(self) -> None:
        self._permutation = [int(i) for i in self.generator.permutation(self.size)]
        self.position = 0
        self.epoch += 1

    def next_indices(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("count must be positive")
        result: list[int] = []
        while len(result) < count:
            if self.position >= len(self._permutation):
                if not self.repeat_data and self._permutation:
                    raise StopIteration("prepared data exhausted and repeat_data is false")
                self._start_epoch()
            take = min(count - len(result), len(self._permutation) - self.position)
            result.extend(self._permutation[self.position : self.position + take])
            self.position += take
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "PCG64",
            "size": self.size,
            "seed": self.seed,
            "repeat_data": self.repeat_data,
            "epoch": self.epoch,
            "position": self.position,
            "permutation": list(self._permutation),
            "bit_generator_state": deepcopy(self.generator.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") != "PCG64" or int(state.get("size", -1)) != self.size:
            raise ValueError("incompatible PCG64 sampler state")
        self.seed = int(state["seed"])
        self.repeat_data = bool(state.get("repeat_data", self.repeat_data))
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self.generator.bit_generator.state = state["bit_generator_state"]
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self._permutation = [int(i) for i in state.get("permutation", [])]
        if not 0 <= self.position <= len(self._permutation):
            raise ValueError("invalid sampler position")


__all__ = ["MMapReader", "MMapTokenDataset", "PCG64Sampler", "SequenceReader"]
