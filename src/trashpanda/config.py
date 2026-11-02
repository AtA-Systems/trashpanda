"""Validated recipes and the deliberately small flat training override surface."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

MISTRAL_TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
MISTRAL_TOKENIZER_REVISION = "27d67f1b5f57dc0953326b2601d68371d40ea8da"
# Raw bytes at the pinned Hugging Face revision.  These are provenance checks,
# not downloaded assets.
MISTRAL_TOKENIZER_FILE_HASHES = {
    "tokenizer_config.json": "ddb008229511e51607002ffe28925001c4a9ca4177dc4de3a655d085cc610b99",
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    "tokenizer.json": "11c08db21487c885d8c792180f0be237f6a261b89a46f128a6a80a3aa4bd1720",
}
# Compatibility aliases used by callers that only track the SentencePiece file.
MISTRAL_TOKENIZER_HASHES = MISTRAL_TOKENIZER_FILE_HASHES
MISTRAL_TOKENIZER_SHA256 = MISTRAL_TOKENIZER_FILE_HASHES["tokenizer.json"]
MISTRAL_TOKENIZER_FILES_SHA256 = MISTRAL_TOKENIZER_FILE_HASHES
MODEL_ARCHITECTURES = ("llama", "gpt2")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    num_layers: int
    hidden_size: int
    intermediate_size: int
    context_length: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    architecture: str = "llama"

    def __post_init__(self) -> None:
        for name in ("vocab_size", "num_layers", "hidden_size", "intermediate_size", "context_length", "num_attention_heads"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("model.hidden_size must divide by num_attention_heads")
        if self.num_key_value_heads is not None:
            if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads:
                raise ValueError("invalid model.num_key_value_heads")
        if self.rms_norm_eps <= 0 or not isinstance(self.tie_embeddings, bool):
            raise ValueError("invalid model normalization or tying setting")
        if self.architecture not in MODEL_ARCHITECTURES:
            raise ValueError(f"unsupported model architecture {self.architecture!r}; choose {', '.join(MODEL_ARCHITECTURES)}")

    @property
    def kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads

    def parameter_count(self) -> int:
        """Count unique trainable parameters for the selected HF architecture."""
        h = self.hidden_size
        if self.architecture == "gpt2":
            # GPT-2 uses learned position embeddings and bias terms throughout.
            # The LM head is tied to the token embedding when requested.
            block = 4 * h * h + 2 * h * self.intermediate_size + 9 * h + self.intermediate_size
            output = self.vocab_size * h + self.context_length * h + 2 * h
            if not self.tie_embeddings:
                output += self.vocab_size * h
            return output + self.num_layers * block
        head_dim = h // self.num_attention_heads
        attention = h * h + 2 * (self.kv_heads * head_dim) * h + h * h
        mlp = 3 * h * self.intermediate_size
        output = self.vocab_size * h + self.num_layers * (attention + mlp + 2 * h) + h
        if not self.tie_embeddings:
            output += self.vocab_size * h
        return output


@dataclass(frozen=True)
class DataConfig:
    train_path: str | None = None
    validation_path: str | None = None
    tokenizer_name: str = MISTRAL_TOKENIZER_NAME
    tokenizer_revision: str = MISTRAL_TOKENIZER_REVISION
    tokenizer_sha256: str | None = None
    seed: int = 0
    pad_id: int = 0
    eos_id: int = 2

    @property
    def eval_path(self) -> str | None:
        return self.validation_path


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    microbatch_size: int
    token_budget: int
    checkpoint_interval: int
    precision: str
    device: str
    learning_rate: float = 3e-4
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    seed: int = 0
    log_interval: int = 1
    repeat_data: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_size", "microbatch_size", "token_budget", "checkpoint_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"training.{name} must be a positive integer")
        if self.batch_size % self.microbatch_size:
            raise ValueError("training.batch_size must divide evenly by microbatch_size")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("training.precision must be fp32, bf16, or fp16")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("training.device must be cpu, cuda, or auto")
        if self.learning_rate <= 0 or self.warmup_steps < 0 or not 0 <= self.min_lr_ratio <= 1 or self.weight_decay < 0 or not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("invalid optimizer or scheduler hyperparameters")
        if self.grad_clip <= 0 or self.log_interval <= 0 or self.seed < 0:
            raise ValueError("invalid training controls")
        if not isinstance(self.repeat_data, bool):
            raise ValueError("training.repeat_data must be boolean")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size

    @property
    def grad_accumulation_steps(self) -> int:
        return self.batch_size // self.microbatch_size

    @property
    def target_tokens(self) -> int:
        return self.token_budget

    @property
    def dtype(self) -> str:
        return self.precision


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str = "runs/tiny"
    checkpoint_dir: str | None = None
    resume: str | None = None
    num_workers: int = 0
    max_steps: int | None = None
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.num_workers < 0 or (self.max_steps is not None and self.max_steps <= 0):
            raise ValueError("invalid runtime limits")


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    recipe_name: str = "custom"

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count()

    # Forward the flat training surface for callers of resolve_training_config.
    @property
    def token_budget(self) -> int:
        return self.training.token_budget

    @property
    def microbatch_size(self) -> int:
        return self.training.microbatch_size

    @property
    def learning_rate(self) -> float:
        return self.training.learning_rate

    @property
    def repeat_data(self) -> bool:
        return self.training.repeat_data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRAINING_OVERRIDE_KEYS = frozenset({"token_budget", "microbatch_size", "learning_rate", "repeat_data"})


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _strict_flat_overrides(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("training overrides must be a flat JSON object")
    unknown = set(value) - TRAINING_OVERRIDE_KEYS
    if unknown:
        raise ValueError(
            "training overrides only allow token_budget, microbatch_size, learning_rate, repeat_data; "
            f"unknown: {sorted(unknown)}"
        )
    result = dict(value)
    for key in ("token_budget", "microbatch_size"):
        if key in result and (type(result[key]) is not int or result[key] <= 0):
            raise ValueError(f"{key} must be an integer > 0")
    if "learning_rate" in result and (type(result["learning_rate"]) is not float or not math.isfinite(result["learning_rate"]) or result["learning_rate"] <= 0):
        raise ValueError("learning_rate must be a finite float > 0")
    if "repeat_data" in result and type(result["repeat_data"]) is not bool:
        raise ValueError("repeat_data must be boolean")
    return result


def resolve_training_config(
    recipe: str | Config = "tiny",
    overrides: Mapping[str, Any] | str | Path | None = None,
    **values: Any,
) -> Config:
    """Resolve recipe defaults plus strict flat overrides.

    ``values`` is intentionally the same four-key surface as the JSON form;
    this makes CLI precedence explicit and prevents arbitrary config mutation.
    """
    base = get_recipe(recipe) if isinstance(recipe, str) else recipe
    if isinstance(overrides, (str, Path)):
        overrides = _read_json(overrides)
    patch = _strict_flat_overrides(overrides if overrides is not None else {})
    patch.update(_strict_flat_overrides({key: value for key, value in values.items() if value is not None}))
    return replace(base, training=replace(base.training, **patch))


def apply_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """Apply the strict flat training override object."""
    return resolve_training_config(config, overrides)


def load_config(
    recipe_name: str | Path = "tiny",
    *,
    config_json: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    if isinstance(recipe_name, Path) or (isinstance(recipe_name, str) and recipe_name not in {"tiny", "300m-a100"} and Path(recipe_name).is_file()):
        if config_json is not None:
            raise ValueError("config JSON was supplied twice")
        config_json, recipe_name = recipe_name, "tiny"
    base = get_recipe(str(recipe_name))
    file_patch = _read_json(config_json) if config_json is not None else None
    config = resolve_training_config(base, file_patch)
    return resolve_training_config(config, overrides)


def config_from_dict(value: Mapping[str, Any]) -> Config:
    if not isinstance(value, Mapping):
        raise TypeError("config must be an object")
    model = ModelConfig(**dict(value["model"]))
    data_values = dict(value.get("data", {}))
    if "eval_path" in data_values and "validation_path" not in data_values:
        data_values["validation_path"] = data_values.pop("eval_path")
    data = DataConfig(**data_values)
    training = TrainingConfig(**dict(value["training"]))
    runtime = RuntimeConfig(**dict(value.get("runtime", {})))
    return Config(model=model, data=data, training=training, runtime=runtime, recipe_name=str(value.get("recipe_name", "custom")))


def _recipes() -> dict[str, Config]:
    return {
        "tiny": Config(
            recipe_name="tiny",
            model=ModelConfig(vocab_size=32, num_layers=2, hidden_size=32, intermediate_size=64, context_length=32, num_attention_heads=4, architecture="llama"),
            data=DataConfig(tokenizer_name="trashpanda/tiny-wordlevel", tokenizer_revision="project"),
            training=TrainingConfig(batch_size=4, microbatch_size=2, token_budget=744, checkpoint_interval=3, precision="fp32", device="cpu", learning_rate=0.001, warmup_steps=0, min_lr_ratio=0.1, seed=42, repeat_data=True),
            runtime=RuntimeConfig(output_dir="runs/tiny"),
        ),
        "300m-a100": Config(
            recipe_name="300m-a100",
            model=ModelConfig(vocab_size=32_000, num_layers=22, hidden_size=1024, intermediate_size=2560, context_length=2048, num_attention_heads=16),
            data=DataConfig(),
            training=TrainingConfig(batch_size=128, microbatch_size=8, token_budget=6_000_000_000, checkpoint_interval=500, precision="bf16", device="cuda", learning_rate=3e-4, warmup_steps=200, min_lr_ratio=0.1, seed=42, repeat_data=True),
            runtime=RuntimeConfig(output_dir="runs/300m-a100"),
        ),
    }


def model_recipe(name: str, architecture: str = "llama") -> Config:
    """Return a supported recipe with an explicitly selected architecture."""
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(f"unsupported model architecture {architecture!r}; choose {', '.join(MODEL_ARCHITECTURES)}")
    base = get_recipe(name)
    if architecture == "gpt2" and name != "tiny":
        raise ValueError("gpt2 only supports the tiny recipe")
    return replace(base, model=replace(base.model, architecture=architecture))


def get_recipe(name: str) -> Config:
    try:
        return _recipes()[name]
    except KeyError as exc:
        raise ValueError(f"unknown recipe {name!r}; choose tiny or 300m-a100") from exc


recipe = get_recipe
load_recipe = get_recipe

__all__ = [
    "Config", "DataConfig", "ModelConfig", "RuntimeConfig", "TrainingConfig",
    "MODEL_ARCHITECTURES",
    "TRAINING_OVERRIDE_KEYS", "MISTRAL_TOKENIZER_FILE_HASHES", "MISTRAL_TOKENIZER_HASHES",
    "MISTRAL_TOKENIZER_NAME", "MISTRAL_TOKENIZER_REVISION", "MISTRAL_TOKENIZER_SHA256",
    "apply_overrides", "config_from_dict", "get_recipe", "load_config", "load_recipe",
    "recipe", "resolve_training_config", "model_recipe",
]
