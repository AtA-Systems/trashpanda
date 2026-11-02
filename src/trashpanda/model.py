"""Small, explicit Hugging Face architecture adapters used by training."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from .config import ModelConfig

try:  # Keep config/prepare usable in a minimal source checkout.
    import torch
    import torch.nn.functional as F
    from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    GPT2Config = None  # type: ignore
    GPT2LMHeadModel = None  # type: ignore
    LlamaConfig = None  # type: ignore
    LlamaForCausalLM = None  # type: ignore


def _require_model_dependencies() -> None:
    if torch is None or LlamaConfig is None or LlamaForCausalLM is None or GPT2Config is None or GPT2LMHeadModel is None:
        raise RuntimeError("model construction requires torch and transformers")


def _make_llama_config(config: ModelConfig) -> Any:
    kwargs = dict(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.kv_heads,
        max_position_embeddings=config.context_length,
        rms_norm_eps=config.rms_norm_eps,
        tie_word_embeddings=config.tie_embeddings,
        attention_bias=False,
        mlp_bias=False,
        rope_theta=10_000.0,
        rope_scaling=None,
        hidden_act="silu",
        attention_dropout=0.0,
        initializer_range=0.02,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=None,
    )
    try:
        hf_config = LlamaConfig(attn_implementation="sdpa", **kwargs)
    except TypeError:  # transformers releases before the public constructor kwarg
        hf_config = LlamaConfig(**kwargs)
    hf_config._attn_implementation = "sdpa"
    hf_config.attn_implementation = "sdpa"
    hf_config.use_cache = False
    hf_config.context_length = config.context_length
    return hf_config


def _make_gpt2_config(config: ModelConfig) -> Any:
    hf_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.context_length,
        n_ctx=config.context_length,
        n_embd=config.hidden_size,
        n_layer=config.num_layers,
        n_head=config.num_attention_heads,
        n_inner=config.intermediate_size,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=config.rms_norm_eps,
        initializer_range=0.02,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=None,
        tie_word_embeddings=config.tie_embeddings,
        use_cache=False,
    )
    hf_config.context_length = config.context_length
    return hf_config


def make_hf_config(config: ModelConfig) -> Any:
    """Build the native Transformers config for the selected architecture."""
    _require_model_dependencies()
    if config.architecture == "llama":
        return _make_llama_config(config)
    if config.architecture == "gpt2":
        return _make_gpt2_config(config)
    raise ValueError(f"unsupported model architecture {config.architecture!r}")


def _model_class(architecture: str) -> Any:
    if architecture == "llama":
        return LlamaForCausalLM
    if architecture == "gpt2":
        return GPT2LMHeadModel
    raise ValueError(f"unsupported model architecture {architecture!r}")


def build_model(config: ModelConfig, *, seed: int | None = None, device: str | None = None) -> Any:
    """Construct an actual upstream Transformers causal-LM instance."""
    _require_model_dependencies()
    if seed is not None:
        torch.manual_seed(seed)
    construction = torch.device(device) if device else None
    model_class = _model_class(config.architecture)
    with torch.device(construction) if construction is not None else nullcontext():
        model = model_class(make_hf_config(config))
    if config.tie_embeddings:
        model.tie_weights()

    def _check_context(_module: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        input_ids = kwargs.get("input_ids")
        if input_ids is None and inputs:
            input_ids = inputs[0]
        if input_ids is not None and getattr(input_ids, "ndim", 0) >= 2:
            length = int(input_ids.shape[-1])
            if length > config.context_length:
                raise ValueError(
                    f"input length {length} exceeds configured context length {config.context_length}"
                )

    model.register_forward_pre_hook(_check_context, with_kwargs=True)
    actual = count_parameters(model)
    expected = config.parameter_count()
    if actual != expected:
        raise AssertionError(
            f"{config.architecture} parameter count mismatch: expected {expected}, got {actual}"
        )
    return model


def count_parameters(model: Any) -> int:
    _require_model_dependencies()
    return sum(parameter.numel() for _, parameter in model.named_parameters())


def causal_loss(model: Any, input_ids: Any, labels: Any) -> Any:
    """Compute loss for already-shifted sequence-reader targets."""
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


# Compatibility aliases for callers that used the old public names.
TransformerLM = LlamaForCausalLM
TinyLanguageModel = LlamaForCausalLM

__all__ = [
    "GPT2Config", "GPT2LMHeadModel", "LlamaConfig", "LlamaForCausalLM",
    "TinyLanguageModel", "TransformerLM", "build_model", "causal_loss",
    "count_parameters", "make_hf_config",
]
