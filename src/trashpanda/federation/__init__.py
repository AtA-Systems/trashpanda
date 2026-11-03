"""Offline, local-process federated training primitives."""

from .config import FederationConfig, ParticipantConfig, default_federation_config
from .bundles import RoundBundle, UpdateBundle, load_round_bundle, load_update_bundle, write_round_bundle, write_update_bundle
from .aggregate import aggregate_updates
from .coordinator import FederationCoordinator

__all__ = [
    "FederationConfig",
    "ParticipantConfig",
    "default_federation_config",
    "RoundBundle",
    "UpdateBundle",
    "load_round_bundle",
    "load_update_bundle",
    "write_round_bundle",
    "write_update_bundle",
    "aggregate_updates",
    "FederationCoordinator",
]
