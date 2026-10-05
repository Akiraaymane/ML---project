"""Chargement de la configuration YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Economics:
    """Paramètres économiques d'une campagne de rétention."""

    contact_cost: float = 10.0
    incentive: float = 80.0
    accept_rate: float = 0.30
    non_churner_accept_rate: float = 1.0
    clv_months: int = 12


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def economics_from_config(cfg: dict[str, Any]) -> Economics:
    return Economics(**cfg.get("economics", {}))
