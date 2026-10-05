"""Économie de la rétention : profit réalisé, seuil optimal et politique en espérance.

Pour un client ciblé :
  - churner      : il accepte l'offre avec proba ``accept_rate`` ; on récupère
                   alors sa valeur (CLV) moins la remise ;
  - non-churner  : il serait resté de toute façon, la remise est perdue ;
  - dans tous les cas on paie ``contact_cost``.
Un client non ciblé rapporte 0 (référence : ne rien faire).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Economics

DEFAULT_MONTHLY_VALUE = 65.0


def customer_value(X: pd.DataFrame, econ: Economics) -> np.ndarray:
    """Valeur d'un client retenu : ``clv_months`` mois de facturation."""
    if "monthly_charges" in X.columns:
        monthly = X["monthly_charges"].fillna(X["monthly_charges"].median()).to_numpy(float)
    else:
        monthly = np.full(len(X), DEFAULT_MONTHLY_VALUE)
    return monthly * econ.clv_months


def targeting_gain(y_true: np.ndarray, value: np.ndarray, econ: Economics) -> np.ndarray:
    """Gain (espéré sur l'acceptation) de cibler chaque client, connaissant son label."""
    y = np.asarray(y_true)
    churner_gain = econ.accept_rate * (value - econ.incentive)
    loyal_loss = econ.non_churner_accept_rate * econ.incentive
    return np.where(y == 1, churner_gain, -loyal_loss) - econ.contact_cost


def campaign_profit(y_true, target_mask, value, econ: Economics) -> float:
    return float(targeting_gain(y_true, value, econ)[np.asarray(target_mask, bool)].sum())


def expected_gain(proba: np.ndarray, value: np.ndarray, econ: Economics) -> np.ndarray:
    """Gain espéré de cibler un client selon sa probabilité de churn prédite."""
    p = np.asarray(proba)
    return (
        p * econ.accept_rate * (value - econ.incentive)
        - (1 - p) * econ.non_churner_accept_rate * econ.incentive
        - econ.contact_cost
    )


def expected_value_policy(proba, value, econ: Economics) -> np.ndarray:
    """Politique de décision bayésienne : cibler ssi le gain espéré est positif."""
    return expected_gain(proba, value, econ) > 0


def profit_curve(y_true, proba, value, econ: Economics) -> pd.DataFrame:
    """Profit cumulé en ciblant les k clients les plus risqués, pour tout k."""
    order = np.argsort(-np.asarray(proba), kind="stable")
    gains = targeting_gain(y_true, value, econ)[order]
    n = len(order)
    return pd.DataFrame({
        "n_targeted": np.arange(0, n + 1),
        "share_targeted": np.arange(0, n + 1) / n,
        "threshold": np.r_[1.0, np.asarray(proba)[order]],
        "profit": np.r_[0.0, np.cumsum(gains)],
    })


def optimize_threshold(y_true, proba, value, econ: Economics) -> tuple[float, float]:
    """Seuil de probabilité qui maximise le profit réalisé (à appliquer sur données OOF)."""
    curve = profit_curve(y_true, proba, value, econ)
    best = curve.loc[curve["profit"].idxmax()]
    return float(best["threshold"]), float(best["profit"])


def baseline_profits(y_true, value, econ: Economics, seed: int = 0, share: float = 0.3) -> dict:
    y = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    random_mask = rng.random(len(y)) < share
    return {
        "nobody": 0.0,
        "everybody": campaign_profit(y, np.ones_like(y, bool), value, econ),
        f"random_{int(share * 100)}pct": campaign_profit(y, random_mask, value, econ),
        "oracle": float(np.clip(targeting_gain(y, value, econ), 0, None).sum()),
    }
