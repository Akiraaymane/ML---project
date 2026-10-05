"""Chargement des données et génération d'un jeu synthétique réaliste (télécom)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "churn"
ID_COLUMN = "customer_id"


def _to_snake_case(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def generate_synthetic(n_samples: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Génère des clients télécom avec un churn issu d'un processus non linéaire.

    Les colonnes reprennent celles du dataset public « Telco Customer Churn »
    (en snake_case) pour que le pipeline fonctionne indifféremment sur les deux.
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20]
    )
    max_tenure = np.select(
        [contract == "Month-to-month", contract == "One year"], [36, 60], default=72
    )
    tenure = np.floor(rng.beta(1.1, 1.8, size=n) * max_tenure).astype(int)
    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.35, 0.45, 0.20])
    has_internet = internet != "No"

    def yes_no(p: float, mask: np.ndarray | None = None) -> np.ndarray:
        out = np.where(rng.random(n) < p, "Yes", "No").astype(object)
        if mask is not None:
            out[~mask] = "No internet service"
        return out

    tech_support = yes_no(0.35, has_internet)
    online_security = yes_no(0.35, has_internet)
    streaming_tv = yes_no(0.45, has_internet)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)",
         "Credit card (automatic)"],
        size=n, p=[0.35, 0.20, 0.22, 0.23],
    )
    paperless = yes_no(0.6)
    senior = (rng.random(n) < 0.16).astype(int)
    partner = yes_no(0.48)
    dependents = np.where(partner == "Yes", yes_no(0.5), yes_no(0.15))

    base = np.select([internet == "Fiber optic", internet == "DSL"], [70.0, 45.0], 20.0)
    monthly = (
        base
        + 10 * (streaming_tv == "Yes")
        + 5 * (tech_support == "Yes")
        + 5 * (online_security == "Yes")
        + rng.normal(0, 6, n)
    ).clip(18, 120).round(2)
    total = (monthly * tenure * rng.normal(1.0, 0.03, n)).clip(0).round(2)
    support_tickets = rng.poisson(0.5 + 1.2 * (internet == "Fiber optic") * (tech_support != "Yes"))

    logit = (
        -1.6
        + 1.4 * (contract == "Month-to-month")
        - 1.2 * (contract == "Two year")
        - 1.8 * (1 - np.exp(-tenure / 12))  # effet de fidélité non linéaire
        + 0.9 * (internet == "Fiber optic")
        + 0.6 * (payment == "Electronic check")
        - 0.6 * (tech_support == "Yes")
        - 0.5 * (online_security == "Yes")
        + 0.3 * (paperless == "Yes")
        + 0.35 * senior
        - 0.3 * (partner == "Yes")
        + 0.02 * (monthly - 65)
        + 0.45 * np.minimum(support_tickets, 4)
        # interaction : client récent sans engagement et facture élevée
        + 0.8 * ((contract == "Month-to-month") & (tenure < 6) & (monthly > 80))
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        ID_COLUMN: [f"C{i:06d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], size=n),
        "senior_citizen": senior,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "contract": contract,
        "paperless_billing": paperless,
        "payment_method": payment,
        "internet_service": internet,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "num_support_tickets": support_tickets,
        "monthly_charges": monthly,
        "total_charges": total,
        TARGET: churn,
    })
    # Valeurs manquantes réalistes
    missing = rng.random(n) < 0.02
    df.loc[missing, "total_charges"] = np.nan
    return df


def load_csv(path: str | Path, target: str = TARGET) -> pd.DataFrame:
    """Charge un CSV (ex. Telco Customer Churn) et normalise colonnes et cible."""
    df = pd.read_csv(path)
    df.columns = [_to_snake_case(c) for c in df.columns]
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = df[target].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    # Colonnes numériques stockées en texte (ex. TotalCharges avec des blancs)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and col != ID_COLUMN:
            text = df[col].astype("string").str.strip().replace("", np.nan)
            converted = pd.to_numeric(text, errors="coerce")
            if text.notna().any() and converted.notna().sum() == text.notna().sum():
                df[col] = converted
    return df.dropna(subset=[target]).astype({target: int})


def load_dataset(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg.get("data", {})
    if data_cfg.get("path"):
        return load_csv(data_cfg["path"], data_cfg.get("target", TARGET))
    return generate_synthetic(data_cfg.get("n_samples", 20_000), cfg.get("seed", 42))


def split_xy(df: pd.DataFrame, target: str = TARGET, id_column: str = ID_COLUMN):
    X = df.drop(columns=[c for c in (target, id_column) if c in df.columns])
    return X, df[target].to_numpy()


def train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, target: str = TARGET):
    return train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
