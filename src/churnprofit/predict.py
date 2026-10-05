"""Scoring d'un fichier clients et génération de la liste de ciblage.

Usage : python -m churnprofit.predict --input clients.csv --output ciblage.csv
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from .data import ID_COLUMN, TARGET, _to_snake_case
from .profit import customer_value, expected_gain, expected_value_policy


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    econ = bundle["economics"]
    X = df.reindex(columns=bundle["features"])
    proba = bundle["model"].predict_proba(X)[:, 1]
    value = customer_value(X, econ)
    if bundle["policy"] == "expected_value":
        target = expected_value_policy(proba, value, econ)
    else:
        target = proba >= bundle["threshold"]
    out = pd.DataFrame({
        "churn_probability": proba.round(4),
        "customer_value": value.round(2),
        "expected_gain": expected_gain(proba, value, econ).round(2),
        "target": target,
    }, index=df.index)
    if ID_COLUMN in df.columns:
        out.insert(0, ID_COLUMN, df[ID_COLUMN])
    return out.sort_values("expected_gain", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="targeting.csv")
    parser.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    df = pd.read_csv(args.input)
    df.columns = [_to_snake_case(c) for c in df.columns]
    result = score(df.drop(columns=[TARGET], errors="ignore"), bundle)
    result.to_csv(args.output, index=False)
    targeted = result[result["target"]]
    print(f"{len(targeted)}/{len(result)} clients à cibler — "
          f"gain espéré total : {targeted['expected_gain'].sum():,.0f} €")


if __name__ == "__main__":
    main()
