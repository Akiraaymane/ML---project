"""Entraînement : sélection de modèle par le profit, calibration, évaluation, export.

Usage : python -m churnprofit.train --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import __version__
from .config import economics_from_config, load_config
from .data import ID_COLUMN, TARGET, load_dataset, split_xy, train_test
from .evaluate import classification_metrics, lift_at, save_importance_plot, save_plots
from .models import build_model
from .profit import (
    baseline_profits,
    campaign_profit,
    customer_value,
    expected_value_policy,
    optimize_threshold,
)

log = logging.getLogger("churnprofit")


def evaluate_candidates(X, y, cfg, econ) -> pd.DataFrame:
    """Prédictions out-of-fold pour chaque modèle ; le critère principal est le profit."""
    seed = cfg.get("seed", 42)
    tcfg = cfg.get("training", {})
    cv = StratifiedKFold(tcfg.get("cv_folds", 5), shuffle=True, random_state=seed)
    value = customer_value(X, econ)
    rows = []
    for name in tcfg.get("models", ["logreg", "random_forest", "hist_gbm"]):
        model = build_model(name, seed, tcfg.get("calibrate", True))
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold, thr_profit = optimize_threshold(y, proba, value, econ)
        ev_profit = campaign_profit(y, expected_value_policy(proba, value, econ), value, econ)
        rows.append({
            "model": name, **classification_metrics(y, proba),
            "threshold": threshold, "profit_threshold": thr_profit, "profit_ev": ev_profit,
        })
        log.info("%-14s AUC=%.4f  profit(seuil)=%10.0f €  profit(EV)=%10.0f €",
                 name, rows[-1]["roc_auc"], thr_profit, ev_profit)
    return pd.DataFrame(rows).sort_values("profit_ev", ascending=False).reset_index(drop=True)


def run(cfg: dict) -> dict:
    seed = cfg.get("seed", 42)
    econ = economics_from_config(cfg)
    out_cfg = cfg.get("output", {})
    artifacts = Path(out_cfg.get("artifacts_dir", "artifacts"))
    reports = Path(out_cfg.get("reports_dir", "reports"))
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg)
    target = cfg.get("data", {}).get("target", TARGET)
    id_col = cfg.get("data", {}).get("id_column", ID_COLUMN)
    log.info("Données : %d clients, taux de churn %.1f%%", len(df), 100 * df[target].mean())
    train_df, test_df = train_test(df, cfg.get("data", {}).get("test_size", 0.2), seed, target)
    X_train, y_train = split_xy(train_df, target, id_col)
    X_test, y_test = split_xy(test_df, target, id_col)

    leaderboard = evaluate_candidates(X_train, y_train, cfg, econ)
    best = leaderboard.iloc[0]
    # Politique retenue : la meilleure des deux en validation croisée
    policy = "expected_value" if best["profit_ev"] >= best["profit_threshold"] else "threshold"
    log.info("Modèle retenu : %s (politique : %s)", best["model"], policy)

    model = build_model(best["model"], seed, cfg.get("training", {}).get("calibrate", True))
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    value = customer_value(X_test, econ)
    if policy == "expected_value":
        target_mask = expected_value_policy(proba, value, econ)
    else:
        target_mask = proba >= best["threshold"]

    baselines = baseline_profits(y_test, value, econ, seed, share=float(target_mask.mean()) or 0.3)
    test_profit = campaign_profit(y_test, target_mask, value, econ)
    metrics = {
        **classification_metrics(y_test, proba),
        "lift_at_10pct": lift_at(y_test, proba, 0.10),
        "share_targeted": float(target_mask.mean()),
        "n_targeted": int(target_mask.sum()),
        "profit": test_profit,
        "profit_per_customer": test_profit / len(y_test),
        "share_of_oracle": test_profit / baselines["oracle"],
        "baselines": baselines,
    }

    imp = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=seed, n_jobs=1)
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)

    save_plots(y_test, proba, value, econ, float(best["threshold"]), reports)
    save_importance_plot(importance, reports)

    bundle = {
        "model": model, "model_name": best["model"], "policy": policy,
        "threshold": float(best["threshold"]), "economics": econ,
        "features": list(X_train.columns), "version": __version__,
    }
    joblib.dump(bundle, artifacts / "model.joblib")

    summary = {
        "model": best["model"], "policy": policy, "threshold": float(best["threshold"]),
        "economics": econ.__dict__, "test": metrics,
        "cv_leaderboard": leaderboard.to_dict(orient="records"),
        "feature_importance": importance.round(5).to_dict(),
    }
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (reports / "report.md").write_text(render_report(summary, leaderboard), encoding="utf-8")
    log.info("Profit test : %.0f € (%.0f%% de l'oracle) en ciblant %.1f%% des clients",
             test_profit, 100 * metrics["share_of_oracle"], 100 * metrics["share_targeted"])
    return summary


def render_report(summary: dict, leaderboard: pd.DataFrame) -> str:
    t = summary["test"]
    b = t["baselines"]
    lb = leaderboard[["model", "roc_auc", "pr_auc", "brier", "profit_threshold", "profit_ev"]]
    rows = "\n".join(
        f"| {r.model} | {r.roc_auc:.4f} | {r.pr_auc:.4f} | {r.brier:.4f} "
        f"| {r.profit_threshold:,.0f} € | {r.profit_ev:,.0f} € |"
        for r in lb.itertuples()
    )
    random_key = next(k for k in b if k.startswith("random"))
    top_features = list(summary["feature_importance"])[:5]
    return f"""# Rapport ChurnProfit

## Modèle retenu : `{summary['model']}` — politique `{summary['policy']}`

| Indicateur (jeu de test) | Valeur |
|---|---|
| ROC AUC | {t['roc_auc']:.4f} |
| PR AUC | {t['pr_auc']:.4f} |
| Brier score | {t['brier']:.4f} |
| Lift @10 % | {t['lift_at_10pct']:.2f}x |
| Clients ciblés | {t['n_targeted']} ({t['share_targeted']:.1%}) |
| **Profit de la campagne** | **{t['profit']:,.0f} €** |
| Profit / client du portefeuille | {t['profit_per_customer']:.2f} € |
| Part du profit oracle capturée | {t['share_of_oracle']:.1%} |

## Comparaison aux stratégies de référence

| Stratégie | Profit |
|---|---|
| Ne cibler personne | {b['nobody']:,.0f} € |
| Cibler tout le monde | {b['everybody']:,.0f} € |
| Cibler au hasard (même volume) | {b[random_key]:,.0f} € |
| **ChurnProfit** | **{t['profit']:,.0f} €** |
| Oracle (borne supérieure) | {b['oracle']:,.0f} € |

## Validation croisée (jeu d'entraînement, prédictions out-of-fold)

| Modèle | ROC AUC | PR AUC | Brier | Profit (seuil optimisé) | Profit (espérance) |
|---|---|---|---|---|---|
{rows}

Variables les plus influentes : {', '.join(f'`{f}`' for f in top_features)}.

![Courbe de profit](profit_curve.png)
![ROC et calibration](roc_calibration.png)
![Importance des variables](feature_importance.png)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(0)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
