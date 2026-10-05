"""Métriques et graphiques d'évaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

from .profit import profit_curve  # noqa: E402


def classification_metrics(y_true, proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "churn_rate": float(np.mean(y_true)),
    }


def lift_at(y_true, proba, share: float = 0.1) -> float:
    k = max(1, int(len(proba) * share))
    top = np.argsort(-np.asarray(proba))[:k]
    return float(np.asarray(y_true)[top].mean() / np.mean(y_true))


def save_plots(y_true, proba, value, econ, threshold: float, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    curve = profit_curve(y_true, proba, value, econ)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve["share_targeted"] * 100, curve["profit"], color="#2a6f97", lw=2)
    best = curve.loc[curve["profit"].idxmax()]
    ax.axvline(best["share_targeted"] * 100, ls="--", color="#c44536",
               label=f"optimum : {best['share_targeted']:.0%} ciblés, {best['profit']:,.0f} €")
    chosen = (np.asarray(proba) >= threshold).mean() * 100
    ax.axvline(chosen, ls=":", color="#555", label=f"seuil appris (OOF) = {threshold:.2f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="% de clients ciblés (par risque décroissant)", ylabel="Profit (€)",
           title="Courbe de profit de la campagne (jeu de test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "profit_curve.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, color="#2a6f97", lw=2,
                 label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="grey")
    axes[0].set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbe ROC")
    axes[0].legend()
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, "o-", color="#2a6f97", label="modèle")
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey", label="calibration parfaite")
    axes[1].set(xlabel="Probabilité prédite", ylabel="Fréquence observée", title="Calibration")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "roc_calibration.png", dpi=120)
    plt.close(fig)


def save_importance_plot(importance: pd.Series, out_dir: str | Path, top: int = 15) -> None:
    imp = importance.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.5))
    ax.barh(imp.index, imp.values, color="#2a6f97")
    ax.set(xlabel="Baisse de ROC AUC (permutation)", title="Importance des variables")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "feature_importance.png", dpi=120)
    plt.close(fig)
