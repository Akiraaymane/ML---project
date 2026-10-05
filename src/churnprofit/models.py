"""Modèles candidats."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def _estimator(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(C=0.5, max_iter=2000), True
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features="sqrt",
            n_jobs=-1, random_state=seed,
        ), False
    if name == "hist_gbm":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            random_state=seed,
        ), False
    raise ValueError(f"Modèle inconnu : {name!r}")


def build_model(name: str, seed: int = 42, calibrate: bool = True):
    """Pipeline complet préprocessing + modèle, éventuellement calibré (isotonic).

    La calibration est essentielle : la politique de ciblage raisonne en
    espérance de gain, elle a donc besoin de vraies probabilités.
    """
    estimator, scale = _estimator(name, seed)
    pipeline = Pipeline([("prep", build_preprocessor(scale=scale)), ("model", estimator)])
    if calibrate:
        return CalibratedClassifierCV(pipeline, method="isotonic", cv=3)
    return pipeline
