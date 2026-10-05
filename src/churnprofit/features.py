"""Feature engineering et préprocessing (scikit-learn, sans fuite de données)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnFeatures(BaseEstimator, TransformerMixin):
    """Ajoute des variables métier ; ignore celles dont les colonnes sources manquent."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = set(X.columns)
        if {"tenure", "total_charges"} <= cols:
            X["avg_charge_per_month"] = X["total_charges"] / (X["tenure"] + 1)
        if {"tenure", "monthly_charges", "total_charges"} <= cols:
            # > 1 : le client paie plus aujourd'hui que sa moyenne historique
            X["charge_trend"] = X["monthly_charges"] / (X["avg_charge_per_month"] + 1)
        if "tenure" in cols:
            X["is_new_customer"] = (X["tenure"] < 6).astype(int)
            X["log_tenure"] = np.log1p(X["tenure"])
        service_cols = [c for c in ("online_security", "tech_support", "streaming_tv") if c in cols]
        if service_cols:
            X["num_services"] = (X[service_cols] == "Yes").sum(axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.transform(pd.DataFrame(columns=self.feature_names_in_)).columns)


def build_preprocessor(scale: bool = True) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler() if scale else "passthrough"),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include=np.number)),
        ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
    ], verbose_feature_names_out=False)
    return Pipeline([("features", ChurnFeatures()), ("columns", columns)])
