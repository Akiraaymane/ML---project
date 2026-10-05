import joblib
import numpy as np

from churnprofit.data import generate_synthetic, split_xy
from churnprofit.models import build_model
from churnprofit.predict import score
from churnprofit.train import run


def test_models_learn_signal():
    df = generate_synthetic(3000, seed=0)
    X, y = split_xy(df)
    for name in ("logreg", "hist_gbm"):
        model = build_model(name, calibrate=False).fit(X[:2000], y[:2000])
        proba = model.predict_proba(X[2000:])[:, 1]
        from sklearn.metrics import roc_auc_score
        assert roc_auc_score(y[2000:], proba) > 0.75


def test_end_to_end_training_beats_baselines(tmp_path):
    cfg = {
        "seed": 0,
        "data": {"n_samples": 4000, "test_size": 0.25},
        "training": {"cv_folds": 3, "models": ["logreg", "hist_gbm"], "calibrate": True},
        "output": {"artifacts_dir": str(tmp_path / "art"), "reports_dir": str(tmp_path / "rep")},
    }
    summary = run(cfg)
    test = summary["test"]
    assert test["profit"] > test["baselines"]["everybody"]
    assert test["profit"] > test["baselines"]["nobody"]
    assert (tmp_path / "rep" / "report.md").exists()

    bundle = joblib.load(tmp_path / "art" / "model.joblib")
    new_clients = generate_synthetic(200, seed=99).drop(columns="churn")
    out = score(new_clients, bundle)
    assert len(out) == 200
    assert out["churn_probability"].between(0, 1).all()
    assert np.all(np.diff(out["expected_gain"].to_numpy()) <= 0)  # trié par gain
