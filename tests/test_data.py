import pandas as pd

from churnprofit.data import _to_snake_case, generate_synthetic, load_csv, split_xy


def test_synthetic_is_reproducible_and_realistic():
    a = generate_synthetic(2000, seed=1)
    b = generate_synthetic(2000, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert 0.1 < a["churn"].mean() < 0.45
    assert a["total_charges"].isna().any()  # valeurs manquantes simulées


def test_snake_case_matches_telco_columns():
    assert _to_snake_case("customerID") == "customer_id"
    assert _to_snake_case("MonthlyCharges") == "monthly_charges"
    assert _to_snake_case("SeniorCitizen") == "senior_citizen"


def test_load_csv_normalizes_telco_format(tmp_path):
    path = tmp_path / "telco.csv"
    pd.DataFrame({
        "customerID": ["a", "b", "c"],
        "tenure": [1, 10, 30],
        "TotalCharges": ["29.9", " ", "1500.5"],
        "Churn": ["Yes", "No", "No"],
    }).to_csv(path, index=False)
    df = load_csv(path)
    assert df["churn"].tolist() == [1, 0, 0]
    assert df["total_charges"].dtype.kind == "f"
    X, y = split_xy(df)
    assert "churn" not in X and "customer_id" not in X
