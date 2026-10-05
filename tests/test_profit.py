import numpy as np
import pandas as pd

from churnprofit.config import Economics
from churnprofit.profit import (
    baseline_profits,
    campaign_profit,
    customer_value,
    expected_gain,
    expected_value_policy,
    optimize_threshold,
)

ECON = Economics(contact_cost=10, incentive=80, accept_rate=0.3,
                 non_churner_accept_rate=1.0, clv_months=12)


def test_customer_value_uses_monthly_charges():
    X = pd.DataFrame({"monthly_charges": [50.0, np.nan, 100.0]})
    np.testing.assert_allclose(customer_value(X, ECON), [600, 900, 1200])


def test_campaign_profit_by_hand():
    y = np.array([1, 0])
    value = np.array([1000.0, 1000.0])
    # churner : 0.3 * (1000 - 80) - 10 = 266 ; fidèle : -80 - 10 = -90
    assert campaign_profit(y, [True, False], value, ECON) == 266
    assert campaign_profit(y, [True, True], value, ECON) == 176


def test_expected_gain_matches_realized_gain_at_certainty():
    value = np.array([1000.0])
    assert expected_gain(np.array([1.0]), value, ECON)[0] == 266
    assert expected_gain(np.array([0.0]), value, ECON)[0] == -90


def test_expected_value_policy_is_monotonic_in_probability():
    value = np.full(101, 800.0)
    mask = expected_value_policy(np.linspace(0, 1, 101), value, ECON)
    assert not mask[0] and mask[-1]
    assert np.all(np.diff(mask.astype(int)) >= 0)


def test_optimal_threshold_with_perfect_model_reaches_oracle():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    value = rng.uniform(300, 1500, 500)
    _, profit = optimize_threshold(y, y.astype(float), value, ECON)
    assert np.isclose(profit, baseline_profits(y, value, ECON)["oracle"])
