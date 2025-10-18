"""
Test Logistic distribution against scipy implementation.
"""

import pytest
from scipy import stats
from distributions import logistic as Logistic
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.5, 4], {"loc": 2.5, "scale": 4}),
        ([-1, 2], {"loc": -1, "scale": 2}),
        ([0, 0.01], {"loc": 0, "scale": 0.01}),
        ([100.0, 10.0], {"loc": 100.0, "scale": 10.0}),
    ],
)
def test_logistic_vs_scipy(params, sp_params):
    """Test Logistic distribution against scipy.stats.logistic."""
    p_params = make_params(*params)
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=Logistic,
        sp_dist=stats.logistic,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="logistic",
    )
