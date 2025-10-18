"""
Test Gamma distribution against scipy implementation.
"""

import pytest
from scipy import stats
from distributions import gamma as Gamma
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 1 / 3], {"a": 2.0, "scale": 3.0}),
        ([1.0, 1.0], {"a": 1.0, "scale": 1.0}),
        ([0.5, 2.0], {"a": 0.5, "scale": 0.5}),
        ([50.0, 0.5], {"a": 50.0, "scale": 2.0}),
        ([0.01, 0.01], {"a": 0.01, "scale": 100.0}),
        ([100.0, 100.0], {"a": 100.0, "scale": 0.01}),
    ],
)
def test_gamma_vs_scipy(params, sp_params):
    """Test Gamma distribution against scipy.stats.gamma."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=Gamma,
        sp_dist=stats.gamma,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="gamma",
    )
