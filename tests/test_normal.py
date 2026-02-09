"""Test Normal distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import normal as Normal
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 1.0], {"loc": 0.0, "scale": 1.0}),
        ([-5.0, 0.5], {"loc": -5.0, "scale": 0.5}),
        ([-1e6, 100.0], {"loc": -1e6, "scale": 100.0}),
        ([10.0, 1e-6], {"loc": 10.0, "scale": 1e-6}),
        ([1.0, 1e-6], {"loc": 1.0, "scale": 1e-6}),
    ],
)
def test_normal_vs_scipy(params, sp_params):
    """Test Normal distribution against scipy."""
    p_params = make_params(*params)
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=Normal,
        sp_dist=stats.norm,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="normal",
    )
