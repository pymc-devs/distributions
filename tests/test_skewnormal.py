"""
Test SkewNormal distribution against scipy implementation.
"""

import pytest
from scipy import stats
from distributions import skewnormal as SkewNormal
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.5, 1.0, -2.0], {"a": -2, "loc": 0.5, "scale": 1.0}),
        ([0.0, 1.0, 10.0], {"a": 10, "loc": 0.0, "scale": 1.0}),
        ([0.0, 1.0, -10.0], {"a": -10, "loc": 0.0, "scale": 1.0}),
    ],
)
def test_skewnormal_vs_scipy(params, sp_params):
    """Test SkewNormal distribution against scipy.stats.skewnorm."""
    p_params = make_params(*params, dtype="float64")
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=SkewNormal,
        sp_dist=stats.skewnorm,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="skewnormal",
        entropy_rtol=1e-2,
    )
