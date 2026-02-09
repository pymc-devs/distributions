"""Test ExGaussian distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import exgaussian as ExGaussian
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 1.0, 1.0], {"K": 1.0, "loc": 0.0, "scale": 1.0}),
        ([2.0, 0.5, 100.0], {"K": 200.0, "loc": 2.0, "scale": 0.5}),
        ([-1.0, 10.0, 0.5], {"K": 0.05, "loc": -1.0, "scale": 10.0}),
        ([1.0, 3.0, 3.0], {"K": 1.0, "loc": 1.0, "scale": 3.0}),
    ],
)
def test_exgaussian_vs_scipy(params, sp_params):
    """Test ExGaussian distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=ExGaussian,
        sp_dist=stats.exponnorm,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="exgaussian",
        entropy_rtol=1e-1,
    )
