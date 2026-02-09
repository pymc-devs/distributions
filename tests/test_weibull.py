"""Test Weibull distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import weibull as Weibull
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params, skip_mode",
    [
        ([2.0, 5.0], {"c": 2.0, "scale": 5.0}, False),
        ([0.5, 3.0], {"c": 0.5, "scale": 3.0}, False),
        ([1.0, 1.0], {"c": 1.0, "scale": 1.0}, False),
        ([100.0, 2.0], {"c": 100.0, "scale": 2.0}, False),
    ],
)
def test_weibull_vs_scipy(params, sp_params, skip_mode):
    """Test Weibull distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=Weibull,
        sp_dist=stats.weibull_min,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="weibull",
        skip_mode=skip_mode,
        use_quantiles_for_rvs=True,
    )
