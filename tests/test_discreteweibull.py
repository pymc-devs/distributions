"""Test Discrete Weibull distribution."""

import pytest

from pytensor_distributions import discreteweibull as DWeibull
from tests.helper_empirical import run_empirical_tests
from tests.helper_scipy import make_params


@pytest.mark.parametrize(
    "params",
    [
        [0.8, 2.0],
        [0.5, 1.5],
        [0.3, 1.0],
        [0.9, 3.0],
    ],
)
def test_discrete_weibull_vs_random(params):
    """Test Discrete Weibull distribution against random samples."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_empirical_tests(
        p_dist=DWeibull,
        p_params=p_params,
        support=support,
        name="discrete_weibull",
        sample_size=500_000,
        mean_rtol=1e-2,
        var_rtol=1e-2,
        std_rtol=1e-2,
        quantiles_rtol=1e-2,
        cdf_rtol=5e-2,
        is_discrete=True,
    )
