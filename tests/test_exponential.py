"""Test Exponential distribution against scipy implementation."""

import pytest
from scipy import stats

from distributions import exponential as Exponential
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1 / 3.7], {"scale": 3.7}),
        ([1e-6], {"scale": 1e6}),
        ([1e6], {"scale": 1e-6}),
    ],
)
def test_exponential_vs_scipy(params, sp_params):
    """Test Exponential distribution against scipy.stats.expon."""
    p_params = make_params(*params)
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=Exponential,
        sp_dist=stats.expon,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="exponential",
    )
