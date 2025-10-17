"""
Test Exponential distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import exponential as Exponential
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1 / 3.7], {"scale": 3.7}),
        ([1e-6], {"scale": 1E6}),
        ([1E6], {"scale": 1e-6}),
    ],
)
def test_exponential_vs_scipy(params, sp_params):
    """Test Exponential distribution against scipy.stats.expon."""
    p_params = make_params(*params)
    support = (0, float('inf'))
    
    run_distribution_tests(
        p_dist=Exponential,
        sp_dist=stats.expon,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="exponential"
    )