"""
Test Poisson distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import poisson as Poisson
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0], {"mu": 2.0}),
        ([0.01], {"mu": 0.01}),
        ([100.0], {"mu": 100.0}),
    ],
)
def test_poisson_vs_scipy(params, sp_params):
    """Test Poisson distribution against scipy.stats.poisson."""
    p_params = make_params(*params)
    support = (0, float('inf'))
    
    run_distribution_tests(
        p_dist=Poisson,
        sp_dist=stats.poisson,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="poisson",
        is_discrete=True,
    )