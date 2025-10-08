"""
Test HalfCauchy distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import halfcauchy as HalfCauchy
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([3.5], {"scale": 3.5}),
        ([1e6], {"scale": 1e6}),
        ([1e-6], {"scale":  1e-6}),
    ],
)
def test_halfcauchy_vs_scipy(params, sp_params):
    """Test HalfCauchy distribution against scipy.stats.halfcauchy."""
    p_params = make_params(*params)
    support = (0, float('inf'))
    
    run_distribution_tests(
        p_dist=HalfCauchy,
        sp_dist=stats.halfcauchy,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="halfcauchy",
    )