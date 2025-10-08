"""
Test NegativeBinomial distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import negativebinomial as NegativeBinomial
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params, entropy_rtol",
    [
        ([2.1, 0.375], {"n": 2.1, "p": 0.375}, 1e-3),
        ([1., 0.8], {"n": 1., "p": 0.8}, 1e-2),
        ([10., 0.1], {"n": 10., "p": 0.1}, 1e-3),
        ([0.5, 0.9], {"n": 0.5, "p": 0.9}, 1e-1),
        ([20., 0.05], {"n": 20., "p": 0.05}, 1e-1),
    ],
)
def test_negativebinomial_vs_scipy(params, sp_params, entropy_rtol):
    """Test NegativeBinomial distribution against scipy.stats.nbinom."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float('inf'))
    
    run_distribution_tests(
        p_dist=NegativeBinomial,
        sp_dist=stats.nbinom,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="negativebinomial",
        entropy_rtol=entropy_rtol,
        is_discrete=True,
    )