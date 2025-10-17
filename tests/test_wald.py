"""
Test Wald (Inverse Gaussian) distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import wald as Wald
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 10.0], {"mu": 2.0 / 10.0, "scale": 10.0}),
        ([1.0, 1.0], {"mu": 1.0, "scale": 1.0}),
        ([3.0, 5.0], {"mu": 3.0 / 5.0, "scale": 5.0}),
        ([1.0, 0.1], {"mu": 10.0, "scale": 0.1}), 
        ([10.0, 100.0], {"mu": 0.1, "scale": 100.0}),
    ],
)
def test_wald_vs_scipy(params, sp_params):
    """Test Wald distribution against scipy.stats.invgauss."""
    p_params = make_params(*params)
    support = (0, float('inf'))
    
    run_distribution_tests(
        p_dist=Wald,
        sp_dist=stats.invgauss,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="wald"
    )