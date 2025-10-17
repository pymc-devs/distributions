"""
Test Cauchy distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import cauchy as Cauchy
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 1.0], {"loc": 0.0, "scale": 1.0}),
        ([-5.0, 0.5], {"loc": -5.0, "scale": 0.5}),
        ([-1e6, 100.0], {"loc": -1e6, "scale": 100.0}),
        ([10.0, 1e-6], {"loc": 10.0, "scale": 1e-6}),
        ([1.0, 1e-6], {"loc": 1.0, "scale": 1e-6}),
    ],
)
def test_cauchy_vs_scipy(params, sp_params):
    """Test Cauchy distribution against scipy.stats.cauchy."""
    p_params = make_params(*params)
    support = (-float('inf'), float('inf'))
    
    # Cauchy distribution has undefined mean and variance, so skip those tests
    run_distribution_tests(
        p_dist=Cauchy,
        sp_dist=stats.cauchy,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="cauchy",
        skip_skewness=True,  # Undefined for Cauchy
        skip_kurtosis=True   # Undefined for Cauchy
    )