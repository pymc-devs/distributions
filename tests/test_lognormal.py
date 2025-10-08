"""
Test LogNormal distribution against scipy implementation.
"""
import pytest
import numpy as np
from scipy import stats
from distributions import lognormal as LogNormal
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.0, 0.5], {"s": 0.5, "loc": 0, "scale": np.exp(2.0)}),
        ([0.0, 1.0], {"s": 1.0, "loc": 0, "scale": 1.0}),
        ([-1.0, 0.25], {"s": 0.25, "loc": 0, "scale": np.exp(-1.0)}),
        ([0.0, 0.001], {"s": 0.001, "loc": 0, "scale": 1.0}),
        ([5.0, 2.0], {"s": 2.0, "loc": 0, "scale": np.exp(5.0)}), 
    ],
)
def test_lognormal_vs_scipy(params, sp_params):
    """Test LogNormal distribution against scipy.stats.lognorm."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float('inf'))
    
    run_distribution_tests(
        p_dist=LogNormal,
        sp_dist=stats.lognorm,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="lognormal"
    )