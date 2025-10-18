"""
Test Chi-squared distribution against scipy implementation.
"""

import pytest
from scipy import stats
from distributions import chisquared as ChiSquared
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1.0], {"df": 1.0}),
        ([5.0], {"df": 5.0}),
        ([0.5], {"df": 0.5}),
        ([100.0], {"df": 100.0}),
    ],
)
def test_chisquared_vs_scipy(params, sp_params):
    """Test Chi-squared distribution against scipy"""
    p_params = make_params(*params, dtype="float64")
    support = (0.0, float("inf"))

    run_distribution_tests(
        p_dist=ChiSquared,
        sp_dist=stats.chi2,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="chisquared",
    )
