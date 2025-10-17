"""
Test StudentT distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import studentt as StudentT
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([5., 0., 2.], {"df": 5, "loc": 0, "scale": 2}),
        ([100., 0., 1.], {"df": 100, "loc": 0, "scale": 1}), 
        ([5., -2., 3.], {"df": 5, "loc": -2, "scale": 3}),
    ],
)
def test_studentt_vs_scipy(params, sp_params):
    """Test StudentT distribution against scipy.stats.t."""
    p_params = make_params(*params, dtype="float64")
    support = (-float('inf'), float('inf'))
    
    run_distribution_tests(
        p_dist=StudentT,
        sp_dist=stats.t,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="studentt",
    )