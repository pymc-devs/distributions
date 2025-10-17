"""
Test InverseGamma distribution against scipy implementation.
"""
import pytest
from scipy import stats
from distributions import inversegamma as InverseGamma
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([5.0, 2.0], {"a": 5.0, "scale": 2.0}),
        ([0.1, 0.1], {"a": 0.1, "scale": 0.1}),
        ([100., 50.0], {"a": 100., "scale": 50.0}),
    ],
)
def test_inversegamma_vs_scipy(params, sp_params):
    """Test InverseGamma distribution against scipy.stats.invgamma."""
    p_params = make_params(*params, dtype="float64")
    support = (0, float('inf'))
    
    run_distribution_tests(
        p_dist=InverseGamma,
        sp_dist=stats.invgamma,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="inversegamma",
        use_quantiles_for_rvs=True,
    )