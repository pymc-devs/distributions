import pytensor.tensor as pt
import pytest
from scipy import stats

from distributions import hypergeometric as Hypergeometric
from tests.helper_scipy import run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([20, 7, 12], {"M": 20, "n": 7, "N": 12}),  # lower=0, upper=7
        ([15, 10, 8], {"M": 15, "n": 10, "N": 8}),  # lower=3, upper=8 (non-zero lower)
        ([50, 20, 10], {"M": 50, "n": 20, "N": 10}),  # larger population
    ],
)
def test_hypergeometric_vs_scipy(params, sp_params):
    """Test Hypergeometric distribution against scipy.stats.hypergeom."""
    M_param = pt.constant(params[0], dtype="int64")
    n_param = pt.constant(params[1], dtype="int64")
    N_param = pt.constant(params[2], dtype="int64")
    p_params = (M_param, n_param, N_param)

    M, n, N = params
    lower = max(0, N + n - M)
    upper = min(n, N)
    support = (lower, upper)

    run_distribution_tests(
        p_dist=Hypergeometric,
        sp_dist=stats.hypergeom,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="hypergeometric",
        is_discrete=True,
        use_quantiles_for_rvs=True,
    )
