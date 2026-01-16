import pytensor.tensor as pt
import pytest
from scipy import stats

from distributions import hypergeometric as Hypergeometric
from tests.helper_scipy import run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        # params = [N, k, n] where N=pop size, k=successes in pop, n=draws
        # scipy uses M=pop size, n=successes in pop, N=draws
        ([20, 7, 12], {"M": 20, "n": 7, "N": 12}),  # lower=0, upper=7
        ([15, 10, 8], {"M": 15, "n": 10, "N": 8}),  # lower=3, upper=8 (non-zero lower)
        ([50, 20, 10], {"M": 50, "n": 20, "N": 10}),  # larger population
    ],
)
def test_hypergeometric_vs_scipy(params, sp_params):
    """Test Hypergeometric distribution against scipy.stats.hypergeom."""
    N_param = pt.constant(params[0], dtype="int64")
    k_param = pt.constant(params[1], dtype="int64")
    n_param = pt.constant(params[2], dtype="int64")
    p_params = (N_param, k_param, n_param)

    N, k, n = params
    lower = max(0, n + k - N)
    upper = min(k, n)
    support = (lower, upper)

    run_distribution_tests(
        p_dist=Hypergeometric,
        sp_dist=stats.hypergeom,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="hypergeometric",
    )
