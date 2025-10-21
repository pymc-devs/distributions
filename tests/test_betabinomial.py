"""Test BetaBinomial distribution against scipy implementation."""

import pytensor.tensor as pt
import pytest
from scipy import stats

from distributions import betabinomial as BetaBinomial
from tests.helper_scipy import run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([10, 2.0, 3.0], {"n": 10, "a": 2.0, "b": 3.0}),
        ([5, 1.0, 1.0], {"n": 5, "a": 1.0, "b": 1.0}),
        ([20, 0.5, 0.5], {"n": 20, "a": 0.5, "b": 0.5}),
        ([15, 5.0, 2.0], {"n": 15, "a": 5.0, "b": 2.0}),
        ([100, 20.0, 20.0], {"n": 100, "a": 20.0, "b": 20.0}),
    ],
)
def test_betabinomial_vs_scipy(params, sp_params):
    """Test BetaBinomial distribution against scipy."""
    n_param = pt.constant(params[0], dtype="int64")
    alpha_param = pt.constant(params[1], dtype="float64")
    beta_param = pt.constant(params[2], dtype="float64")
    p_params = (n_param, alpha_param, beta_param)
    support = (0, params[0])

    run_distribution_tests(
        p_dist=BetaBinomial,
        sp_dist=stats.betabinom,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        is_discrete=True,
        name="betabinomial",
    )
