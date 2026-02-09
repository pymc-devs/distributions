"""Test Gumbel distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import gumbel as Gumbel
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([2.5, 3.5], {"loc": 2.5, "scale": 3.5}),
        ([0.0, 1.0], {"loc": 0.0, "scale": 1.0}),
        ([-1.0, 2.0], {"loc": -1, "scale": 2}),
        ([-2.0, 0.01], {"loc": -2, "scale": 0.01}),
        ([100.0, 100.0], {"loc": 100, "scale": 100}),
    ],
)
def test_gumbel_vs_scipy(params, sp_params):
    """Test Gumbel distribution against scipy.stats.gumbel_r."""
    p_params = make_params(*params)
    support = (-float("inf"), float("inf"))

    run_distribution_tests(
        p_dist=Gumbel,
        sp_dist=stats.gumbel_r,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="gumbel",
    )
