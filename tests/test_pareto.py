"""Test Pareto distribution against scipy implementation."""

import pytest
from scipy import stats

from pytensor_distributions import pareto as Pareto
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([3.0, 1.0], {"b": 3.0, "scale": 1.0}),
        ([2.5, 2.0], {"b": 2.5, "scale": 2.0}),
        ([4.1, 0.5], {"b": 4.1, "scale": 0.5}),
        ([1.5, 10.0], {"b": 1.5, "scale": 10.0}),
        ([0.5, 1.0], {"b": 0.5, "scale": 1.0}),
    ],
)
def test_pareto_vs_scipy(params, sp_params):
    """Test Pareto distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (params[1], float("inf"))

    run_distribution_tests(
        p_dist=Pareto,
        sp_dist=stats.pareto,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="pareto",
    )
