"""Test Triangular distribution against scipy implementation."""

import pytest
from scipy import stats

from distributions import triangular as Triangular
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.0, 0.5, 1.0], {"loc": 0.0, "scale": 1.0, "c": 0.5}),
        ([-5.0, -4.0, -3.0], {"loc": -5.0, "scale": 2.0, "c": 0.5}),
        ([0.0, 0.1, 1.0], {"loc": 0.0, "scale": 1.0, "c": 0.1}),
        ([-1e6, 0, 1e6], {"loc": -1e6, "scale": 2e6, "c": 0}),
        ([1.0, 1.001, 1.004], {"loc": 1.0, "scale": 0.004, "c": 0.5}),
    ],
)
def test_triangular_vs_scipy(params, sp_params):
    """Test Triangular distribution against scipy."""
    lower, c, upper = params
    scale = upper - lower
    c_rel = (c - lower) / scale
    sp_params = dict(sp_params)
    sp_params["c"] = c_rel
    p_params = make_params(*params, dtype="float64")
    support = (lower, upper)

    run_distribution_tests(
        p_dist=Triangular,
        sp_dist=stats.triang,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="triangular",
    )
