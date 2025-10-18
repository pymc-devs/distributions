"""
Test Beta distribution against scipy implementation.
"""

import pytest
import numpy as np
from scipy import stats
from distributions import beta as Beta
from .helper_scipy import run_distribution_tests, make_params


@pytest.mark.parametrize(
    "params, sp_params, skip_mode",
    [
        ([2.0, 5.0], {"a": 2, "b": 5}, False),
        ([0.5, 3.0], {"a": 0.5, "b": 3.0}, True),
        ([100.0, 100.0], {"a": 100.0, "b": 100.0}, False),
        ([1.0, 1.0], {"a": 1, "b": 1}, True),
    ],
)
def test_beta_vs_scipy(params, sp_params, skip_mode):
    """Test Beta distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (0, 1)

    run_distribution_tests(
        p_dist=Beta,
        sp_dist=stats.beta,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="beta",
        skip_mode=skip_mode,
    )
