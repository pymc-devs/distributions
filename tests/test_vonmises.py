"""Test Von Mises distribution against scipy implementation."""

import numpy as np
import pytest
from scipy import stats

from distributions import vonmises as VonMises
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([0.5, 1], {"loc": 0.5, "kappa": 1}),
        ([1.5, 2.0], {"loc": 1.5, "kappa": 2.0}),
        ([1.0, 60.0], {"loc": 1.0, "kappa": 60.0}),
    ],
)
def test_vonmises_vs_scipy(params, sp_params):
    """Test Von Mises distribution against scipy."""
    p_params = make_params(*params, dtype="float64")
    support = (-np.pi, np.pi)

    run_distribution_tests(
        p_dist=VonMises,
        sp_dist=stats.vonmises,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="vonmises",
        # The current implementation of the ppf is very slow
        skip_mode=True,
        skip_isf=True,
        # The following moments are implemented incorrectly in SciPy.
        skip_standard_deviation=True,
        skip_variance=True,
        skip_skewness=True,
        skip_kurtosis=True,
    )
