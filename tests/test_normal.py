import numpy as np
import pytest
from distributions import normal as Normal
from tests.helper_consistency import run_all_consistency_tests

PARAM_SETS = [
    ((0.0, 1.0), None),
    ((-5.0, 0.5), None),
    ((-1e6, 100.0), None),
    ((10.0, 1e-6), None),
    ((1.0, 1e6), None),
]

SUPPORT = (-np.inf, np.inf)
IS_DISCRETE = False


@pytest.mark.parametrize("params, skip_tests", PARAM_SETS)
def test_normal_consistency(params, skip_tests):
    """Run all consistency tests for each parameter combination."""
    run_all_consistency_tests(Normal, params, SUPPORT, IS_DISCRETE, skip_tests=skip_tests)
