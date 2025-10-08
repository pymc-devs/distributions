import numpy as np
import pytest
from distributions import poisson as Poisson
from tests.helper_consistency import run_all_consistency_tests

PARAM_SETS = [
    ((3.5,), None),
    # ((0.0001,), None),
    # ((100,), None),
]

SUPPORT = (0, np.inf)
IS_DISCRETE = True


@pytest.mark.parametrize("params, skip_tests", PARAM_SETS)
def test_poisson_consistency(params, skip_tests):
    """Run all consistency tests for each parameter combination."""
    run_all_consistency_tests(Poisson, params, SUPPORT, IS_DISCRETE, skip_tests=skip_tests)
