import pytest
from scipy import stats

from distributions import loglogistic as LogLogistic
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1.0, 5.0], {"c": 5.0, "scale": 1.0}),
        ([2.0, 3.0], {"c": 3.0, "scale": 2.0}),
        ([0.5, 10.0], {"c": 10.0, "scale": 0.5}),
        ([3.0, 2.5], {"c": 2.5, "scale": 3.0}),
        ([1.0, 1.5], {"c": 1.5, "scale": 1.0}),
    ],
)
def test_loglogistic_vs_scipy(params, sp_params):
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=LogLogistic,
        sp_dist=stats.fisk,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="loglogistic",
        skip_skewness=sp_params["c"] <= 3,
        skip_kurtosis=sp_params["c"] <= 4,
    )


@pytest.mark.parametrize(
    "params, sp_params",
    [
        ([1.0, 1.5], {"c": 1.5, "scale": 1.0}),
        ([2.0, 1.2], {"c": 1.2, "scale": 2.0}),
    ],
)
def test_loglogistic_low_beta(params, sp_params):
    p_params = make_params(*params, dtype="float64")
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=LogLogistic,
        sp_dist=stats.fisk,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="loglogistic_low_beta",
        skip_variance=True,
        skip_standard_deviation=True,
        skip_skewness=True,
        skip_kurtosis=True,
    )
