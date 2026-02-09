import numpy as np
import pytensor.tensor as pt
import pytest
from scipy import stats

from pytensor_distributions import truncatednormal as TruncatedNormal
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params, sp_params",
    [
        # mu, sigma, lower, upper -> scipy truncnorm uses a, b, loc, scale
        # a = (lower - mu) / sigma, b = (upper - mu) / sigma
        # One-sided truncation (avoids median near 0)
        ([0.0, 1.0, 0.0, 3.0], {"a": 0.0, "b": 3.0, "loc": 0.0, "scale": 1.0}),
        ([0.0, 1.0, -3.0, 0.0], {"a": -3.0, "b": 0.0, "loc": 0.0, "scale": 1.0}),
        # Non-zero mu with symmetric truncation
        ([5.0, 2.0, 0.0, 10.0], {"a": -2.5, "b": 2.5, "loc": 5.0, "scale": 2.0}),
        ([3.0, 1.0, 1.0, 5.0], {"a": -2.0, "b": 2.0, "loc": 3.0, "scale": 1.0}),
        ([1.0, 0.5, 0.0, 2.0], {"a": -2.0, "b": 2.0, "loc": 1.0, "scale": 0.5}),
        # Asymmetric truncation
        ([0.0, 1.0, -3.0, 1.0], {"a": -3.0, "b": 1.0, "loc": 0.0, "scale": 1.0}),
        ([2.0, 1.5, -1.0, 5.0], {"a": -2.0, "b": 2.0, "loc": 2.0, "scale": 1.5}),
    ],
)
def test_truncatednormal_vs_scipy(params, sp_params):
    p_params = make_params(*params, dtype="float64")
    lower, upper = params[2], params[3]
    support = (lower, upper)

    run_distribution_tests(
        p_dist=TruncatedNormal,
        sp_dist=stats.truncnorm,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="truncatednormal",
        use_quantiles_for_rvs=True,
        skewness_rtol=1e-4,
        kurtosis_rtol=1e-4,
    )


def test_truncatednormal_extreme_tail():
    # mu=0, sigma=1, support=[10, 12]
    # Scipy fails here, but our implementation should be robust
    mu = pt.as_tensor_variable(0.0)
    sigma = pt.as_tensor_variable(1.0)
    lower = pt.as_tensor_variable(10.0)
    upper = pt.as_tensor_variable(12.0)

    pdf_res = TruncatedNormal.pdf(11.0, mu, sigma, lower, upper).eval()
    assert np.isfinite(pdf_res)
    assert pdf_res > 0

    entropy_res = TruncatedNormal.entropy(mu, sigma, lower, upper).eval()
    assert np.isfinite(entropy_res)

    ppf_res = TruncatedNormal.ppf(0.5, mu, sigma, lower, upper).eval()
    assert np.isfinite(ppf_res)
    assert 10.0 <= ppf_res <= 12.0
