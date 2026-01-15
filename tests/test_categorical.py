"""Test Categorical distribution against scipy implementation."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from distributions import categorical as Categorical


def make_params(p, dtype="float64"):
    return (pt.as_tensor_variable(np.array(p, dtype=dtype)),)


def make_scipy_categorical(probs):
    """Create a scipy rv_discrete that acts as a categorical distribution."""
    probs = np.asarray(probs)
    probs = probs / probs.sum()
    xk = np.arange(len(probs))
    return stats.rv_discrete(values=(xk, probs))


@pytest.mark.parametrize(
    "probs",
    [
        [0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.2, 0.1],
        [0.01, 0.01, 0.98],
    ],
)
def test_categorical_vs_scipy(probs):
    """Test Categorical distribution against scipy.stats.rv_discrete."""
    p_params = make_params(probs)
    k = len(probs)
    scipy_dist = make_scipy_categorical(probs)
    param_info = f"\nCategorical params: {probs}"

    # Entropy
    assert_allclose(
        Categorical.entropy(*p_params).eval(),
        scipy_dist.entropy(),
        rtol=1e-6,
        err_msg=f"Entropy test failed with {param_info}",
    )

    # Mean
    assert_allclose(
        Categorical.mean(*p_params).eval(),
        scipy_dist.mean(),
        rtol=1e-6,
        err_msg=f"Mean test failed with {param_info}",
    )

    # Variance
    assert_allclose(
        Categorical.var(*p_params).eval(),
        scipy_dist.var(),
        rtol=1e-6,
        err_msg=f"Variance test failed with {param_info}",
    )

    # Standard deviation
    assert_allclose(
        Categorical.std(*p_params).eval(),
        scipy_dist.std(),
        rtol=1e-6,
        err_msg=f"Std test failed with {param_info}",
    )

    # Skewness
    assert_allclose(
        Categorical.skewness(*p_params).eval(),
        scipy_dist.stats(moments="s"),
        rtol=1e-6,
        err_msg=f"Skewness test failed with {param_info}",
    )

    # Kurtosis
    assert_allclose(
        Categorical.kurtosis(*p_params).eval(),
        scipy_dist.stats(moments="k"),
        rtol=1e-6,
        err_msg=f"Kurtosis test failed with {param_info}",
    )

    # Median
    assert_allclose(
        Categorical.median(*p_params).eval(),
        scipy_dist.median(),
        err_msg=f"Median test failed with {param_info}",
    )

    # Test values including outside support
    x_vals = np.array([-2, -1, 0, 1, k - 1, k, k + 1], dtype="float64")
    x_vals = x_vals[x_vals <= k + 1]  # Adjust for small k

    # PMF
    assert_allclose(
        Categorical.pdf(x_vals, *p_params).eval(),
        scipy_dist.pmf(x_vals),
        rtol=1e-6,
        err_msg=f"PMF test failed with {param_info}",
    )

    # logPMF
    assert_allclose(
        Categorical.logpdf(x_vals, *p_params).eval(),
        scipy_dist.logpmf(x_vals),
        rtol=1e-6,
        err_msg=f"logPMF test failed with {param_info}",
    )

    # CDF
    assert_allclose(
        Categorical.cdf(x_vals, *p_params).eval(),
        scipy_dist.cdf(x_vals),
        rtol=1e-6,
        err_msg=f"CDF test failed with {param_info}",
    )

    # logCDF
    assert_allclose(
        Categorical.logcdf(x_vals, *p_params).eval(),
        scipy_dist.logcdf(x_vals),
        rtol=1e-6,
        err_msg=f"logCDF test failed with {param_info}",
    )

    # SF
    assert_allclose(
        Categorical.sf(x_vals, *p_params).eval(),
        scipy_dist.sf(x_vals),
        rtol=1e-6,
        err_msg=f"SF test failed with {param_info}",
    )

    # logSF
    assert_allclose(
        Categorical.logsf(x_vals, *p_params).eval(),
        scipy_dist.logsf(x_vals),
        rtol=1e-6,
        err_msg=f"logSF test failed with {param_info}",
    )

    # PPF
    q_vals = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    assert_allclose(
        Categorical.ppf(q_vals, *p_params).eval(),
        scipy_dist.ppf(q_vals),
        err_msg=f"PPF test failed with {param_info}",
    )

    # ISF
    assert_allclose(
        Categorical.isf(q_vals, *p_params).eval(),
        scipy_dist.isf(q_vals),
        err_msg=f"ISF test failed with {param_info}",
    )

    # RVS - compare quantiles since RNGs differ
    rng_p = pt.random.default_rng(42)
    p_rvs = Categorical.rvs(*p_params, size=50_000, random_state=rng_p).eval()
    s_rvs = scipy_dist.rvs(size=50_000, random_state=np.random.default_rng(42))
    assert_allclose(
        np.quantile(p_rvs, [0.25, 0.5, 0.75]),
        np.quantile(s_rvs, [0.25, 0.5, 0.75]),
        rtol=1e-1,
        err_msg=f"RVS quantiles test failed with {param_info}",
    )

    # Mode - verify it's at a maximum of the PMF
    mode_val = Categorical.mode(*p_params).eval()
    pmf_mode = Categorical.pdf(mode_val, *p_params).eval()
    for i in range(k):
        pmf_i = Categorical.pdf(i, *p_params).eval()
        assert pmf_mode >= pmf_i - 1e-10, (
            f"Mode test failed: pmf({mode_val})={pmf_mode} < pmf({i})={pmf_i}"
        )


def test_categorical_batched_p():
    """Test with batched probability vectors (moments only)."""
    p = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    p_pt = pt.as_tensor_variable(p)

    # Mean: [1.6, 1.0]
    expected_mean = np.array([1.6, 1.0])
    assert_allclose(Categorical.mean(p_pt).eval(), expected_mean)

    # Variance
    expected_var = np.array(
        [0.1 * 1.6**2 + 0.2 * 0.6**2 + 0.7 * 0.4**2, 0.3 * 1.0**2 + 0.4 * 0.0**2 + 0.3 * 1.0**2]
    )
    assert_allclose(Categorical.var(p_pt).eval(), expected_var, rtol=1e-10)
