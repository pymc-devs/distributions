import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from pytensor_distributions import rice as Rice
from tests.helper_scipy import make_params, run_distribution_tests


@pytest.mark.parametrize(
    "params",
    [
        (0.5, 1.0),  # nu, sigma - low non-centrality
        (1.0, 1.0),  # moderate non-centrality
        (2.0, 1.0),  # higher non-centrality
        (1.0, 0.5),  # smaller scale
        (3.0, 2.0),  # larger parameters
        (10.0, 1.0),  # high non-centrality (tests CDF stability)
        (50.0, 2.0),  # very high non-centrality (uses normal approx for CDF)
    ],
)
def test_rice_vs_scipy(params):
    nu, sigma = params
    b = nu / sigma

    p_params = make_params(nu, sigma, dtype="float64")
    sp_params = {"b": b, "scale": sigma}
    support = (0, float("inf"))

    run_distribution_tests(
        p_dist=Rice,
        sp_dist=stats.rice,
        p_params=p_params,
        sp_params=sp_params,
        support=support,
        name="rice",
        use_quantiles_for_rvs=True,
        mean_rtol=1e-2,
        var_rtol=1e-2,
        std_rtol=1e-2,
        skewness_rtol=1e-1,
        kurtosis_rtol=1e-1,
        entropy_rtol=1e-2,
        cdf_rtol=1e-4,
        sf_rtol=1e-4,
        pdf_rtol=1e-5,
        logpdf_rtol=1e-5,
        logcdf_rtol=1e-3,
        logsf_rtol=1e-3,
    )


def test_rice_vectorized():
    """Test Rice distribution with array parameters."""
    nu_vals = np.array([1.0, 2.0, 3.0])
    sigma_vals = np.array([1.0, 1.0, 1.0])
    p_params = make_params(nu_vals, sigma_vals, dtype="float64")

    # Test mean - should return array of shape (3,)
    mean_result = Rice.mean(*p_params).eval()
    assert mean_result.shape == (3,), f"mean shape mismatch: {mean_result.shape}"
    for i, (nu, sigma) in enumerate(zip(nu_vals, sigma_vals)):
        expected = stats.rice(nu / sigma, scale=sigma).mean()
        assert_allclose(mean_result[i], expected, rtol=1e-2)

    # Test var - should return array of shape (3,)
    var_result = Rice.var(*p_params).eval()
    assert var_result.shape == (3,), f"var shape mismatch: {var_result.shape}"
    for i, (nu, sigma) in enumerate(zip(nu_vals, sigma_vals)):
        expected = stats.rice(nu / sigma, scale=sigma).var()
        assert_allclose(var_result[i], expected, rtol=1e-2)

    # Test mode - should return array of shape (3,)
    mode_result = Rice.mode(*p_params).eval()
    assert mode_result.shape == (3,), f"mode shape mismatch: {mode_result.shape}"
    for i, mode_val in enumerate(mode_result):
        pdf_at_mode = Rice.pdf(mode_val, p_params[0][i], p_params[1][i]).eval()
        pdf_left = Rice.pdf(mode_val - 0.05, p_params[0][i], p_params[1][i]).eval()
        pdf_right = Rice.pdf(mode_val + 0.05, p_params[0][i], p_params[1][i]).eval()
        assert pdf_at_mode >= pdf_left - 1e-4, f"mode test failed (left) at index {i}"
        assert pdf_at_mode >= pdf_right - 1e-4, f"mode test failed (right) at index {i}"

    # Test CDF with array x and array params (same shape)
    x_vals = np.array([1.5, 2.5, 3.5])
    cdf_result = Rice.cdf(x_vals, *p_params).eval()
    assert cdf_result.shape == (3,), f"cdf shape mismatch: {cdf_result.shape}"
    for i, (x, nu, sigma) in enumerate(zip(x_vals, nu_vals, sigma_vals)):
        expected = stats.rice(nu / sigma, scale=sigma).cdf(x)
        assert_allclose(cdf_result[i], expected, rtol=1e-4)

    # Test CDF with scalar x and array params (broadcasting)
    cdf_broadcast = Rice.cdf(2.0, *p_params).eval()
    assert cdf_broadcast.shape == (3,), f"cdf broadcast shape mismatch: {cdf_broadcast.shape}"
    for i, (nu, sigma) in enumerate(zip(nu_vals, sigma_vals)):
        expected = stats.rice(nu / sigma, scale=sigma).cdf(2.0)
        assert_allclose(cdf_broadcast[i], expected, rtol=1e-4)

    # Test entropy - should return array of shape (3,)
    entropy_result = Rice.entropy(*p_params).eval()
    assert entropy_result.shape == (3,), f"entropy shape mismatch: {entropy_result.shape}"
    for i, (nu, sigma) in enumerate(zip(nu_vals, sigma_vals)):
        expected = stats.rice(nu / sigma, scale=sigma).entropy()
        assert_allclose(entropy_result[i], expected, rtol=1e-2)

    # Test skewness - should return array of shape (3,)
    skewness_result = Rice.skewness(*p_params).eval()
    assert skewness_result.shape == (3,), f"skewness shape mismatch: {skewness_result.shape}"
    for i, (nu, sigma) in enumerate(zip(nu_vals, sigma_vals)):
        expected = stats.rice(nu / sigma, scale=sigma).stats(moments="s")
        assert_allclose(skewness_result[i], expected, rtol=1e-1)
