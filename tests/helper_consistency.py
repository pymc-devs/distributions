"""
Consistency Test Framework for Probability Distributions

This module provides reusable functions for testing mathematical consistency
of probability distribution implementations without relying on reference implementations.
"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.stats import skew, kurtosis
from scipy.integrate import quad
import pytensor.tensor as pt


# def support_points(low, high, dist, params, n=20, is_discrete=False):
#     """Return representative x points in support."""
#     if np.isfinite(low) and np.isfinite(high):
#         if is_discrete:
#             return np.arange(low, high + 1)
#         return np.linspace(low, high, n)
#     if is_discrete:
#         return np.arange(dist.ppf(0.0001, *params).eval(), dist.ppf(0.9999, *params).eval() + 1)
#     ps = np.linspace(0.0001, 0.9999, n)
#     return dist.ppf(ps, *params).eval()


def xvals(low, high, dist, params, is_discrete, n_points=None):
    lower_ep, upper_ep = finite_endpoints(dist, params, low, high)

    if is_discrete:
        if n_points is None:
            n_points = 200
        upper_ep = int(upper_ep)
        lower_ep = int(lower_ep)
        range_x = upper_ep - lower_ep
        if range_x <= n_points:
            x_vals = np.arange(lower_ep, upper_ep + 1, dtype=int)
        else:
            x_vals = np.linspace(lower_ep, upper_ep + 1, n_points, dtype=int)

        return x_vals
    else:
        if n_points is None:
            n_points = 1000
        return np.linspace(lower_ep, upper_ep, n_points)


def finite_endpoints(dist, params, lower_ep, upper_ep):
    """Return finite endpoints for support, using PPF if necessary."""
    if not np.isfinite(lower_ep):
        lower_ep = dist.ppf(0.0001, *params).eval()
    if not np.isfinite(upper_ep):
        upper_ep = dist.ppf(0.9999, *params).eval()
    return lower_ep, upper_ep


def test_pdf_normalization(dist, params_tensor, params_value, support, is_discrete, atol=1e-2):
    """Check that PDF integrates/sums to 1."""
    xs = xvals(*support, dist, params_tensor, is_discrete)

    if is_discrete:
        total = dist.pdf(xs, *params_tensor).eval().sum()
    else:
        total = dist.pdf(xs, *params_tensor).eval().sum() * (xs[-1] - xs[0]) / (len(xs) - 1)

    assert abs(total - 1) < atol, f"PDF normalization failed: {total} (params: {params_value})"


def test_pdf_logpdf_consistency(dist, params_tensor, params_value, support, is_discrete, atol=1e-8):
    """Check that logpdf(x) = log(pdf(x))."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=10)
    pdf = dist.pdf(x, *params_tensor).eval()
    logpdf = dist.logpdf(x, *params_tensor).eval()
    mask = pdf > 0
    assert np.allclose(
        np.log(pdf[mask]), logpdf[mask], atol=atol
    ), f"PDF/LogPDF consistency failed (params: {params_value})"


def test_pdf_cdf_consistency(
    dist, params_tensor, params_value, support, is_discrete, atol=1e-3, rtol=1e-3
):
    """Check that PDF is the derivative of CDF."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=20)

    if is_discrete:
        pmf_values = dist.pdf(x, *params_tensor).eval()
        cdf_from_pmf = np.cumsum(pmf_values)
        actual_cdf = dist.cdf(x, *params_tensor).eval()
        assert_allclose(
            actual_cdf,
            cdf_from_pmf,
            atol=atol,
            rtol=rtol,
            err_msg=f"PMF/CDF consistency failed (params: {params_value})",
        )
    else:
        # Check that PDF is the derivative of CDF
        eps = 1e-8
        approx_pdf = (
            dist.cdf(x + eps, *params_tensor).eval() - dist.cdf(x - eps, *params_tensor).eval()
        ) / (2 * eps)
        actual_pdf = dist.pdf(x, *params_tensor).eval()
        assert_allclose(
            actual_pdf,
            approx_pdf,
            atol=atol,
            rtol=rtol,
            err_msg=f"PDF/CDF consistency failed (params: {params_value})",
        )


def test_cdf_logcdf_consistency(dist, params_tensor, params_value, support, is_discrete, atol=1e-8):
    """Check that logcdf(x) = log(cdf(x))."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=10)
    cdf = dist.cdf(x, *params_tensor).eval()
    logcdf = dist.logcdf(x, *params_tensor).eval()
    mask = cdf > 0
    assert np.allclose(
        np.log(cdf[mask]), logcdf[mask], atol=atol
    ), f"CDF/LogCDF consistency failed (params: {params_value})"


def test_cdf_ppf_roundtrip(
    dist, params_tensor, params_value, support, is_discrete, atol=1e-5, rtol=1e-5
):
    """Check that CDF and PPF are inverse functions."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=20)
    ps = np.linspace(0.001, 0.999, 20)

    assert_allclose(
        dist.ppf(dist.cdf(x, *params_tensor).eval(), *params_tensor).eval(),
        x,
        atol=atol,
        rtol=rtol,
        err_msg=f"CDF/PPF roundtrip failed (params: {params_value})",
    )
    if is_discrete:
        assert np.all(
            dist.cdf(dist.ppf(ps, *params_tensor).eval(), *params_tensor).eval() >= ps - atol
        ), f"CDF(PPF(p)) should be >= p for discrete distributions (params: {params_value})"
    else:
        assert_allclose(
            dist.cdf(dist.ppf(ps, *params_tensor).eval(), *params_tensor).eval(),
            ps,
            atol=atol,
            rtol=rtol,
            err_msg=f"PPF/CDF roundtrip failed (params: {params_value})",
        )


def test_monotonicity(dist, params_tensor, params_value, support, is_discrete):
    """Check that CDF and PPF are monotonically increasing."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=100)
    cdf_vals = dist.cdf(x, *params_tensor).eval()
    assert np.all(np.diff(cdf_vals) >= 0), f"CDF is not monotonic (params: {params_value})"
    sf_vals = dist.sf(x, *params_tensor).eval()
    assert np.all(np.diff(sf_vals) <= 0), f"SF is not monotonic (params: {params_value})"

    ps = np.linspace(0.0001, 0.9999, 100)
    ppf_vals = dist.ppf(ps, *params_tensor).eval()
    assert np.all(np.diff(ppf_vals) >= 0), f"PPF is not monotonic (params: {params_value})"
    isf_vals = dist.isf(ps, *params_tensor).eval()
    assert np.all(np.diff(isf_vals) <= 0), f"ISF is not monotonic (params: {params_value})"


def test_sf_isf_roundtrip(
    dist, params_tensor, params_value, support, is_discrete, atol=1e-5, rtol=1e-5
):
    """Check survival function and inverse survival function consistency."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=20)
    ps = np.linspace(0.0001, 0.9999, 100)

    sf_vals = dist.sf(x, *params_tensor).eval()
    cdf_vals = dist.cdf(x, *params_tensor).eval()
    assert np.allclose(sf_vals, 1 - cdf_vals, atol=1e-10)

    # if is_discrete:
    #     # For discrete distributions, SF(ISF(p)) >= p should always hold
    #     isf_ps = dist.isf(ps, *params_tensor).eval()
    #     sf_isf_ps = dist.sf(isf_ps, *params_tensor).eval()
    #     assert np.all(sf_isf_ps <= ps), f"SF(ISF(p)) should be <= p for discrete distributions (params: {params_value})"
    # else:
    # assert np.allclose(dist.isf(sf_vals, *params_tensor).eval(), x, atol=atol, rtol=rtol), f"SF/ISF roundtrip failed (params: {params_value})"
    # assert np.allclose(dist.sf(dist.isf(ps, *params_tensor).eval(), *params_tensor).eval(), ps, atol=atol, rtol=rtol), f"ISF/SF roundtrip failed (params: {params_value})"


def test_sf_logsf_consistency(dist, params_tensor, params_value, support, is_discrete, atol=1e-8):
    """Check that logsf(x) = log(sf(x))."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=10)
    sf_vals = dist.sf(x, *params_tensor).eval()
    logsf_vals = dist.logsf(x, *params_tensor).eval()
    mask = sf_vals > 0
    assert np.allclose(
        np.log(sf_vals[mask]), logsf_vals[mask], atol=atol
    ), f"SF/LogSF consistency failed (params: {params_value})"


def test_cdf_bounds(dist, params_tensor, params_value, support, is_discrete):
    """Check that CDF is bounded between 0 and 1."""
    x = xvals(*support, dist, params_tensor, is_discrete, n_points=100)
    cdf_vals = dist.cdf(x, *params_tensor).eval()
    assert np.all(cdf_vals >= 0), f"CDF has negative values (params: {params_value})"
    assert np.all(cdf_vals <= 1), f"CDF exceeds 1 (params: {params_value})"


def test_moments_consistency(dist, params_tensor, params_value, n_samples=50_000):
    """Check consistency between theoretical and empirical moments."""
    rng = pt.random.default_rng(71094)
    samples = dist.rvs(*params_tensor, n_samples, random_state=rng).eval()

    # Empirical moments
    mean_emp = np.mean(samples)
    std_emp = np.std(samples, ddof=1)  # Use ddof=1 for unbiased estimate
    var_emp = np.var(samples, ddof=1)
    skew_emp = skew(samples)
    kurt_emp = kurtosis(samples)

    # Theoretical moments
    mean_theo = dist.mean(*params_tensor).eval()
    std_theo = dist.std(*params_tensor).eval()
    var_theo = dist.var(*params_tensor).eval()
    skew_theo = dist.skewness(*params_tensor).eval()
    kurt_theo = dist.kurtosis(*params_tensor).eval()

    se_mean = std_emp / np.sqrt(n_samples)
    se_std = std_emp / np.sqrt(2 * n_samples)
    se_var = np.sqrt(2 * var_emp**2 / n_samples)

    assert_allclose(
        mean_emp, mean_theo, atol=4 * se_mean, err_msg=f"Mean mismatch (params: {params_value})"
    )
    assert_allclose(
        std_emp, std_theo, atol=4 * se_std, err_msg=f"Std mismatch (params: {params_value})"
    )
    assert_allclose(
        var_emp, var_theo, atol=4 * se_var, err_msg=f"Var mismatch (params: {params_value})"
    )

    assert_allclose(
        skew_emp,
        skew_theo,
        rtol=0.2,
        atol=0.3,
        err_msg=f"Skewness mismatch (params: {params_value})",
    )
    assert_allclose(
        kurt_emp,
        kurt_theo,
        rtol=0.3,
        atol=0.5,
        err_msg=f"Kurtosis mismatch (params: {params_value})",
    )


def test_mode(dist, params_tensor, params_value, is_discrete):
    """Check that mode is at a local maximum of PDF."""
    mode_val = dist.mode(*params_tensor).eval()

    if is_discrete:
        eps = 1
    else:
        eps = dist.std(*params_tensor).eval() * 0.001

    pdf_mode = dist.pdf(mode_val, *params_tensor).eval()
    pdf_left = dist.pdf(mode_val - eps, *params_tensor).eval()
    pdf_right = dist.pdf(mode_val + eps, *params_tensor).eval()

    assert (
        pdf_mode >= pdf_left - 1e-8
    ), f"Mode PDF {pdf_mode} < left neighbor {pdf_left} (params: {params_value})"
    assert (
        pdf_mode >= pdf_right - 1e-8
    ), f"Mode PDF {pdf_mode} < right neighbor {pdf_right} (params: {params_value})"


def test_median_properties(dist, params_tensor, params_value, is_discrete):
    """Check that median corresponds to CDF = 0.5."""
    median_val = dist.median(*params_tensor).eval()
    cdf_at_median = dist.cdf(median_val, *params_tensor).eval()
    if is_discrete:
        atol = 0.1
    else:
        atol = 1e-6

    assert np.isclose(
        cdf_at_median, 0.5, atol=atol
    ), f"CDF at median should be 0.5, got {cdf_at_median} (params: {params_value})"


def test_entropy_finite(dist, params_tensor, params_value):
    """Check that entropy is finite."""
    entropy_val = dist.entropy(*params_tensor).eval()
    assert np.isfinite(
        entropy_val
    ), f"Entropy should be finite, got {entropy_val}, (params: {params_value})"


def run_all_consistency_tests(dist, param_values, support, is_discrete, skip_tests=None):
    """Run all consistency tests for a distribution with given parameters."""
    params_tensor = tuple(pt.constant(v) for v in param_values)

    if skip_tests is None:
        skip_tests = []

    # test_pdf_normalization(dist, params_tensor, param_values, support, is_discrete)
    # test_pdf_logpdf_consistency(dist, params_tensor, param_values, support, is_discrete)
    # test_cdf_logcdf_consistency(dist, params_tensor, param_values, support, is_discrete)
    # test_pdf_cdf_consistency(dist, params_tensor, param_values, support, is_discrete)
    # test_cdf_ppf_roundtrip(dist, params_tensor, param_values, support, is_discrete)
    test_sf_isf_roundtrip(dist, params_tensor, param_values, support, is_discrete)
    # test_sf_logsf_consistency(dist, params_tensor, param_values, support, is_discrete)
    # test_monotonicity(dist, params_tensor, param_values, support, is_discrete)
    # test_cdf_bounds(dist, params_tensor, param_values, support, is_discrete)
    # test_mode(dist, params_tensor, param_values, is_discrete)
    # test_median_properties(dist, params_tensor, param_values, is_discrete)
    # test_moments_consistency(dist, params_tensor, param_values)
    # test_entropy_finite(dist, params_tensor, param_values)
