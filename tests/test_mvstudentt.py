"""Test Multivariate Student's t distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_t

from pytensor_distributions import mvstudentt as MvStudentT

TEST_CASES = [
    (5.0, np.array([0.0, 0.0]), np.eye(2)),
    (3.0, np.array([1.0, -1.0]), np.array([[2.0, 0.0], [0.0, 0.5]])),
    (10.0, np.array([0.0, 0.0, 0.0]), np.eye(3)),
    (
        2.5,
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 0.5, 0.2], [0.5, 2.0, -0.3], [0.2, -0.3, 0.8]]),
    ),
]


@pytest.mark.parametrize("nu, mu, cov", TEST_CASES)
def test_mvstudentt_logpdf(nu, mu, cov):
    scipy_dist = multivariate_t(df=nu, loc=mu, shape=cov)

    p_nu = pt.constant(nu)
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvStudentT.logpdf(mu, p_nu, p_mu, p_cov).eval()
    expected = scipy_dist.logpdf(mu)
    assert_allclose(actual, expected, rtol=1e-3, err_msg=f"logpdf at mean failed for nu={nu}")

    x_samples = scipy_dist.rvs(size=10, random_state=814)
    actual = MvStudentT.logpdf(x_samples, p_nu, p_mu, p_cov).eval()
    expected = scipy_dist.logpdf(x_samples)
    assert_allclose(actual, expected, rtol=1e-5, err_msg=f"logpdf at samples failed for nu={nu}")


@pytest.mark.parametrize("nu, mu, cov", TEST_CASES)
def test_mvstudentt_pdf(nu, mu, cov):
    scipy_dist = multivariate_t(df=nu, loc=mu, shape=cov)

    p_nu = pt.constant(nu)
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvStudentT.pdf(mu, p_nu, p_mu, p_cov).eval()
    expected = scipy_dist.pdf(mu)
    assert_allclose(actual, expected, rtol=1e-5, err_msg="pdf at mean failed")


@pytest.mark.parametrize("nu, mu, cov", TEST_CASES)
def test_mvstudentt_moments(nu, mu, cov):
    p_nu = pt.constant(nu)
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvStudentT.mean(p_nu, p_mu, p_cov).eval()
    assert_allclose(actual, mu, rtol=1e-10, err_msg="Mean should equal mu")

    actual = MvStudentT.mode(p_nu, p_mu, p_cov).eval()
    assert_allclose(actual, mu, rtol=1e-10, err_msg="Mode should equal mu")

    actual = MvStudentT.median(p_nu, p_mu, p_cov).eval()
    assert_allclose(actual, mu, rtol=1e-10, err_msg="Median should equal mu")

    actual = MvStudentT.skewness(p_nu, p_mu, p_cov).eval()
    if nu > 3:
        expected = np.zeros_like(mu)
        assert_allclose(actual, expected, atol=1e-10, err_msg="Skewness should be zero")
    else:
        assert np.all(np.isnan(actual)), "Skewness should be NaN when nu <= 3"

    actual = MvStudentT.kurtosis(p_nu, p_mu, p_cov).eval()
    if nu > 4:
        k = len(mu)
        expected = 6 * (nu - 2) / ((k + 2) * (nu - 4))
        expected = np.full_like(mu, expected)
        assert_allclose(actual, expected, rtol=1e-10, err_msg="Kurtosis formula check")
    else:
        expected = np.full_like(mu, np.inf)
        assert np.all(np.isinf(actual)), "Kurtosis should be infinite when nu <= 4"


@pytest.mark.parametrize("nu, mu, cov", TEST_CASES)
def test_mvstudentt_var(nu, mu, cov):
    p_nu = pt.constant(nu)
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvStudentT.var(p_nu, p_mu, p_cov).eval()
    expected = nu / (nu - 2) * np.diagonal(cov)
    assert_allclose(
        actual, expected, rtol=1e-10, err_msg="Variance should equal nu/(nu-2) * diagonal of cov"
    )


@pytest.mark.parametrize("nu, mu, cov", TEST_CASES)
def test_mvstudentt_entropy(nu, mu, cov):
    scipy_dist = multivariate_t(df=nu, loc=mu, shape=cov)

    p_nu = pt.constant(nu)
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvStudentT.entropy(p_nu, p_mu, p_cov).eval()
    expected = scipy_dist.entropy()
    assert_allclose(actual, expected, rtol=1e-5, err_msg="Entropy test failed")


@pytest.mark.parametrize("nu, mu, cov", TEST_CASES)
def test_mvstudentt_rvs(nu, mu, cov):
    p_nu = pt.constant(nu)
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)
    rng = pt.random.default_rng(205)

    samples = MvStudentT.rvs(p_nu, p_mu, p_cov, size=50000, random_state=rng).eval()

    assert samples.shape == (50000, len(mu)), f"Shape mismatch: got {samples.shape}"
    assert_allclose(samples.mean(axis=0), mu, atol=0.1, err_msg="Sample mean should be close to mu")

    rtol = 0.15 if nu <= 5 else 0.01
    assert_allclose(
        np.cov(samples, rowvar=False).diagonal(),
        np.diagonal(cov) * nu / (nu - 2),
        rtol=rtol,
        err_msg="Sample cov diagonal should match theoretical",
    )
