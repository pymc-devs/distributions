"""Test Multivariate Normal distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from distributions import mvnormal as MvNormal

# Test parameters for multivariate normal
TEST_CASES = [
    (np.array([0.0, 0.0]), np.eye(2)),
    (np.array([1.0, -1.0]), np.array([[2.0, 0.0], [0.0, 0.5]])),
    (np.array([0.0, 0.0, 0.0]), np.eye(3)),
    (
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 0.5, 0.2], [0.5, 2.0, -0.3], [0.2, -0.3, 0.8]]),
    ),
]


@pytest.mark.parametrize("mu, cov", TEST_CASES)
def test_mvnormal_logpdf(mu, cov):
    scipy_dist = multivariate_normal(mean=mu, cov=cov)

    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvNormal.logpdf(mu, p_mu, p_cov).eval()
    expected = scipy_dist.logpdf(mu)
    assert_allclose(actual, expected, rtol=1e-3, err_msg=f"logpdf at mean failed for mu={mu}")

    x_samples = scipy_dist.rvs(size=10, random_state=814)
    actual = MvNormal.logpdf(x_samples, p_mu, p_cov).eval()
    expected = scipy_dist.logpdf(x_samples)
    assert_allclose(actual, expected, rtol=1e-5, err_msg=f"logpdf at samples failed for mu={mu}")


@pytest.mark.parametrize("mu, cov", TEST_CASES)
def test_mvnormal_pdf(mu, cov):
    scipy_dist = multivariate_normal(mean=mu, cov=cov)

    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvNormal.pdf(mu, p_mu, p_cov).eval()
    expected = scipy_dist.pdf(mu)
    assert_allclose(actual, expected, rtol=1e-5, err_msg="pdf at mean failed")


@pytest.mark.parametrize("mu, cov", TEST_CASES)
def test_mvnormal_moments(mu, cov):
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvNormal.mean(p_mu, p_cov).eval()
    assert_allclose(actual, mu, rtol=1e-10, err_msg="Mean should equal mu")

    actual = MvNormal.mode(p_mu, p_cov).eval()
    assert_allclose(actual, mu, rtol=1e-10, err_msg="Mode should equal mu")

    actual = MvNormal.median(p_mu, p_cov).eval()
    assert_allclose(actual, mu, rtol=1e-10, err_msg="Median should equal mu")

    actual = MvNormal.skewness(p_mu, p_cov).eval()
    expected = np.zeros_like(mu)
    assert_allclose(actual, expected, atol=1e-10, err_msg="Skewness should be zero")

    actual = MvNormal.kurtosis(p_mu, p_cov).eval()
    expected = np.zeros_like(mu)
    assert_allclose(actual, expected, atol=1e-10, err_msg="Kurtosis should be zero")


@pytest.mark.parametrize("mu, cov", TEST_CASES)
def test_mvnormal_var(mu, cov):
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvNormal.var(p_mu, p_cov).eval()
    expected = np.diagonal(cov)
    assert_allclose(actual, expected, rtol=1e-10, err_msg="Variance should equal diagonal of cov")


@pytest.mark.parametrize("mu, cov", TEST_CASES)
def test_mvnormal_std(mu, cov):
    """Test standard deviation."""
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvNormal.std(p_mu, p_cov).eval()
    expected = np.sqrt(np.diagonal(cov))
    assert_allclose(actual, expected, rtol=1e-10, err_msg="Std should equal sqrt of diagonal")


@pytest.mark.parametrize("mu, cov", TEST_CASES)
def test_mvnormal_entropy(mu, cov):
    scipy_dist = multivariate_normal(mean=mu, cov=cov)

    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)

    actual = MvNormal.entropy(p_mu, p_cov).eval()
    expected = scipy_dist.entropy()
    assert_allclose(actual, expected, rtol=1e-5, err_msg="Entropy test failed")


@pytest.mark.parametrize("mu, cov", TEST_CASES)
def test_mvnormal_rvs(mu, cov):
    p_mu = pt.constant(mu)
    p_cov = pt.constant(cov)
    rng = pt.random.default_rng(205)

    samples = MvNormal.rvs(p_mu, p_cov, size=10000, random_state=rng).eval()

    assert samples.shape == (10000, len(mu)), f"Shape mismatch: got {samples.shape}"
    assert_allclose(samples.mean(axis=0), mu, atol=0.2, err_msg="Sample mean should be close to mu")
    assert_allclose(
        np.cov(samples.T), cov, rtol=0.2, atol=0.1, err_msg="Sample cov should be close to cov"
    )
