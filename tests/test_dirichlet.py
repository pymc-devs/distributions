"""Test Dirichlet distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import dirichlet, kurtosis, skew

from distributions import dirichlet as Dirichlet

TEST_CASES = [
    np.array([1.0, 1.0]),
    np.array([2.0, 3.0, 5.0]),
    np.array([0.5, 0.5, 0.5]),
    np.array([10.0, 10.0, 10.0]),
]


@pytest.mark.parametrize("alpha", TEST_CASES)
def test_dirichlet_logpdf(alpha):
    scipy_dist = dirichlet(alpha=alpha)

    p_alpha = pt.constant(alpha)

    x = scipy_dist.mean()
    actual = Dirichlet.logpdf(x, p_alpha).eval()
    expected = scipy_dist.logpdf(x)
    assert_allclose(actual, expected, rtol=1e-5, err_msg=f"logpdf at mean failed for alpha={alpha}")

    x_samples = scipy_dist.rvs(size=10, random_state=814)
    actual = Dirichlet.logpdf(x_samples, p_alpha).eval()
    expected = scipy_dist.logpdf(x_samples.T)
    assert_allclose(
        actual, expected, rtol=1e-5, err_msg=f"logpdf at samples failed for alpha={alpha}"
    )


@pytest.mark.parametrize("alpha", TEST_CASES)
def test_dirichlet_pdf(alpha):
    scipy_dist = dirichlet(alpha=alpha)

    p_alpha = pt.constant(alpha)

    x = scipy_dist.mean()
    actual = Dirichlet.pdf(x, p_alpha).eval()
    expected = scipy_dist.pdf(x)
    assert_allclose(actual, expected, rtol=1e-5, err_msg="pdf at mean failed")


@pytest.mark.parametrize("alpha", TEST_CASES)
def test_dirichlet_moments(alpha):
    scipy_dist = dirichlet(alpha=alpha)

    p_alpha = pt.constant(alpha)

    actual = Dirichlet.mean(p_alpha).eval()
    expected = scipy_dist.mean()
    assert_allclose(actual, expected, rtol=1e-10, err_msg="Mean should match scipy")

    actual = Dirichlet.var(p_alpha).eval()
    expected = scipy_dist.var()
    assert_allclose(actual, expected, rtol=1e-10, err_msg="Variance should match scipy")

    actual = Dirichlet.std(p_alpha).eval()
    expected = np.sqrt(scipy_dist.var())
    assert_allclose(actual, expected, rtol=1e-10, err_msg="Std should match scipy")


@pytest.mark.parametrize("alpha", TEST_CASES)
def test_dirichlet_skewness_kurtosis(alpha):
    p_alpha = pt.constant(alpha)
    rng = pt.random.default_rng(432)

    samples = Dirichlet.rvs(p_alpha, size=50000, random_state=rng).eval()

    theoretical_skew = Dirichlet.skewness(p_alpha).eval()
    empirical_skew = skew(samples, axis=0)

    assert_allclose(
        theoretical_skew,
        empirical_skew,
        rtol=0.1,
        atol=0.05,
        err_msg=f"Theoretical skewness should match empirical for alpha={alpha}",
    )

    theoretical_kurt = Dirichlet.kurtosis(p_alpha).eval()
    empirical_kurt = kurtosis(samples, axis=0)

    assert_allclose(
        theoretical_kurt,
        empirical_kurt,
        rtol=0.15,
        atol=0.1,
        err_msg=f"Theoretical kurtosis should match empirical for alpha={alpha}",
    )


@pytest.mark.parametrize("alpha", TEST_CASES)
def test_dirichlet_entropy(alpha):
    scipy_dist = dirichlet(alpha=alpha)

    p_alpha = pt.constant(alpha)

    actual = Dirichlet.entropy(p_alpha).eval()
    expected = scipy_dist.entropy()
    assert_allclose(actual, expected, rtol=1e-5, err_msg="Entropy test failed")


@pytest.mark.parametrize("alpha", TEST_CASES)
def test_dirichlet_rvs(alpha):
    p_alpha = pt.constant(alpha)
    rng = pt.random.default_rng(432)

    samples = Dirichlet.rvs(p_alpha, size=1000, random_state=rng).eval()

    assert samples.shape == (1000, len(alpha)), f"Shape mismatch: got {samples.shape}"

    assert_allclose(
        samples.sum(axis=1), np.ones(1000), rtol=1e-6, err_msg="Samples should sum to 1"
    )

    assert_allclose(
        samples.mean(axis=0),
        alpha / alpha.sum(),
        atol=0.05,
        err_msg="Sample mean should be close to theoretical mean",
    )

    alpha_sum = alpha.sum()
    expected_var = (alpha * (alpha_sum - alpha)) / (alpha_sum**2 * (alpha_sum + 1))
    assert_allclose(
        samples.var(axis=0),
        expected_var,
        rtol=0.15,
        atol=0.01,
        err_msg="Sample variance should be close to theoretical variance",
    )


def test_dirichlet_mode():
    alpha = np.array([2.0, 3.0, 5.0])
    p_alpha = pt.constant(alpha)

    actual = Dirichlet.mode(p_alpha).eval()
    expected = (alpha - 1) / (alpha.sum() - len(alpha))
    assert_allclose(
        actual, expected, rtol=1e-10, err_msg="Mode should match formula when alpha > 1"
    )

    alpha = np.array([0.5, 0.5, 0.5])
    p_alpha = pt.constant(alpha)

    actual = Dirichlet.mode(p_alpha).eval()
    assert np.all(np.isnan(actual)), "Mode should be NaN when any alpha <= 1"


def test_dirichlet_constraints():
    """Test that logpdf returns -inf for invalid inputs."""
    alpha = np.array([2.0, 3.0])
    p_alpha = pt.constant(alpha)

    x = np.array([-0.1, 1.1])
    actual = Dirichlet.logpdf(x, p_alpha).eval()
    assert actual == -np.inf, "logpdf should be -inf for negative values"

    x = np.array([0.5, 1.5])
    actual = Dirichlet.logpdf(x, p_alpha).eval()
    assert actual == -np.inf, "logpdf should be -inf for values > 1"
