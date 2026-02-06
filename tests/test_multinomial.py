"""Test Multinomial distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import multinomial as scipy_multinomial

from distributions import multinomial as Multinomial

TEST_CASES = [
    (10, np.array([0.5, 0.5])),
    (20, np.array([0.2, 0.3, 0.5])),
    (15, np.array([0.25, 0.25, 0.25, 0.25])),
    (30, np.array([0.1, 0.2, 0.3, 0.4])),
]


@pytest.mark.parametrize("n, p", TEST_CASES)
def test_multinomial_logpdf(n, p):
    scipy_dist = scipy_multinomial(n=n, p=p)

    p_n = pt.constant(n)
    p_p = pt.constant(p)

    x = np.round(scipy_dist.mean()).astype(int)
    x[-1] = n - x[:-1].sum()

    actual = Multinomial.logpdf(x, p_n, p_p).eval()
    expected = scipy_dist.logpmf(x)
    assert_allclose(actual, expected, rtol=1e-5, err_msg=f"logpdf at mean failed for n={n}, p={p}")


@pytest.mark.parametrize("n, p", TEST_CASES)
def test_multinomial_pdf(n, p):
    scipy_dist = scipy_multinomial(n=n, p=p)

    p_n = pt.constant(n)
    p_p = pt.constant(p)

    x = np.round(scipy_dist.mean()).astype(int)
    x[-1] = n - x[:-1].sum()

    actual = Multinomial.pdf(x, p_n, p_p).eval()
    expected = scipy_dist.pmf(x)
    assert_allclose(actual, expected, rtol=1e-5, err_msg="pdf should match scipy")


@pytest.mark.parametrize("n, p", TEST_CASES)
def test_multinomial_moments(n, p):
    scipy_dist = scipy_multinomial(n=n, p=p)

    p_n = pt.constant(n)
    p_p = pt.constant(p)

    actual_mean = Multinomial.mean(p_n, p_p).eval()
    expected_mean = scipy_dist.mean()
    assert_allclose(actual_mean, expected_mean, rtol=1e-10, err_msg="Mean should match scipy")

    actual_var = Multinomial.var(p_n, p_p).eval()
    expected_var = np.diag(scipy_dist.cov())
    assert_allclose(actual_var, expected_var, rtol=1e-10, err_msg="Variance should match scipy")

    actual_std = Multinomial.std(p_n, p_p).eval()
    expected_std = np.sqrt(expected_var)
    assert_allclose(actual_std, expected_std, rtol=1e-10, err_msg="Std should match scipy")


@pytest.mark.parametrize("n, p", TEST_CASES)
def test_multinomial_rvs(n, p):
    samples = Multinomial.rvs(n, p, size=1000).eval()

    assert np.allclose(samples.sum(axis=1), n), "All samples should sum to n"

    expected_mean = n * p
    assert_allclose(
        samples.mean(axis=0),
        expected_mean,
        rtol=0.15,
        err_msg="Sample mean should be close to theoretical mean",
    )

    expected_var = np.diag(n * (np.diag(p) - np.outer(p, p)))
    assert_allclose(
        samples.var(axis=0),
        expected_var,
        rtol=0.2,
        atol=2.0,
        err_msg="Sample variance should be close to theoretical variance",
    )


def test_multinomial_mode():
    n = 20
    p = np.array([0.2, 0.3, 0.5])
    p_n = pt.constant(n)
    p_p = pt.constant(p)

    actual = Multinomial.mode(p_n, p_p).eval()
    expected = np.floor((n + 1) * p)
    assert_allclose(actual, expected, rtol=1e-10, err_msg="Mode should match formula")


def test_multinomial_constraints():
    """Test that logpdf returns -inf for invalid inputs."""
    n = 10
    p = np.array([0.5, 0.5])
    p_n = pt.constant(n)
    p_p = pt.constant(p)

    x = np.array([-1, 11])
    actual = Multinomial.logpdf(x, p_n, p_p).eval()
    assert actual == -np.inf, "logpdf should be -inf for negative values"

    x = np.array([5, 6])
    actual = Multinomial.logpdf(x, p_n, p_p).eval()
    assert actual == -np.inf, "logpdf should be -inf when sum != n"
