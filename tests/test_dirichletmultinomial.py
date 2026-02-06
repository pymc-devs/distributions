"""Test Dirichlet-Multinomial distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import dirichlet_multinomial, kurtosis, skew

from distributions import dirichletmultinomial as DirichletMultinomial

TEST_CASES = [
    (10, np.array([1.0, 1.0])),
    (20, np.array([2.0, 3.0, 5.0])),
    (15, np.array([0.5, 0.5, 0.5])),
    (30, np.array([10.0, 10.0, 10.0])),
]


@pytest.mark.parametrize("n, a", TEST_CASES)
def test_dirichletmultinomial_logpdf(n, a):
    scipy_dist = dirichlet_multinomial(alpha=a, n=n)
    p_n = pt.constant(n)
    p_a = pt.constant(a)

    mean_val = scipy_dist.mean()
    x = np.round(mean_val).astype(int)
    x[-1] = n - x[:-1].sum()

    actual = DirichletMultinomial.logpdf(x, p_n, p_a).eval()
    expected = scipy_dist.logpmf(x)
    assert_allclose(actual, expected, rtol=1e-5, err_msg=f"logpdf at mean failed for n={n}, a={a}")


@pytest.mark.parametrize("n, a", TEST_CASES)
def test_dirichletmultinomial_pdf(n, a):
    scipy_dist = dirichlet_multinomial(alpha=a, n=n)

    p_n = pt.constant(n)
    p_a = pt.constant(a)

    mean_val = scipy_dist.mean()
    x = np.round(mean_val).astype(int)
    x[-1] = n - x[:-1].sum()

    actual = DirichletMultinomial.pdf(x, p_n, p_a).eval()
    expected = scipy_dist.pmf(x)
    assert_allclose(actual, expected, rtol=1e-5, err_msg="pdf should match scipy")


@pytest.mark.parametrize("n, a", TEST_CASES)
def test_dirichletmultinomial_moments(n, a):
    scipy_dist = dirichlet_multinomial(alpha=a, n=n)

    p_n = pt.constant(n)
    p_a = pt.constant(a)

    actual_mean = DirichletMultinomial.mean(p_n, p_a).eval()
    expected_mean = scipy_dist.mean()
    assert_allclose(actual_mean, expected_mean, rtol=1e-10, err_msg="Mean should match scipy")

    actual_var = DirichletMultinomial.var(p_n, p_a).eval()
    expected_var = scipy_dist.var()
    assert_allclose(actual_var, expected_var, rtol=1e-10, err_msg="Variance should match scipy")

    actual_std = DirichletMultinomial.std(p_n, p_a).eval()
    expected_std = np.sqrt(expected_var)
    assert_allclose(actual_std, expected_std, rtol=1e-10, err_msg="Std should match sqrt(var)")


@pytest.mark.parametrize("n, a", TEST_CASES)
def test_dirichletmultinomial_rvs(n, a):
    p_n = pt.constant(n)
    p_a = pt.constant(a)
    rng = pt.random.default_rng(432)

    samples = DirichletMultinomial.rvs(p_n, p_a, size=1000, random_state=rng).eval()

    assert samples.shape == (1000, len(a)), f"Shape mismatch: got {samples.shape}"

    assert_allclose(
        samples.sum(axis=1), np.full(1000, n), rtol=1e-6, err_msg="Samples should sum to n"
    )

    expected_mean = n * a / a.sum()
    assert_allclose(
        samples.mean(axis=0),
        expected_mean,
        rtol=0.1,
        atol=1.0,
        err_msg="Sample mean should be close to theoretical mean",
    )

    a_sum = a.sum()
    p = a / a_sum
    expected_var = n * p * (1 - p) * (n + a_sum) / (a_sum + 1)
    assert_allclose(
        samples.var(axis=0),
        expected_var,
        rtol=0.2,
        atol=2.0,
        err_msg="Sample variance should be close to theoretical variance",
    )


def test_dirichletmultinomial_mode():
    n = 20
    a = np.array([2.0, 3.0, 5.0])
    p_n = pt.constant(n)
    p_a = pt.constant(a)

    actual = DirichletMultinomial.mode(p_n, p_a).eval()
    k = len(a)
    a_sum = a.sum()
    expected = np.floor((n - k + 1) * a / (a_sum - k))
    assert_allclose(
        actual, expected, rtol=1e-10, err_msg="Mode should match formula when a > 1 and n > k-1"
    )

    n = 1
    a = np.array([0.5, 0.5, 0.5])
    p_n = pt.constant(n)
    p_a = pt.constant(a)

    actual = DirichletMultinomial.mode(p_n, p_a).eval()
    assert np.all(np.isnan(actual)), "Mode should be NaN when conditions not met"


def test_dirichletmultinomial_constraints():
    n = 10
    a = np.array([2.0, 3.0])
    p_n = pt.constant(n)
    p_a = pt.constant(a)

    x = np.array([-1, 11])
    actual = DirichletMultinomial.logpdf(x, p_n, p_a).eval()
    assert actual == -np.inf, "logpdf should be -inf for negative values"

    x = np.array([5, 6])
    actual = DirichletMultinomial.logpdf(x, p_n, p_a).eval()
    assert actual == -np.inf, "logpdf should be -inf when sum != n"


@pytest.mark.parametrize("n, a", TEST_CASES)
def test_dirichletmultinomial_skewness(n, a):
    from distributions import dirichletmultinomial as DM

    p_n = pt.constant(n)
    p_a = pt.constant(a)

    samples = DM.rvs(n, a, size=50000).eval()

    theoretical_skew = DM.skewness(p_n, p_a).eval()
    empirical_skew = np.array([skew(samples[:, i]) for i in range(len(a))])

    assert_allclose(
        theoretical_skew,
        empirical_skew,
        rtol=0.2,
        atol=0.05,
        err_msg=f"Skewness should match empirical for n={n}, a={a}",
    )


@pytest.mark.parametrize("n, a", [(10, np.array([1.0, 1.0]))])
def test_dirichletmultinomial_kurtosis(n, a):
    p_n = pt.constant(n)
    p_a = pt.constant(a)

    samples = DirichletMultinomial.rvs(n, a, size=50000).eval()

    theoretical_kurt = DirichletMultinomial.kurtosis(p_n, p_a).eval()

    empirical_kurt = np.array([kurtosis(samples[:, i]) for i in range(len(a))])

    assert_allclose(
        theoretical_kurt,
        empirical_kurt,
        rtol=0.3,
        atol=0.3,
    )
