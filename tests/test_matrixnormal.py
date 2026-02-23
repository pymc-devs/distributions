"""Test Matrix Normal distribution."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import matrix_normal as scipy_matrix_normal

from pytensor_distributions import matrixnormal as MatrixNormal

# Test cases: (M, U, V)
TEST_CASES = [
    # 2x2 with identity covariances
    (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.eye(2),
        np.eye(2),
    ),
    # 2x3 with non-trivial covariances
    (
        np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]]),
        np.array([[2.0, 0.5], [0.5, 1.0]]),
        np.array([[1.0, 0.3, 0.1], [0.3, 2.0, 0.4], [0.1, 0.4, 1.5]]),
    ),
    # 3x2 with correlated covariances
    (
        np.zeros((3, 2)),
        np.array([[3.0, 1.0, 0.5], [1.0, 2.0, 0.3], [0.5, 0.3, 1.5]]),
        np.array([[1.0, 0.7], [0.7, 2.0]]),
    ),
]


@pytest.mark.parametrize("M, U, V", TEST_CASES)
def test_matrixnormal_mean(M, U, V):
    actual = MatrixNormal.mean(pt.constant(M), pt.constant(U), pt.constant(V)).eval()
    assert_allclose(actual, M, rtol=1e-10)


@pytest.mark.parametrize("M, U, V", TEST_CASES)
def test_matrixnormal_mode(M, U, V):
    actual = MatrixNormal.mode(pt.constant(M), pt.constant(U), pt.constant(V)).eval()
    assert_allclose(actual, M, rtol=1e-10)


@pytest.mark.parametrize("M, U, V", TEST_CASES)
def test_matrixnormal_logpdf(M, U, V):
    scipy_dist = scipy_matrix_normal(mean=M, rowcov=U, colcov=V)
    X = M + 0.1 * np.ones_like(M)

    actual = MatrixNormal.logpdf(
        pt.constant(X), pt.constant(M), pt.constant(U), pt.constant(V)
    ).eval()
    expected = scipy_dist.logpdf(X)
    assert_allclose(actual, expected, rtol=1e-5, err_msg="logpdf should match scipy")


@pytest.mark.parametrize("M, U, V", TEST_CASES)
def test_matrixnormal_pdf(M, U, V):
    scipy_dist = scipy_matrix_normal(mean=M, rowcov=U, colcov=V)
    X = M + 0.1 * np.ones_like(M)

    actual = MatrixNormal.pdf(pt.constant(X), pt.constant(M), pt.constant(U), pt.constant(V)).eval()
    expected = scipy_dist.pdf(X)
    assert_allclose(actual, expected, rtol=1e-5, err_msg="pdf should match scipy")


@pytest.mark.parametrize("M, U, V", TEST_CASES)
def test_matrixnormal_entropy(M, U, V):
    scipy_dist = scipy_matrix_normal(mean=M, rowcov=U, colcov=V)

    actual = MatrixNormal.entropy(pt.constant(M), pt.constant(U), pt.constant(V)).eval()
    expected = scipy_dist.entropy()
    assert_allclose(actual, expected, rtol=1e-5, err_msg="entropy should match scipy")


@pytest.mark.parametrize("M, U, V", TEST_CASES)
def test_matrixnormal_var(M, U, V):
    actual = MatrixNormal.var(pt.constant(M), pt.constant(U), pt.constant(V)).eval()
    expected = np.outer(np.diag(U), np.diag(V))
    assert_allclose(actual, expected, rtol=1e-10, err_msg="var should be outer product of diags")


@pytest.mark.parametrize("M, U, V", TEST_CASES)
def test_matrixnormal_rvs(M, U, V):
    m, n = M.shape

    sample = MatrixNormal.rvs(pt.constant(M), pt.constant(U), pt.constant(V), size=None).eval()
    assert sample.shape == (m, n), f"Single sample should have shape ({m}, {n})"

    n_samples = 2000
    samples = MatrixNormal.rvs(
        pt.constant(M), pt.constant(U), pt.constant(V), size=n_samples
    ).eval()
    assert samples.shape == (n_samples, m, n)

    sample_mean = np.mean(samples, axis=0)
    assert_allclose(
        sample_mean, M, atol=0.2, err_msg="Sample mean should approximate theoretical mean"
    )

    sample_var = np.var(samples, axis=0)
    theoretical_var = np.outer(np.diag(U), np.diag(V))
    assert_allclose(
        sample_var,
        theoretical_var,
        rtol=0.3,
        atol=0.2,
        err_msg="Sample variance should approximate theoretical variance",
    )
