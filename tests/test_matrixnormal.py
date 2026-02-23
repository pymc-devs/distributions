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


# --- Batched tests ---

# Reuse a single test case for batched tests
M0 = np.array([[1.0, 2.0], [3.0, 4.0]])
U0 = np.array([[2.0, 0.5], [0.5, 1.0]])
V0 = np.eye(2)


class TestBatchedLogpdf:
    """Test logpdf with batched observations."""

    def test_batch_of_observations(self):
        Xs = np.array([M0 + 0.1, M0 - 0.1, M0 + 0.5])  # (3, 2, 2)
        result = MatrixNormal.logpdf(
            pt.constant(Xs), pt.constant(M0), pt.constant(U0), pt.constant(V0)
        )
        actual = result.eval()

        scipy_dist = scipy_matrix_normal(mean=M0, rowcov=U0, colcov=V0)
        expected = np.array([scipy_dist.logpdf(x) for x in Xs])

        assert actual.shape == (3,)
        assert_allclose(actual, expected, rtol=1e-5)

    def test_single_observation_still_scalar(self):
        X = M0 + 0.1
        result = MatrixNormal.logpdf(
            pt.constant(X), pt.constant(M0), pt.constant(U0), pt.constant(V0)
        )
        actual = result.eval()
        expected = scipy_matrix_normal(mean=M0, rowcov=U0, colcov=V0).logpdf(X)
        assert actual.shape == ()
        assert_allclose(actual, expected, rtol=1e-5)


class TestBatchedVar:
    """Test var with batched covariance matrices."""

    def test_batched_rowcov(self):
        rowcovs = np.array([np.eye(2), 2 * np.eye(2)])  # (2, 2, 2)
        result = MatrixNormal.var(pt.constant(M0), pt.constant(rowcovs), pt.constant(V0))
        actual = result.eval()
        assert actual.shape == (2, 2, 2)
        for i in range(2):
            expected = np.outer(np.diag(rowcovs[i]), np.diag(V0))
            assert_allclose(actual[i], expected, rtol=1e-10)

    def test_unbatched_still_works(self):
        result = MatrixNormal.var(pt.constant(M0), pt.constant(U0), pt.constant(V0))
        actual = result.eval()
        expected = np.outer(np.diag(U0), np.diag(V0))
        assert_allclose(actual, expected, rtol=1e-10)


class TestBatchedRvs:
    """Test rvs with tuple sizes and batched parameters."""

    def test_size_as_tuple(self):
        result = MatrixNormal.rvs(pt.constant(M0), pt.constant(U0), pt.constant(V0), size=(3, 5))
        samples = result.eval()
        assert samples.shape == (3, 5, 2, 2)

    def test_size_as_int(self):
        result = MatrixNormal.rvs(pt.constant(M0), pt.constant(U0), pt.constant(V0), size=10)
        samples = result.eval()
        assert samples.shape == (10, 2, 2)

    def test_size_none(self):
        result = MatrixNormal.rvs(pt.constant(M0), pt.constant(U0), pt.constant(V0), size=None)
        sample = result.eval()
        assert sample.shape == (2, 2)

    def test_batched_params(self):
        mus = np.stack([M0, M0 + 1])  # (2, 2, 2)
        result = MatrixNormal.rvs(pt.constant(mus), pt.constant(U0), pt.constant(V0), size=None)
        samples = result.eval()
        assert samples.shape == (2, 2, 2)

    def test_batched_params_with_size(self):
        mus = np.stack([M0, M0 + 1])  # (2, 2, 2)
        result = MatrixNormal.rvs(pt.constant(mus), pt.constant(U0), pt.constant(V0), size=(5,))
        samples = result.eval()
        assert samples.shape == (5, 2, 2, 2)


class TestBatchedMeanModeMedian:
    """Test mean/mode/median broadcasting with batched parameters."""

    def test_mean_broadcasts(self):
        rowcovs = np.array([np.eye(2), 2 * np.eye(2)])  # (2, 2, 2)
        result = MatrixNormal.mean(pt.constant(M0), pt.constant(rowcovs), pt.constant(V0))
        actual = result.eval()
        assert actual.shape == (2, 2, 2)
        for i in range(2):
            assert_allclose(actual[i], M0)

    def test_unbatched_mean(self):
        result = MatrixNormal.mean(pt.constant(M0), pt.constant(U0), pt.constant(V0))
        actual = result.eval()
        assert_allclose(actual, M0)
