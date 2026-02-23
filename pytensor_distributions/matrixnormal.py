"""Matrix Normal distribution."""

from functools import partial

import pytensor.tensor as pt
from pytensor.tensor.linalg import solve_triangular

from pytensor_distributions.mvnormal import _logdet_from_cholesky

solve_lower = partial(solve_triangular, lower=True)


def mean(mu, rowcov, colcov):
    return pt.as_tensor(mu)


def mode(mu, rowcov, colcov):
    return pt.as_tensor(mu)


def median(mu, rowcov, colcov):
    return pt.as_tensor(mu)


def var(mu, rowcov, colcov):
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)
    row_diag = pt.diagonal(rowcov, axis1=-2, axis2=-1)
    col_diag = pt.diagonal(colcov, axis1=-2, axis2=-1)
    return pt.outer(row_diag, col_diag)


def std(mu, rowcov, colcov):
    return pt.sqrt(var(mu, rowcov, colcov))


def skewness(mu, rowcov, colcov):
    return pt.zeros_like(pt.as_tensor(mu))


def kurtosis(mu, rowcov, colcov):
    return pt.zeros_like(pt.as_tensor(mu))


def entropy(mu, rowcov, colcov):
    mu = pt.as_tensor(mu)
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)
    m = mu.shape[-2]
    n = mu.shape[-1]
    mn = m * n
    _, logdet_U = pt.linalg.slogdet(rowcov)
    _, logdet_V = pt.linalg.slogdet(colcov)
    return 0.5 * mn * pt.log(2 * pt.pi * pt.e) + 0.5 * n * logdet_U + 0.5 * m * logdet_V


def logpdf(X, mu, rowcov, colcov):
    X = pt.as_tensor(X)
    mu = pt.as_tensor(mu)
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)

    m = mu.shape[-2]
    n = mu.shape[-1]
    mn = m * n

    chol_row = pt.linalg.cholesky(rowcov, lower=True)
    chol_col = pt.linalg.cholesky(colcov, lower=True)

    logdet_row, _ = _logdet_from_cholesky(chol_row)
    logdet_col, _ = _logdet_from_cholesky(chol_col)

    delta = X - mu

    # Compute tr[V^-1 (X-M)^T U^-1 (X-M)] via Cholesky solves
    # Using vec identity: quadform = ||L_U^-1 delta L_V^-T||^2_F
    Y = solve_lower(chol_row, delta)  # L_U^-1 delta, shape (m, n)
    Z = solve_lower(chol_col, Y.T)  # L_V^-1 (L_U^-1 delta)^T, shape (n, m)
    quadform = pt.sum(Z**2)

    log_norm = 0.5 * mn * pt.log(2 * pt.pi) + 0.5 * n * logdet_row + 0.5 * m * logdet_col

    return -0.5 * quadform - log_norm


def pdf(X, mu, rowcov, colcov):
    return pt.exp(logpdf(X, mu, rowcov, colcov))


def rvs(mu, rowcov, colcov, size=None, random_state=None):
    mu = pt.as_tensor(mu)
    rowcov = pt.as_tensor(rowcov)
    colcov = pt.as_tensor(colcov)

    m = mu.shape[-2]
    n = mu.shape[-1]

    L_row = pt.linalg.cholesky(rowcov, lower=True)
    L_col = pt.linalg.cholesky(colcov, lower=True)

    if size is None:
        Z = pt.random.normal(0, 1, size=(m, n), rng=random_state)
        return mu + L_row @ Z @ L_col.T
    else:
        Z = pt.random.normal(0, 1, size=(size, m, n), rng=random_state)
        return mu + L_row @ Z @ L_col.T
