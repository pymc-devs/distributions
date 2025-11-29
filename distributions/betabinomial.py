import pytensor.tensor as pt

from distributions.helper import cdf_bounds, discrete_entropy
from distributions.optimization import find_ppf_discrete


def mean(n, alpha, beta):
    return n * alpha / (alpha + beta)


def mode(n, alpha, beta):
    # The standard formula only applies when alpha > 1 and beta > 1
    # For other cases, the mode is at the boundaries or distribution is uniform
    standard_mode = pt.floor((n + 1) * ((alpha - 1) / (alpha + beta - 2)))

    # Handle different cases:
    # - alpha < 1 and beta < 1: U-shaped, modes at 0 and n (return 0)
    # - alpha <= 1 and beta > 1: mode at 0
    # - alpha > 1 and beta <= 1: mode at n
    # - alpha = beta = 1: uniform, any value valid (return floor(n/2))
    # - alpha > 1 and beta > 1: use standard formula

    result = pt.switch(
        pt.and_(pt.gt(alpha, 1), pt.gt(beta, 1)),
        # Standard case: alpha > 1 and beta > 1
        pt.clip(standard_mode, 0, n),
        pt.switch(
            pt.and_(pt.le(alpha, 1), pt.le(beta, 1)),
            # Both <= 1: U-shaped or uniform, mode at boundary (use 0)
            0,
            pt.switch(
                pt.le(alpha, 1),
                # alpha <= 1, beta > 1: mode at 0
                0,
                # alpha > 1, beta <= 1: mode at n
                n,
            ),
        ),
    )
    return result


def median(n, alpha, beta):
    return ppf(0.5, n, alpha, beta)


def var(n, alpha, beta):
    return n * alpha * beta * (alpha + beta + n) / ((alpha + beta) ** 2 * (alpha + beta + 1))


def std(n, alpha, beta):
    return pt.sqrt(var(n, alpha, beta))


def skewness(n, alpha, beta):
    return (
        (alpha + beta + 2 * n)
        * (beta - alpha)
        / (alpha + beta + 2)
        * pt.sqrt((1 + alpha + beta) / (n * alpha * beta * (alpha + beta + n)))
    )


def kurtosis(n, alpha, beta):
    alpha_beta_sum = alpha + beta
    alpha_beta_product = alpha * beta
    numerator = ((alpha_beta_sum) ** 2) * (1 + alpha_beta_sum)
    denominator = (
        (n * alpha_beta_product)
        * (alpha_beta_sum + 2)
        * (alpha_beta_sum + 3)
        * (alpha_beta_sum + n)
    )
    left = numerator / denominator
    right = (
        (alpha_beta_sum) * (alpha_beta_sum - 1 + 6 * n)
        + 3 * alpha_beta_product * (n - 2)
        + 6 * n**2
    )
    right -= (3 * alpha_beta_product * n * (6 - n)) / alpha_beta_sum
    right -= (18 * alpha_beta_product * n**2) / (alpha_beta_sum) ** 2
    return (left * right) - 3


def entropy(n, alpha, beta):
    return discrete_entropy(0, n + 1, logpdf, n, alpha, beta)


def pdf(x, n, alpha, beta):
    return pt.exp(logpdf(x, n, alpha, beta))


def cdf(x, n, alpha, beta):
    broadcast_shape = pt.broadcast_arrays(x, n, alpha, beta)[0]
    k_vals = pt.arange(0, pt.max(x) + 1)
    k_broadcast = k_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)

    prob = pt.sum(
        pt.where(pt.le(k_broadcast, pt.floor(x)), pdf(k_broadcast, n, alpha, beta), 0.0), axis=0
    )
    return cdf_bounds(prob, x, 0, n)


def ppf(q, n, alpha, beta):
    params = (n, alpha, beta)
    return find_ppf_discrete(q, 0, n, cdf, *params)


def sf(x, n, alpha, beta):
    return 1 - cdf(x, n, alpha, beta)


def isf(x, n, alpha, beta):
    return ppf(1 - x, n, alpha, beta)


def rvs(n, alpha, beta, size=None, random_state=None):
    return pt.random.betabinom(n, alpha, beta, size=size, rng=random_state)


def logcdf(x, n, alpha, beta):
    return pt.log(cdf(x, n, alpha, beta))


def logsf(x, n, alpha, beta):
    return pt.log(sf(x, n, alpha, beta))


def logpdf(x, n, alpha, beta):
    return pt.switch(
        pt.or_(pt.lt(x, 0), pt.gt(x, n)),
        -pt.inf,
        (
            pt.gammaln(n + 1)
            - pt.gammaln(x + 1)
            - pt.gammaln(n - x + 1)
            + pt.gammaln(x + alpha)
            + pt.gammaln(n - x + beta)
            - pt.gammaln(n + alpha + beta)
            + pt.gammaln(alpha + beta)
            - pt.gammaln(alpha)
            - pt.gammaln(beta)
        ),
    )
