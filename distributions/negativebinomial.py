import pytensor.tensor as pt
from pytensor.tensor.xlogx import xlogy0

from .helper import cdf_bounds, discrete_entropy
from .optimization import find_ppf_discrete


def mean(n, p):
    return n * (1 - p) / p


def mode(n, p):
    return pt.switch(pt.lt(n, 1), 0, pt.floor((n - 1) * (1 - p) / p))


def median(n, p):
    return ppf(0.5, n, p)


def var(n, p):
    return n * (1 - p) / (p * p)


def std(n, p):
    return pt.sqrt(var(n, p))


def skewness(n, p):
    return (2 - p) / pt.sqrt(n * (1 - p))


def kurtosis(n, p):
    return 6.0 / n + (p * p) / (n * (1 - p))


def entropy(n, p):
    lower = pt.cast(ppf(0.0001, n, p), "int32")
    upper = pt.cast(ppf(0.9999, n, p), "int32")
    return discrete_entropy(lower, upper, logpdf, n, p)


def pdf(x, n, p):
    return pt.exp(logpdf(x, n, p))


def cdf(x, n, p):
    return cdf_bounds(pt.betainc(n, x + 1, p), x, 0, pt.inf)


def ppf(q, n, p):
    params = (n, p)
    return find_ppf_discrete(q, 0, pt.inf, cdf, *params)


def sf(x, n, p):
    return 1.0 - cdf(x, n, p)


def isf(q, n, p):
    return ppf(1.0 - q, n, p)


def rvs(n, p, size=None, random_state=None):
    return pt.random.negative_binomial(n, p, rng=random_state, size=size)


def logpdf(x, n, p):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.gammaln(x + n) - pt.gammaln(n) - pt.gammaln(x + 1) + xlogy0(n, p) + xlogy0(x, 1 - p),
    )


def logcdf(x, n, p):
    return pt.log(cdf(x, n, p))


def logsf(x, n, p):
    return pt.log1p(-cdf(x, n, p))


def from_mu_alpha(mu, alpha):
    p = alpha / (mu + alpha)
    n = alpha
    return n, p


def to_mu_alpha(n, p):
    mu = n * (1 - p) / p
    alpha = n
    return mu, alpha
