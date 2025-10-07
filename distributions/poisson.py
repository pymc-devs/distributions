import pytensor.tensor as pt
from pytensor.tensor import gammaincinv

from .helper import cdf_bounds, discrete_entropy, ppf_bounds_disc


def mean(mu):
    return mu


def mode(mu):
    return pt.floor(mu)


def median(mu):
    return pt.floor(mu + 1.0 / 3.0 - 0.02 / mu)


def var(mu):
    return pt.as_tensor_variable(mu)


def std(mu):
    return pt.sqrt(mu)


def skewness(mu):
    return 1.0 / pt.sqrt(mu)


def kurtosis(mu):
    return 1.0 / mu


def entropy(mu):
    # Use explicit sum for small mu and Stirling's approximation for large mu
    lower = pt.cast(ppf(0.0001, mu), "int32")
    upper = pt.cast(ppf(0.9999, mu), "int32")
    return pt.switch(
        pt.lt(mu, 100),
        discrete_entropy(lower, upper, logpdf, mu),
        0.5 * pt.log(2.0 * pt.pi * pt.e * mu) - 1.0 / (12.0 * mu) - 1.0 / (24.0 * mu * mu),
    )


def pdf(x, mu):
    return pt.exp(logpdf(x, mu))


def cdf(x, mu):
    return cdf_bounds(pt.gammaincc(x + 1, mu), x, 0, pt.inf)


def ppf(q, mu):
    return ppf_bounds_disc(pt.round(gammaincinv(mu + 1, q)) - 1, q, 0, pt.inf)


def sf(x, mu):
    return 1.0 - cdf(x, mu)


def isf(q, mu):
    return ppf(1.0 - q, mu)


def rvs(mu, size=None, random_state=None):
    return pt.random.poisson(mu, rng=random_state, size=size)


def logpdf(x, mu):
    x = pt.as_tensor_variable(x)
    mu = pt.as_tensor_variable(mu)
    return pt.switch(pt.lt(x, 0), -pt.inf, x * pt.log(mu) - pt.gammaln(x + 1) - mu)


def logcdf(x, mu):
    return pt.log(cdf(x, mu))


def logsf(x, mu):
    return pt.log1p(-cdf(x, mu))
