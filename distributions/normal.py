import numpy as np
import pytensor.tensor as pt

from .helper import ppf_bounds_cont


def mean(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, mu)

def mode(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, mu)

def median(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, mu)

def var(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, pt.square(sigma))

def std(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, sigma)

def skewness(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, 0)

def kurtosis(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, 0)

def entropy(mu, sigma):
    shape = pt.broadcast_shape(mu, sigma)
    return pt.full(shape, 0.5 * (pt.log(2 * pt.pi * pt.e * sigma**2)))

def cdf(x, mu, sigma):
    return 0.5 * (1 + pt.erf((x - mu) / (sigma * 2**0.5)))

def isf(x, mu, sigma):
    return ppf(1 - x, mu, sigma)

def pdf(x, mu, sigma):
    return 1 / pt.sqrt(2 * pt.pi * sigma**2) * pt.exp(-0.5 * ((x - mu) / sigma) ** 2)

def ppf(q, mu, sigma):
    return ppf_bounds_cont(mu + sigma * 2**0.5 * pt.erfinv(2 * q - 1), q, -pt.inf, pt.inf)

def sf(x, mu, sigma):
    return pt.exp(logsf(x, mu, sigma))


def rvs(mu, sigma, rng=None, size=None):
    if size is None:
        size = pt.broadcast_shape(mu, sigma)
    return mu + pt.random.normal(rng=rng, size=size) * sigma

def logcdf(x, mu, sigma):
    z = (x - mu) / sigma
    return pt.switch(
        pt.lt(z, -1.0),
        pt.log(pt.erfcx(-z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
        pt.log1p(-pt.erfc(z / pt.sqrt(2.0)) / 2.0),
    )

def logpdf(x, mu, sigma):
    return -0.5 * pt.pow((x - mu) / sigma, 2) - pt.log(pt.sqrt(2.0 * np.pi)) - pt.log(sigma)

def logsf(x, mu, sigma):
    return logcdf(-x, -mu, sigma)

def neg_logpdf(x, mu, sigma):
    return -(logpdf(x, mu, sigma)).sum()

