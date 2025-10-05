import pytensor.tensor as pt
from pytensor.tensor.xlogx import xlogx

from .helper import ppf_bounds_disc, cdf_bounds


def mean(p):
    return p


def mode(p):
    return median(p)


def median(p):
    return pt.switch(pt.le(p, 0.5), 0, 1)


def var(p):
    return p * (1 - p)


def std(p):
    return pt.sqrt(var(p))


def skewness(p):
    q = 1 - p
    return (q - p) / pt.sqrt(p * q)


def kurtosis(p):
    return (1 - 6 * p * (1 - p)) / (p * (1 - p))


def entropy(p):
    q = 1 - p
    return -xlogx(p) - xlogx(q)


def rvs(p, size=None, random_state=None):
    return pt.random.binomial(1, p, size=size, rng=random_state)


def cdf(x, p):
    x = pt.as_tensor_variable(x)
    return cdf_bounds(pt.switch(pt.lt(x, 1), 1 - p, 1.0), x, 0, 1)


def ppf(q, p):
    q = pt.as_tensor_variable(q)
    x_val = pt.switch(pt.lt(q, 1 - p), 0, 1)
    return ppf_bounds_disc(x_val, q, 0, 1)


def sf(x, p):
    return pt.switch(pt.lt(x, 0), 1.0, pt.switch(pt.lt(x, 1), p, 0.0))


def isf(q, p):
    return ppf(1 - q, p)


def pdf(x, p):
    x = pt.as_tensor_variable(x)
    return pt.switch(pt.eq(x, 1), p, pt.switch(pt.eq(x, 0), 1 - p, 0.0))


def logpdf(x, p):
    x = pt.as_tensor_variable(x)
    return pt.switch(pt.eq(x, 1), pt.log(p), pt.switch(pt.eq(x, 0), pt.log(1 - p), -pt.inf))


def logcdf(x, p):
    x = pt.as_tensor_variable(x)
    return pt.switch(pt.lt(x, 0), -pt.inf, pt.switch(pt.lt(x, 1), pt.log(1 - p), 0.0))


def logsf(x, mu):
    return pt.log1p(-cdf(x, mu))
