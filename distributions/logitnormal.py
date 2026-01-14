import pytensor.tensor as pt

from distributions.helper import (
    cdf_bounds,
    continuous_entropy,
    continuous_kurtosis,
    continuous_mean,
    continuous_skewness,
    continuous_variance,
    ppf_bounds_cont,
)
from distributions.normal import ppf as normal_ppf

# Support bounds for logitnormal (open interval (0, 1))
_LOWER = 0.001
_UPPER = 0.999


def _logit(x):
    return pt.log(x) - pt.log1p(-x)


def _expit(y):
    return pt.sigmoid(y)


def mean(mu, sigma):
    return continuous_mean(_LOWER, _UPPER, logpdf, mu, sigma)


def mode(mu, sigma):
    return _expit(mu)


def median(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, _expit(mu))


def var(mu, sigma):
    return continuous_variance(_LOWER, _UPPER, logpdf, mu, sigma)


def std(mu, sigma):
    return pt.sqrt(var(mu, sigma))


def skewness(mu, sigma):
    return continuous_skewness(_LOWER, _UPPER, logpdf, mu, sigma)


def kurtosis(mu, sigma):
    return continuous_kurtosis(_LOWER, _UPPER, logpdf, mu, sigma)


def entropy(mu, sigma):
    return continuous_entropy(_LOWER, _UPPER, logpdf, mu, sigma)


def pdf(x, mu, sigma):
    return pt.exp(logpdf(x, mu, sigma))


def logpdf(x, mu, sigma):
    logit_x = _logit(x)
    return pt.switch(
        pt.or_(pt.le(x, 0), pt.ge(x, 1)),
        -pt.inf,
        -0.5 * ((logit_x - mu) / sigma) ** 2
        - pt.log(sigma)
        - 0.5 * pt.log(2 * pt.pi)
        - pt.log(x)
        - pt.log1p(-x),
    )


def cdf(x, mu, sigma):
    logit_x = _logit(x)
    prob = 0.5 * (1 + pt.erf((logit_x - mu) / (sigma * pt.sqrt(2))))
    return cdf_bounds(prob, x, 0, 1)


def logcdf(x, mu, sigma):
    logit_x = _logit(x)
    z = (logit_x - mu) / sigma
    return pt.switch(
        pt.le(x, 0),
        -pt.inf,
        pt.switch(
            pt.ge(x, 1),
            0.0,
            pt.switch(
                pt.lt(z, -1.0),
                pt.log(pt.erfcx(-z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
                pt.log1p(-pt.erfc(z / pt.sqrt(2.0)) / 2.0),
            ),
        ),
    )


def sf(x, mu, sigma):
    return pt.exp(logsf(x, mu, sigma))


def logsf(x, mu, sigma):
    logit_x = _logit(x)
    z = (logit_x - mu) / sigma
    return pt.switch(
        pt.le(x, 0),
        0.0,
        pt.switch(
            pt.ge(x, 1),
            -pt.inf,
            pt.switch(
                pt.gt(z, 1.0),
                pt.log(pt.erfcx(z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
                pt.log1p(-0.5 * (1 + pt.erf(z / pt.sqrt(2.0)))),
            ),
        ),
    )


def ppf(q, mu, sigma):
    return ppf_bounds_cont(_expit(normal_ppf(q, mu, sigma)), q, 0, 1)


def isf(q, mu, sigma):
    return ppf(1 - q, mu, sigma)


def rvs(mu, sigma, size=None, random_state=None):
    return _expit(pt.random.normal(mu, sigma, rng=random_state, size=size))
