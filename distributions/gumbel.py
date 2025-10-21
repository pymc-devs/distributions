import pytensor.tensor as pt

from distributions.helper import cdf_bounds, ppf_bounds_cont


def mean(mu, beta):
    return mu + beta * pt.euler_gamma


def mode(mu, beta):
    shape = pt.broadcast_arrays(mu, beta)[0]
    return pt.full_like(shape, mu)


def median(mu, beta):
    return mu - beta * -0.3665129


def var(mu, beta):
    shape = pt.broadcast_arrays(mu, beta)[0]
    return pt.full_like(shape, 1.64493407 * beta**2)


def std(mu, beta):
    shape = pt.broadcast_arrays(mu, beta)[0]
    return pt.full_like(shape, beta * 1.2825498)


def skewness(mu, beta):
    shape = pt.broadcast_arrays(mu, beta)[0]
    return pt.full_like(shape, 1.1395471)


def kurtosis(mu, beta):
    shape = pt.broadcast_arrays(mu, beta)[0]
    return pt.full_like(shape, 2.4)


def entropy(mu, beta):
    shape = pt.broadcast_arrays(mu, beta)[0]
    return pt.full_like(shape, pt.log(beta) + 1 + pt.euler_gamma)


def cdf(x, mu, beta):
    prob = pt.exp(-pt.exp(-(x - mu) / beta))
    return cdf_bounds(prob, x, -pt.inf, pt.inf)


def isf(x, mu, beta):
    return ppf(1 - x, mu, beta)


def pdf(x, mu, beta):
    return pt.exp(logpdf(x, mu, beta))


def ppf(q, mu, beta):
    x_val = mu - beta * pt.log(-pt.log(q))
    return ppf_bounds_cont(x_val, q, -pt.inf, pt.inf)


def sf(x, mu, beta):
    return pt.exp(logsf(x, mu, beta))


def rvs(mu, beta, size=None, random_state=None):
    return ppf(pt.random.uniform(0, 1, rng=random_state, size=size), mu, beta)


def logcdf(x, mu, beta):
    return -pt.exp(-(x - mu) / beta)


def logpdf(x, mu, beta):
    z = (x - mu) / beta
    return -z - pt.exp(-z) - pt.log(beta)


def logsf(x, mu, beta):
    return pt.log1p(-pt.exp(-pt.exp(-(x - mu) / beta)))
