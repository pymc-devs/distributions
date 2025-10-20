import pytensor.tensor as pt

from .helper import logdiffexp, continuous_entropy
from .optimization import find_ppf
from .normal import logcdf as normal_logcdf
from .normal import logpdf as normal_logpdf


def mean(mu, sigma, nu):
    return mu + nu


def mode(mu, sigma, nu):
    tau = 1 / nu
    return (
        mu
        - pt.sign(tau) * pt.sqrt(2) * sigma * pt.erfcinv(pt.abs(tau) / sigma * pt.sqrt(2 / pt.pi))
        + sigma**2 / tau
    )


def median(mu, sigma, nu):
    return ppf(0.5, mu, sigma, nu)


def var(mu, sigma, nu):
    return pt.square(sigma) + pt.square(nu)


def std(mu, sigma, nu):
    return pt.sqrt(pt.square(sigma) + pt.square(nu))


def skewness(mu, sigma, nu):
    nus2 = pt.square(nu / sigma)
    opnus2 = 1.0 + nus2
    return 2 * pt.pow(nu / sigma, 3) * pt.pow(opnus2, -1.5)


def kurtosis(mu, sigma, nu):
    nus2 = pt.square(nu / sigma)
    opnus2 = 1.0 + nus2
    return 6.0 * pt.square(nus2) * pt.pow(opnus2, -2)


def entropy(mu, sigma, nu):
    min_x = ppf(0.0001, mu, sigma, nu)
    max_x = ppf(0.9999, mu, sigma, nu)
    params = (mu, sigma, nu)
    return continuous_entropy(min_x, max_x, logpdf, *params)


def cdf(x, mu, sigma, nu):
    return pt.exp(logcdf(x, mu, sigma, nu))


def isf(x, mu, sigma, nu):
    return ppf(1 - x, mu, sigma, nu)


def pdf(x, mu, sigma, nu):
    return pt.exp(logpdf(x, mu, sigma, nu))


def ppf(q, mu, sigma, nu):
    return find_ppf(q, -pt.inf, pt.inf, cdf, mu, sigma, nu)


def sf(x, mu, sigma, nu):
    return 1.0 - cdf(x, mu, sigma, nu)


def rvs(mu, sigma, nu, size=None, random_state=None):
    next_rng, exp_rvs = pt.random.exponential(nu, rng=random_state, size=size).owner.outputs
    normal_rvs = pt.random.normal(mu, sigma, rng=next_rng, size=size)
    return normal_rvs + exp_rvs


def logcdf(x, mu, sigma, nu):
    return pt.switch(
        pt.eq(x, -pt.inf),
        -pt.inf,
        pt.switch(
            pt.eq(x, pt.inf),
            0.0,
            pt.switch(
                pt.ge(nu, 0.05 * sigma),
                logdiffexp(
                    normal_logcdf(x, mu, sigma),
                    (
                        (mu - x) / nu
                        + 0.5 * (sigma / nu) ** 2
                        + normal_logcdf(x - (mu + (sigma**2) / nu), 0.0, sigma)
                    ),
                ),
                normal_logcdf(x, mu, sigma),
            ),
        ),
    )


def logpdf(x, mu, sigma, nu):
    return pt.switch(
        pt.ge(nu, 0.05 * sigma),
        (
            -pt.log(nu)
            + (mu - x) / nu
            + 0.5 * (sigma / nu) ** 2
            + normal_logcdf(x - (mu + (sigma**2) / nu), 0.0, sigma)
        ),
        normal_logpdf(x, mu, sigma),
    )


def logsf(x, mu, sigma, nu):
    return pt.log1p(-pt.exp(logcdf(x, mu, sigma, nu)))
