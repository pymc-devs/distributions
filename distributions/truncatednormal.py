import pytensor.tensor as pt

from distributions import normal as Normal
from distributions.helper import ppf_bounds_cont


def _phi(z):
    return 0.5 * (1 + pt.erf(z / pt.sqrt(2.0)))


def _Phi_inv(p):
    return pt.sqrt(2.0) * pt.erfinv(2 * p - 1)


def _pdf_standard(z):
    return pt.exp(-0.5 * z**2) / pt.sqrt(2 * pt.pi)


def _alpha_beta(mu, sigma, lower, upper):
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    return alpha, beta


def _Z(alpha, beta):
    return _phi(beta) - _phi(alpha)


def mean(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)
    return mu + sigma * (_pdf_standard(alpha) - _pdf_standard(beta)) / Z


def mode(mu, sigma, lower, upper):
    return pt.clip(mu, lower, upper)


def median(mu, sigma, lower, upper):
    return ppf(0.5, mu, sigma, lower, upper)


def var(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)
    phi_alpha = _pdf_standard(alpha)
    phi_beta = _pdf_standard(beta)

    term1 = (alpha * phi_alpha - beta * phi_beta) / Z
    term2 = ((phi_alpha - phi_beta) / Z) ** 2

    return sigma**2 * (1 + term1 - term2)


def std(mu, sigma, lower, upper):
    return pt.sqrt(var(mu, sigma, lower, upper))


def _raw_moments(alpha, beta):
    Z = _Z(alpha, beta)
    phi_a = _pdf_standard(alpha)
    phi_b = _pdf_standard(beta)

    m1 = (phi_a - phi_b) / Z
    m2 = 1 + (alpha * phi_a - beta * phi_b) / Z
    m3 = 2 * m1 + (alpha**2 * phi_a - beta**2 * phi_b) / Z
    m4 = 3 + 3 * (alpha * phi_a - beta * phi_b) / Z + (alpha**3 * phi_a - beta**3 * phi_b) / Z

    return m1, m2, m3, m4


def skewness(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    m1, m2, m3, _ = _raw_moments(alpha, beta)

    mu2 = m2 - m1**2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1**3

    return mu3 / pt.pow(mu2, 1.5)


def kurtosis(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    m1, m2, m3, m4 = _raw_moments(alpha, beta)

    mu2 = m2 - m1**2
    mu4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4

    return mu4 / mu2**2 - 3


def entropy(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)
    phi_alpha = _pdf_standard(alpha)
    phi_beta = _pdf_standard(beta)

    return pt.log(pt.sqrt(2 * pt.pi * pt.e) * sigma * Z) + (alpha * phi_alpha - beta * phi_beta) / (
        2 * Z
    )


def pdf(x, mu, sigma, lower, upper):
    return pt.exp(logpdf(x, mu, sigma, lower, upper))


def logpdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)

    base_logpdf = Normal.logpdf(x, mu, sigma)
    result = base_logpdf - pt.log(Z)

    return pt.switch(
        pt.or_(pt.lt(x, lower), pt.gt(x, upper)),
        -pt.inf,
        result,
    )


def cdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)
    xi = (x - mu) / sigma

    result = (_phi(xi) - _phi(alpha)) / Z

    return pt.switch(
        pt.lt(x, lower),
        0.0,
        pt.switch(pt.gt(x, upper), 1.0, result),
    )


def logcdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)
    xi = (x - mu) / sigma

    result = pt.log(_phi(xi) - _phi(alpha)) - pt.log(Z)

    return pt.switch(
        pt.le(x, lower),
        -pt.inf,
        pt.switch(pt.ge(x, upper), 0.0, result),
    )


def sf(x, mu, sigma, lower, upper):
    return 1.0 - cdf(x, mu, sigma, lower, upper)


def logsf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)
    xi = (x - mu) / sigma

    result = pt.log(_phi(beta) - _phi(xi)) - pt.log(Z)

    return pt.switch(
        pt.le(x, lower),
        0.0,
        pt.switch(pt.ge(x, upper), -pt.inf, result),
    )


def ppf(q, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    Z = _Z(alpha, beta)

    result = mu + sigma * _Phi_inv(q * Z + _phi(alpha))

    return ppf_bounds_cont(result, q, lower, upper)


def isf(q, mu, sigma, lower, upper):
    return ppf(1.0 - q, mu, sigma, lower, upper)


def rvs(mu, sigma, lower, upper, size=None, random_state=None):
    u = pt.random.uniform(0, 1, size=size, rng=random_state)
    return ppf(u, mu, sigma, lower, upper)
