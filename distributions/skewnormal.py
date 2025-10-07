import pytensor.tensor as pt

from .optimization import find_ppf
from .halfnormal import entropy as halfnormal_entropy
from .normal import entropy as normal_entropy


def mean(mu, sigma, alpha):
    mu_b, sigma_b, alpha_b = pt.broadcast_arrays(mu, sigma, alpha)
    return mu_b + sigma_b * pt.sqrt(2 / pt.pi) * alpha_b / pt.sqrt(1 + alpha_b**2)


def mode(mu, sigma, alpha):
    # For skew normal, the mode is approximately mu + sigma * delta * sqrt(2/pi)
    # where delta = alpha / sqrt(1 + alpha^2), but this is an approximation
    mu_b, sigma_b, alpha_b = pt.broadcast_arrays(mu, sigma, alpha)
    delta = alpha_b / pt.sqrt(1 + alpha_b**2)
    return mu_b + sigma_b * delta * pt.sqrt(2 / pt.pi)


def median(mu, sigma, alpha):
    return ppf(0.5, mu, sigma, alpha)


def var(mu, sigma, alpha):
    _, sigma_b, alpha_b = pt.broadcast_arrays(mu, sigma, alpha)
    delta = alpha_b / pt.sqrt(1 + alpha_b**2)
    return sigma_b**2 * (1 - 2 * delta**2 / pt.pi)


def std(mu, sigma, alpha):
    return pt.sqrt(var(mu, sigma, alpha))


def skewness(mu, sigma, alpha):
    delta = alpha / pt.sqrt(1 + alpha**2)
    mean_z = pt.sqrt(2 / pt.pi) * delta
    return ((4 - pt.pi) / 2) * (mean_z**3 / (1 - mean_z**2) ** (3 / 2))


def kurtosis(mu, sigma, alpha):
    _, _, alpha_b = pt.broadcast_arrays(mu, sigma, alpha)
    delta = alpha_b / pt.sqrt(1 + alpha_b**2)
    mean_z = delta * pt.sqrt(2 / pt.pi)
    var_z = 1 - 2 * delta**2 / pt.pi
    return 2 * (pt.pi - 3) * (mean_z**4 / var_z**2)


def entropy(mu, sigma, alpha):
    # The entropy of the skew normal distribution does not have a simple closed form
    # we use a linear combination of the normal and half-normal entropies as approximation
    H_normal = normal_entropy(mu, sigma)
    H_halfn = halfnormal_entropy(sigma)

    factor = 1 / pt.sqrt(1 + alpha**2)
    return H_normal * factor + H_halfn * (1 - factor)


def cdf(x, mu, sigma, alpha):
    return 0.5 * (1 + pt.erf((x - mu) / (sigma * 2**0.5))) - 2 * pt.owens_t((x - mu) / sigma, alpha)


def logcdf(x, mu, sigma, alpha):
    return pt.log(cdf(x, mu, sigma, alpha))


def isf(x, mu, sigma, alpha):
    return ppf(1 - x, mu, sigma, alpha)


def pdf(x, mu, sigma, alpha):
    return pt.exp(logpdf(x, mu, sigma, alpha))


def ppf(q, mu, sigma, alpha):
    mu_b, sigma_b, alpha_b = pt.broadcast_arrays(mu, sigma, alpha)
    params = (mu_b, sigma_b, alpha_b)
    return find_ppf(q, -pt.inf, pt.inf, cdf, *params)


def sf(x, mu, sigma, alpha):
    return 1 - cdf(x, mu, sigma, alpha)


def rvs(mu, sigma, alpha, size=None, random_state=None):
    mu_b, sigma_b, alpha_b = pt.broadcast_arrays(mu, sigma, alpha)
    next_rng, u_0 = pt.random.normal(0, 1, rng=random_state, size=size).owner.outputs
    v = pt.random.normal(0, 1, rng=next_rng, size=size)
    d = alpha_b / pt.sqrt(1 + alpha_b**2)
    u_1 = d * u_0 + v * pt.sqrt(1 - d**2)
    return pt.sign(u_0) * u_1 * sigma_b + mu_b


def logpdf(x, mu, sigma, alpha):
    tau = 1.0 / (sigma**2)
    return (
        pt.log(1 + pt.erf(((x - mu) * pt.sqrt(tau) * alpha) / pt.sqrt(2)))
        + (-tau * (x - mu) ** 2 + pt.log(tau / pt.pi / 2.0)) / 2.0
    )


def logsf(x, mu, sigma, alpha):
    return pt.log1mexp(logcdf(x, mu, sigma, alpha))
