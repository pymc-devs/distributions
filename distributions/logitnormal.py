import pytensor.tensor as pt

from distributions.helper import cdf_bounds, continuous_entropy, ppf_bounds_cont


def _logit(x):
    return pt.log(x) - pt.log1p(-x)


def _expit(y):
    return pt.sigmoid(y)


def mean(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]

    x_values = pt.linspace(0.001, 0.999, 1000)
    x_broadcast = x_values.reshape((-1,) + (1,) * shape.ndim)

    pdf_vals = pt.exp(logpdf(x_broadcast, mu, sigma))
    integrand = x_broadcast * pdf_vals

    dx = 0.998 / 999
    result = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1], axis=0) + 0.5 * integrand[-1])

    return pt.squeeze(result) if shape.ndim == 0 else result


def mode(mu, sigma):
    return _expit(mu)


def median(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]
    return pt.full_like(shape, _expit(mu))


def var(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]

    x_values = pt.linspace(0.001, 0.999, 1000)
    x_broadcast = x_values.reshape((-1,) + (1,) * shape.ndim)

    pdf_vals = pt.exp(logpdf(x_broadcast, mu, sigma))

    integrand_mean = x_broadcast * pdf_vals
    dx = 0.998 / 999
    mean_val = dx * (
        0.5 * integrand_mean[0] + pt.sum(integrand_mean[1:-1], axis=0) + 0.5 * integrand_mean[-1]
    )

    integrand_x2 = x_broadcast**2 * pdf_vals
    mean_x2 = dx * (
        0.5 * integrand_x2[0] + pt.sum(integrand_x2[1:-1], axis=0) + 0.5 * integrand_x2[-1]
    )

    result = mean_x2 - mean_val**2
    return pt.squeeze(result) if shape.ndim == 0 else result


def std(mu, sigma):
    return pt.sqrt(var(mu, sigma))


def skewness(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]

    x_values = pt.linspace(0.001, 0.999, 1000)
    x_broadcast = x_values.reshape((-1,) + (1,) * shape.ndim)
    pdf_vals = pt.exp(logpdf(x_broadcast, mu, sigma))
    dx = 0.998 / 999

    integrand_mean = x_broadcast * pdf_vals
    mean_val = dx * (
        0.5 * integrand_mean[0] + pt.sum(integrand_mean[1:-1], axis=0) + 0.5 * integrand_mean[-1]
    )

    integrand_x2 = x_broadcast**2 * pdf_vals
    mean_x2 = dx * (
        0.5 * integrand_x2[0] + pt.sum(integrand_x2[1:-1], axis=0) + 0.5 * integrand_x2[-1]
    )

    integrand_x3 = x_broadcast**3 * pdf_vals
    mean_x3 = dx * (
        0.5 * integrand_x3[0] + pt.sum(integrand_x3[1:-1], axis=0) + 0.5 * integrand_x3[-1]
    )

    variance = mean_x2 - mean_val**2
    std_val = pt.sqrt(variance)
    third_central = mean_x3 - 3 * mean_val * mean_x2 + 2 * mean_val**3

    result = third_central / (std_val**3)
    return pt.squeeze(result) if shape.ndim == 0 else result


def kurtosis(mu, sigma):
    shape = pt.broadcast_arrays(mu, sigma)[0]

    x_values = pt.linspace(0.001, 0.999, 1000)
    x_broadcast = x_values.reshape((-1,) + (1,) * shape.ndim)
    pdf_vals = pt.exp(logpdf(x_broadcast, mu, sigma))
    dx = 0.998 / 999

    integrand_mean = x_broadcast * pdf_vals
    mean_val = dx * (
        0.5 * integrand_mean[0] + pt.sum(integrand_mean[1:-1], axis=0) + 0.5 * integrand_mean[-1]
    )

    integrand_x2 = x_broadcast**2 * pdf_vals
    mean_x2 = dx * (
        0.5 * integrand_x2[0] + pt.sum(integrand_x2[1:-1], axis=0) + 0.5 * integrand_x2[-1]
    )

    integrand_x3 = x_broadcast**3 * pdf_vals
    mean_x3 = dx * (
        0.5 * integrand_x3[0] + pt.sum(integrand_x3[1:-1], axis=0) + 0.5 * integrand_x3[-1]
    )

    integrand_x4 = x_broadcast**4 * pdf_vals
    mean_x4 = dx * (
        0.5 * integrand_x4[0] + pt.sum(integrand_x4[1:-1], axis=0) + 0.5 * integrand_x4[-1]
    )

    variance = mean_x2 - mean_val**2
    fourth_central = mean_x4 - 4 * mean_val * mean_x3 + 6 * mean_val**2 * mean_x2 - 3 * mean_val**4

    result = fourth_central / (variance**2) - 3
    return pt.squeeze(result) if shape.ndim == 0 else result


def entropy(mu, sigma):
    return continuous_entropy(0.001, 0.999, logpdf, mu, sigma)


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
    normal_ppf = mu + sigma * pt.sqrt(2) * pt.erfinv(2 * q - 1)
    return ppf_bounds_cont(_expit(normal_ppf), q, 0, 1)


def isf(q, mu, sigma):
    return ppf(1 - q, mu, sigma)


def rvs(mu, sigma, size=None, random_state=None):
    return _expit(pt.random.normal(mu, sigma, rng=random_state, size=size))
