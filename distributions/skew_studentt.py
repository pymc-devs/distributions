import pytensor.tensor as pt

from distributions.helper import continuous_entropy, continuous_mode, ppf_bounds_cont
from distributions.optimization import find_ppf


def _t_logpdf(z, nu):
    return (
        pt.gammaln((nu + 1) / 2)
        - pt.gammaln(nu / 2)
        - 0.5 * pt.log(nu * pt.pi)
        - 0.5 * (nu + 1) * pt.log1p(z**2 / nu)
    )


def _t_cdf(z, nu):
    factor = 0.5 * pt.betainc(0.5 * nu, 0.5, nu / (z**2 + nu))
    return pt.switch(pt.lt(z, 0), factor, 1 - factor)


def _t_logcdf(z, nu):
    cdf_val = _t_cdf(z, nu)
    return pt.switch(pt.le(cdf_val, 0), -pt.inf, pt.log(cdf_val))


def _delta(alpha):
    return alpha / pt.sqrt(1 + alpha**2)


def mean(nu, alpha, mu, sigma):
    nu_b, alpha_b, mu_b, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    d = _delta(alpha_b)
    b_nu = pt.sqrt(nu_b / pt.pi) * pt.exp(pt.gammaln((nu_b - 1) / 2) - pt.gammaln(nu_b / 2))
    result = mu_b + sigma_b * b_nu * d
    return pt.switch(nu_b > 1, result, pt.nan)


def mode(nu, alpha, mu, sigma):
    nu_b, alpha_b, mu_b, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    # The mode is between mu and the mean. Use a generous range around mu.
    std_approx = sigma_b * pt.sqrt(nu_b / pt.switch(nu_b > 2, nu_b - 2, 1.0))
    lower = mu_b - 5 * std_approx
    upper = mu_b + 5 * std_approx
    return continuous_mode(lower, upper, logpdf, nu_b, alpha_b, mu_b, sigma_b)


def median(nu, alpha, mu, sigma):
    return ppf(0.5, nu, alpha, mu, sigma)


def var(nu, alpha, mu, sigma):
    nu_b, alpha_b, mu_b, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    d = _delta(alpha_b)
    b_nu = pt.sqrt(nu_b / pt.pi) * pt.exp(pt.gammaln((nu_b - 1) / 2) - pt.gammaln(nu_b / 2))
    result = sigma_b**2 * (nu_b / (nu_b - 2) - (b_nu * d) ** 2)
    return pt.switch(nu_b > 2, result, pt.switch(nu_b > 1, pt.inf, pt.nan))


def std(nu, alpha, mu, sigma):
    variance = var(nu, alpha, mu, sigma)
    return pt.switch(pt.isinf(variance) | pt.isnan(variance), variance, pt.sqrt(variance))


def skewness(nu, alpha, mu, sigma):
    nu_b, alpha_b, _, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    d = _delta(alpha_b)
    b_nu = pt.sqrt(nu_b / pt.pi) * pt.exp(pt.gammaln((nu_b - 1) / 2) - pt.gammaln(nu_b / 2))
    mu_z = b_nu * d
    sigma_z_sq = nu_b / (nu_b - 2) - mu_z**2

    m3_factor = (
        b_nu * d * (nu_b * (3 - d**2) / (nu_b - 3) - 3 * nu_b / (nu_b - 2) + 2 * (b_nu * d) ** 2)
    )

    result = m3_factor / (sigma_z_sq ** (3 / 2))
    return pt.switch(nu_b > 3, result, pt.nan)


def kurtosis(nu, alpha, mu, sigma):
    nu_b, alpha_b, _, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    d = _delta(alpha_b)
    b_nu = pt.sqrt(nu_b / pt.pi) * pt.exp(pt.gammaln((nu_b - 1) / 2) - pt.gammaln(nu_b / 2))
    mu_z = b_nu * d
    sigma_z_sq = nu_b / (nu_b - 2) - mu_z**2

    m4_raw = 3 * nu_b**2 / ((nu_b - 2) * (nu_b - 4))

    m4_factor = (
        m4_raw
        - 4 * mu_z * skewness(nu_b, alpha_b, 0.0, 1.0) * (sigma_z_sq ** (3 / 2))
        - 6 * mu_z**2 * sigma_z_sq
        - mu_z**4
    )

    excess_kurt = m4_factor / (sigma_z_sq**2) - 3
    return pt.switch(nu_b > 4, excess_kurt, pt.nan)


def entropy(nu, alpha, mu, sigma):
    nu_b, alpha_b, mu_b, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    std_approx = sigma_b * pt.sqrt(nu_b / pt.switch(nu_b > 2, nu_b - 2, 1.0))
    lower = mu_b - 8 * std_approx
    upper = mu_b + 8 * std_approx
    return continuous_entropy(lower, upper, logpdf, nu, alpha, mu, sigma)


def pdf(x, nu, alpha, mu, sigma):
    return pt.exp(logpdf(x, nu, alpha, mu, sigma))


def logpdf(x, nu, alpha, mu, sigma):
    z = (x - mu) / sigma
    skew_arg = alpha * z * pt.sqrt((nu + 1) / (nu + z**2))
    result = pt.log(2) - pt.log(sigma) + _t_logpdf(z, nu) + _t_logcdf(skew_arg, nu + 1)
    return result


def _cdf_standardized(z, nu, alpha):
    n_points = 50
    broadcast_shape = pt.broadcast_arrays(z, nu, alpha)[0]

    std_approx = pt.sqrt(nu / pt.switch(pt.gt(nu, 2), nu - 2, 1.0))
    lower = -6 * std_approx

    t_vals = pt.linspace(0.0, 1.0, n_points)
    if t_vals.ndim == 1 and broadcast_shape.ndim > 0:
        t_broadcast = t_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)
    else:
        t_broadcast = t_vals

    z_vals = lower + t_broadcast * (z - lower)
    z_std = z_vals
    skew_arg = alpha * z_std * pt.sqrt((nu + 1) / (nu + z_std**2))
    pdf_vals = 2 * pt.exp(_t_logpdf(z_std, nu)) * _t_cdf(skew_arg, nu + 1)

    dz = (z - lower) / (n_points - 1)
    integral = dz * (0.5 * pdf_vals[0] + pt.sum(pdf_vals[1:-1], axis=0) + 0.5 * pdf_vals[-1])

    return pt.clip(integral, 0.0, 1.0)


def cdf(x, nu, alpha, mu, sigma):
    z = (x - mu) / sigma
    result = _cdf_standardized(z, nu, alpha)
    result = pt.switch(pt.eq(x, -pt.inf), 0.0, result)
    result = pt.switch(pt.eq(x, pt.inf), 1.0, result)
    return result


def logcdf(x, nu, alpha, mu, sigma):
    return pt.log(cdf(x, nu, alpha, mu, sigma))


def sf(x, nu, alpha, mu, sigma):
    return 1 - cdf(x, nu, alpha, mu, sigma)


def logsf(x, nu, alpha, mu, sigma):
    return pt.log(sf(x, nu, alpha, mu, sigma))


def ppf(q, nu, alpha, mu, sigma):
    nu_b, alpha_b, mu_b, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    z_ppf = find_ppf(q, -pt.inf, pt.inf, _cdf_standardized, nu_b, alpha_b)
    return ppf_bounds_cont(mu_b + sigma_b * z_ppf, q, -pt.inf, pt.inf)


def isf(q, nu, alpha, mu, sigma):
    return ppf(1 - q, nu, alpha, mu, sigma)


def rvs(nu, alpha, mu, sigma, size=None, random_state=None):
    nu_b, alpha_b, mu_b, sigma_b = pt.broadcast_arrays(nu, alpha, mu, sigma)
    d = _delta(alpha_b)

    next_rng, u0 = pt.random.normal(0, 1, rng=random_state, size=size).owner.outputs
    next_rng2, u1 = pt.random.normal(0, 1, rng=next_rng, size=size).owner.outputs
    chi_sq = pt.random.chisquare(nu_b, rng=next_rng2, size=size)

    z_sn = d * pt.abs(u0) + pt.sqrt(1 - d**2) * u1
    z_st = z_sn / pt.sqrt(chi_sq / nu_b)

    return mu_b + sigma_b * z_st
