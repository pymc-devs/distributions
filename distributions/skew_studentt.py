"""
Jones-Faddy Skew Student-t Distribution.

This implements the skewed Student's t distribution from Jones and Faddy (2003),
matching the parameterization used in PyMC and scipy.stats.jf_skew_t.

References
----------
Jones, M.C. and Faddy, M.J. (2003). "A skew extension of the t distribution,
with applications." Journal of the Royal Statistical Society, Series B, 65(1), 159-174.
"""

import pytensor.tensor as pt
from pytensor.tensor.math import betaincinv
from pytensor.tensor.special import betaln

from distributions.helper import continuous_entropy, continuous_mode, ppf_bounds_cont


def _raw_moment(n, a, b):
    """
    Compute the n-th raw moment of the standardized Jones-Faddy skew-t distribution.

    Uses the formula from scipy:
    E[T^n] = (a + b)^(n/2) / (2^n * B(a, b)) * sum_{k=0}^{n} C(n,k) * (-1)^k * B(a + n/2 - k, b - n/2 + k)

    The moment exists when a > n/2 and b > n/2.
    """
    # Compute in log space for numerical stability
    log_coeff = 0.5 * n * pt.log(a + b) - n * pt.log(2) - betaln(a, b)

    # Sum over k from 0 to n
    # For small n (1-4), we can expand this explicitly
    if n == 1:
        # k=0: C(1,0)*(-1)^0*B(a+0.5, b-0.5) = B(a+0.5, b-0.5)
        # k=1: C(1,1)*(-1)^1*B(a-0.5, b+0.5) = -B(a-0.5, b+0.5)
        term0 = pt.exp(betaln(a + 0.5, b - 0.5))
        term1 = pt.exp(betaln(a - 0.5, b + 0.5))
        sum_terms = term0 - term1
    elif n == 2:
        # k=0: B(a+1, b-1)
        # k=1: -2*B(a, b)
        # k=2: B(a-1, b+1)
        term0 = pt.exp(betaln(a + 1, b - 1))
        term1 = 2 * pt.exp(betaln(a, b))
        term2 = pt.exp(betaln(a - 1, b + 1))
        sum_terms = term0 - term1 + term2
    elif n == 3:
        # k=0: B(a+1.5, b-1.5)
        # k=1: -3*B(a+0.5, b-0.5)
        # k=2: 3*B(a-0.5, b+0.5)
        # k=3: -B(a-1.5, b+1.5)
        term0 = pt.exp(betaln(a + 1.5, b - 1.5))
        term1 = 3 * pt.exp(betaln(a + 0.5, b - 0.5))
        term2 = 3 * pt.exp(betaln(a - 0.5, b + 0.5))
        term3 = pt.exp(betaln(a - 1.5, b + 1.5))
        sum_terms = term0 - term1 + term2 - term3
    elif n == 4:
        # k=0: B(a+2, b-2)
        # k=1: -4*B(a+1, b-1)
        # k=2: 6*B(a, b)
        # k=3: -4*B(a-1, b+1)
        # k=4: B(a-2, b+2)
        term0 = pt.exp(betaln(a + 2, b - 2))
        term1 = 4 * pt.exp(betaln(a + 1, b - 1))
        term2 = 6 * pt.exp(betaln(a, b))
        term3 = 4 * pt.exp(betaln(a - 1, b + 1))
        term4 = pt.exp(betaln(a - 2, b + 2))
        sum_terms = term0 - term1 + term2 - term3 + term4
    else:
        raise ValueError(f"Moment order {n} not supported")

    return pt.exp(log_coeff) * sum_terms


def mean(a, b, mu, sigma):
    """
    Mean of the Jones-Faddy skew-t distribution.

    The mean exists when a > 0.5 and b > 0.5.
    """
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)

    mean_std = _raw_moment(1, a_b, b_b)
    result = mu_b + sigma_b * mean_std

    return pt.switch((a_b > 0.5) & (b_b > 0.5), result, pt.nan)


def mode(a, b, mu, sigma):
    """
    Mode of the Jones-Faddy skew-t distribution.

    Found numerically by maximizing the PDF.
    """
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    std_approx = sigma_b * pt.sqrt((a_b + b_b) / (2 * a_b * b_b))
    lower = mu_b - 5 * std_approx
    upper = mu_b + 5 * std_approx
    return continuous_mode(lower, upper, logpdf, a_b, b_b, mu_b, sigma_b)


def median(a, b, mu, sigma):
    """Median of the Jones-Faddy skew-t distribution."""
    return ppf(0.5, a, b, mu, sigma)


def var(a, b, mu, sigma):
    """
    Variance of the Jones-Faddy skew-t distribution.

    The variance exists when a > 1 and b > 1.
    """
    a_b, b_b, _, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)

    e_z = _raw_moment(1, a_b, b_b)
    e_z2 = _raw_moment(2, a_b, b_b)

    var_std = e_z2 - e_z**2
    result = sigma_b**2 * var_std

    return pt.switch((a_b > 1) & (b_b > 1), result, pt.nan)


def std(a, b, mu, sigma):
    """Compute standard deviation of the Jones-Faddy skew-t distribution."""
    variance = var(a, b, mu, sigma)
    return pt.switch(pt.isinf(variance) | pt.isnan(variance), variance, pt.sqrt(variance))


def skewness(a, b, mu, sigma):
    """
    Skewness of the Jones-Faddy skew-t distribution.

    The skewness exists when a > 1.5 and b > 1.5.
    """
    a_b, b_b, _, _ = pt.broadcast_arrays(a, b, mu, sigma)

    e_z = _raw_moment(1, a_b, b_b)
    e_z2 = _raw_moment(2, a_b, b_b)
    e_z3 = _raw_moment(3, a_b, b_b)

    var_z = e_z2 - e_z**2
    std_z = pt.sqrt(var_z)

    # E[(Z - mu)^3] = E[Z^3] - 3*E[Z]*E[Z^2] + 2*E[Z]^3
    third_central = e_z3 - 3 * e_z * e_z2 + 2 * e_z**3

    result = third_central / std_z**3
    return pt.switch((a_b > 1.5) & (b_b > 1.5), result, pt.nan)


def kurtosis(a, b, mu, sigma):
    """
    Excess kurtosis of the Jones-Faddy skew-t distribution.

    The kurtosis exists when a > 2 and b > 2.
    """
    a_b, b_b, _, _ = pt.broadcast_arrays(a, b, mu, sigma)

    e_z = _raw_moment(1, a_b, b_b)
    e_z2 = _raw_moment(2, a_b, b_b)
    e_z3 = _raw_moment(3, a_b, b_b)
    e_z4 = _raw_moment(4, a_b, b_b)

    var_z = e_z2 - e_z**2

    # E[(Z - mu)^4] = E[Z^4] - 4*E[Z]*E[Z^3] + 6*E[Z]^2*E[Z^2] - 3*E[Z]^4
    fourth_central = e_z4 - 4 * e_z * e_z3 + 6 * e_z**2 * e_z2 - 3 * e_z**4

    result = fourth_central / var_z**2 - 3
    return pt.switch((a_b > 2) & (b_b > 2), result, pt.nan)


def entropy(a, b, mu, sigma):
    """Compute differential entropy of the Jones-Faddy skew-t distribution."""
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)
    # Use wide symmetric bounds scaled by sigma
    # 50*sigma covers essentially all probability mass for reasonable a, b values
    lower = mu_b - 50 * sigma_b
    upper = mu_b + 50 * sigma_b
    return continuous_entropy(lower, upper, logpdf, a_b, b_b, mu_b, sigma_b)


def pdf(x, a, b, mu, sigma):
    """Probability density function of the Jones-Faddy skew-t distribution."""
    return pt.exp(logpdf(x, a, b, mu, sigma))


def logpdf(x, a, b, mu, sigma):
    """
    Log probability density function of the Jones-Faddy skew-t distribution.

    logpdf = (a + 0.5) * log(1 + z/sqrt(a+b+z^2))
           + (b + 0.5) * log(1 - z/sqrt(a+b+z^2))
           - (a + b - 1) * log(2) - betaln(a, b) - 0.5 * log(a + b)
           - log(sigma)

    where z = (x - mu) / sigma
    """
    z = (x - mu) / sigma
    sqrt_term = pt.sqrt(a + b + z**2)

    a_term = (a + 0.5) * pt.log(1 + z / sqrt_term)
    b_term = (b + 0.5) * pt.log(1 - z / sqrt_term)
    norm_const = (a + b - 1) * pt.log(2) + betaln(a, b) + 0.5 * pt.log(a + b)

    result = a_term + b_term - norm_const - pt.log(sigma)
    result = pt.switch(pt.isinf(x), pt.nan, result)
    return result


def cdf(x, a, b, mu, sigma):
    """
    Cumulative distribution function of the Jones-Faddy skew-t distribution.

    CDF = I_y(a, b) where y = 0.5 * (1 + z / sqrt(a + b + z^2))
    and I_y is the regularized incomplete beta function.
    """
    z = (x - mu) / sigma
    y = 0.5 * (1 + z / pt.sqrt(a + b + z**2))
    result = pt.betainc(a, b, y)
    result = pt.switch(pt.eq(x, -pt.inf), 0.0, result)
    result = pt.switch(pt.eq(x, pt.inf), 1.0, result)
    return result


def logcdf(x, a, b, mu, sigma):
    """
    Log cumulative distribution function of the Jones-Faddy skew-t distribution.

    Uses log of betainc directly for better numerical stability.
    """
    z = (x - mu) / sigma
    y = 0.5 * (1 + z / pt.sqrt(a + b + z**2))
    result = pt.log(pt.betainc(a, b, y))
    result = pt.switch(pt.eq(x, -pt.inf), -pt.inf, result)
    result = pt.switch(pt.eq(x, pt.inf), 0.0, result)
    return result


def sf(x, a, b, mu, sigma):
    """
    Survival function (1 - CDF) of the Jones-Faddy skew-t distribution.

    Uses symmetry property: sf(x; a, b, mu, sigma) = cdf(-x; b, a, -mu, sigma)
    This avoids computing 1 - cdf which loses precision for values near 1.
    """
    return cdf(-x, b, a, -mu, sigma)


def logsf(x, a, b, mu, sigma):
    """
    Log survival function of the Jones-Faddy skew-t distribution.

    Uses symmetry property: logsf(x; a, b) = logcdf(-x; b, a)
    This avoids computing log(1 - cdf) which loses precision.
    """
    return logcdf(-x, b, a, -mu, sigma)


def ppf(q, a, b, mu, sigma):
    """
    Percent point function (inverse CDF) of the Jones-Faddy skew-t distribution.

    Uses the inverse beta function: if y = betaincinv(a, b, q),
    then z = (2*y - 1) * sqrt(a + b) / (2 * sqrt(y * (1 - y)))
    """
    a_b, b_b, mu_b, sigma_b = pt.broadcast_arrays(a, b, mu, sigma)

    bval = betaincinv(a_b, b_b, q)
    num = (2 * bval - 1) * pt.sqrt(a_b + b_b)
    denom = 2 * pt.sqrt(bval * (1 - bval))
    z = num / denom

    result = mu_b + sigma_b * z
    return ppf_bounds_cont(result, q, -pt.inf, pt.inf)


def isf(q, a, b, mu, sigma):
    """Inverse survival function of the Jones-Faddy skew-t distribution."""
    return ppf(1 - q, a, b, mu, sigma)


def rvs(a, b, mu, sigma, size=None, random_state=None):
    """
    Random variates from the Jones-Faddy skew-t distribution.

    Uses inverse transform sampling: generate uniform samples and apply ppf.
    """
    u = pt.random.uniform(0, 1, size=size, rng=random_state)
    return ppf(u, a, b, mu, sigma)
