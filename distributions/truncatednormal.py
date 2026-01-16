import pytensor.tensor as pt

from distributions import normal as Normal
from distributions.helper import cdf_bounds, logdiffexp, ppf_bounds_cont


def _phi(z):
    return 0.5 * (1 + pt.erf(z / pt.sqrt(2.0)))


def _Phi_inv(p):
    return pt.sqrt(2.0) * pt.erfinv(2 * p - 1)


def _log_phi(z):
    return -0.5 * z**2 - pt.log(pt.sqrt(2 * pt.pi))


def _alpha_beta(mu, sigma, lower, upper):
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    return alpha, beta


def _log_erfc(x):
    # log(erfc(x)) = log(erfcx(x) * exp(-x^2)) = log(erfcx(x)) - x^2
    # Safe for large positive x.
    return pt.log(pt.erfcx(x)) - x**2


def _log_Z(alpha, beta):
    """Compute log(Phi(beta) - Phi(alpha)) robustly."""
    a = alpha / pt.sqrt(2.0)
    b = beta / pt.sqrt(2.0)
    log_half = pt.log(0.5)

    # Case 1: alpha > 0 (implies beta > 0). Upper tail.
    # Z = 0.5 * (erfc(a) - erfc(b)). a < b.
    safe_a_pos = pt.switch(pt.gt(alpha, 0), a, 0.0)
    safe_b_pos = pt.switch(pt.gt(alpha, 0), b, 0.0)
    res_pos = log_half + logdiffexp(_log_erfc(safe_a_pos), _log_erfc(safe_b_pos))

    # Case 2: beta < 0 (implies alpha < 0). Lower tail.
    # Z = 0.5 * (erfc(-b) - erfc(-a)). -b < -a.
    safe_neg_b = pt.switch(pt.lt(beta, 0), -b, 0.0)
    safe_neg_a = pt.switch(pt.lt(beta, 0), -a, 0.0)
    res_neg = log_half + logdiffexp(_log_erfc(safe_neg_b), _log_erfc(safe_neg_a))

    # Case 3: Straddles 0. alpha <= 0 <= beta.
    # Z = 0.5 * (erf(b) - erf(a)).
    # erf(b) >= 0, erf(a) <= 0. Difference is sum of magnitudes.
    res_straddle = log_half + pt.log(pt.erf(b) - pt.erf(a))

    return pt.switch(
        pt.gt(alpha, 0),
        res_pos,
        pt.switch(pt.lt(beta, 0), res_neg, res_straddle),
    )


def mean(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    log_Z = _log_Z(alpha, beta)

    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    term = pt.exp(log_phi_a - log_Z) - pt.exp(log_phi_b - log_Z)
    return mu + sigma * term


def mode(mu, sigma, lower, upper):
    return pt.clip(mu, lower, upper)


def median(mu, sigma, lower, upper):
    return ppf(0.5, mu, sigma, lower, upper)


def var(mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    log_Z = _log_Z(alpha, beta)

    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    term_a = pt.exp(log_phi_a - log_Z)
    term_b = pt.exp(log_phi_b - log_Z)

    term1 = alpha * term_a - beta * term_b
    term2 = pt.sqr(term_a - term_b)

    return sigma**2 * (1 + term1 - term2)


def std(mu, sigma, lower, upper):
    return pt.sqrt(var(mu, sigma, lower, upper))


def _raw_moments(alpha, beta):
    # This function needs to be updated or removed if we inline logic.
    # It returns m1, m2, m3, m4 for standard truncated normal.
    log_Z = _log_Z(alpha, beta)
    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    term_a = pt.exp(log_phi_a - log_Z)
    term_b = pt.exp(log_phi_b - log_Z)

    phi_a_Z = term_a
    phi_b_Z = term_b

    # m1 = (phi_a - phi_b) / Z
    m1 = phi_a_Z - phi_b_Z

    # m2 = 1 + (alpha * phi_a - beta * phi_b) / Z
    m2 = 1 + alpha * phi_a_Z - beta * phi_b_Z

    # m3 = 2*m1 + (alpha^2 * phi_a - beta^2 * phi_b) / Z
    m3 = 2 * m1 + alpha**2 * phi_a_Z - beta**2 * phi_b_Z

    # m4 = 3 + 3*(alpha*phi_a - beta*phi_b)/Z + (alpha^3*phi_a - beta^3*phi_b)/Z
    # m4 = 3 + 3*(m2 - 1) + ...
    m4 = 3 * m2 + alpha**3 * phi_a_Z - beta**3 * phi_b_Z

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
    log_Z = _log_Z(alpha, beta)

    # H = log(sqrt(2pi*e)*sigma*Z) + (alpha*phi(alpha) - beta*phi(beta))/(2Z)
    # H = log(sqrt(2pi*e)*sigma) + log(Z) + 0.5 * (alpha * (phi(a)/Z) - beta * (phi(b)/Z))

    log_phi_a = _log_phi(alpha)
    log_phi_b = _log_phi(beta)

    term_a = pt.exp(log_phi_a - log_Z)
    term_b = pt.exp(log_phi_b - log_Z)

    return (
        pt.log(pt.sqrt(2 * pt.pi * pt.e) * sigma) + log_Z + 0.5 * (alpha * term_a - beta * term_b)
    )


def pdf(x, mu, sigma, lower, upper):
    return pt.exp(logpdf(x, mu, sigma, lower, upper))


def logpdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    log_Z = _log_Z(alpha, beta)

    base_logpdf = Normal.logpdf(x, mu, sigma)
    result = base_logpdf - log_Z

    return pt.switch(
        pt.or_(pt.lt(x, lower), pt.gt(x, upper)),
        -pt.inf,
        result,
    )


def cdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    # Z(alpha, xi) / Z(alpha, beta)
    # logcdf = logZ(alpha, xi) - logZ(alpha, beta)
    # But for numerical stability in cdf (0 to 1), exp(logdiff) is fine.
    # Note: xi might be > beta or < alpha? Handled by cdf_bounds.

    xi = (x - mu) / sigma

    # We need log_Z(alpha, xi).
    # But _log_Z handles ordering alpha < beta.
    # If xi < alpha, log_Z(alpha, xi) might be invalid (beta < alpha).
    # But cdf_bounds handles x < lower (xi < alpha).
    # What if xi > beta? cdf_bounds handles it.
    # So we can compute _log_Z(alpha, xi) safely assuming alpha <= xi?
    # No, xi is variable.
    # But we can just use _log_Z(alpha, xi) and if xi < alpha, the result is garbage but masked by cdf_bounds.

    # However, _log_Z checks alpha > 0 etc.
    # If xi is used as beta, we need to ensure inputs are reasonable for the logic?
    # _log_Z(alpha, beta) expects alpha <= beta?
    # Actually logic:
    # Case 1: alpha > 0. Then beta > 0.
    # Case 2: beta < 0. Then alpha < 0.
    # If we pass xi as beta.
    # If alpha > 0, we expect xi > 0? Not necessarily.
    # If xi < alpha, xi might be < 0.
    # But if xi < alpha, cdf is 0.

    log_num = _log_Z(alpha, xi)
    log_den = _log_Z(alpha, beta)

    result = pt.exp(log_num - log_den)

    return cdf_bounds(result, x, lower, upper)


def logcdf(x, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    xi = (x - mu) / sigma

    log_num = _log_Z(alpha, xi)
    log_den = _log_Z(alpha, beta)

    result = log_num - log_den

    return pt.switch(
        pt.le(x, lower),
        -pt.inf,
        pt.switch(pt.ge(x, upper), 0.0, result),
    )


def sf(x, mu, sigma, lower, upper):
    return 1.0 - cdf(x, mu, sigma, lower, upper)


def logsf(x, mu, sigma, lower, upper):
    # log( (Phi(beta) - Phi(xi)) / Z )
    # log_num = _log_Z(xi, beta)

    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    xi = (x - mu) / sigma

    log_num = _log_Z(xi, beta)
    log_den = _log_Z(alpha, beta)

    result = log_num - log_den

    return pt.switch(
        pt.le(x, lower),
        0.0,
        pt.switch(pt.ge(x, upper), -pt.inf, result),
    )


def ppf(q, mu, sigma, lower, upper):
    alpha, beta = _alpha_beta(mu, sigma, lower, upper)
    # We use different strategies for tails.

    # Strategy 1 (Standard): x = Phi_inv( q*Phi(beta) + (1-q)*Phi(alpha) )
    # q * Z + Phi(alpha)
    # Z = Phi(beta) - Phi(alpha)
    # This works when alpha, beta are around 0 or negative (lower tail).
    # If alpha large positive, Phi(alpha) ~ 1, precision loss.

    # Strategy 2 (Survival): x = -Phi_inv( q*Sf(beta) + (1-q)*Sf(alpha) )? No.
    # Sf(x) = 1 - Phi(x). Phi(x) = 1 - Sf(x).
    # u = q*(1-Sf(beta)) + (1-q)*(1-Sf(alpha))
    #   = q - q*Sf(beta) + 1 - Sf(alpha) - q + q*Sf(alpha)
    #   = 1 - (q*Sf(beta) + (1-q)*Sf(alpha))
    # Phi(x) = 1 - Sf(x) => Sf(x) = 1 - u = q*Sf(beta) + (1-q)*Sf(alpha).
    # x = Sf_inv( q*Sf(beta) + (1-q)*Sf(alpha) )
    # Sf(z) = 0.5 * erfc(z/sqrt2).
    # Sf_inv(y) = sqrt2 * erfcinv(2y).

    # Use Strategy 2 when alpha > 0 (upper tail).
    # Use Strategy 1 otherwise.

    # Standard:
    # Z = _phi(beta) - _phi(alpha) (Not using _log_Z here as we need raw diff for q*Z?)
    # But q*Z + phi(alpha). If Z is tiny, we just get phi(alpha).
    # We can assume alpha < beta.

    def ppf_standard(q, alpha, beta):
        Z = _phi(beta) - _phi(alpha)
        return _Phi_inv(q * Z + _phi(alpha))

    def ppf_survival(q, alpha, beta):
        # Sf(z) = 0.5 * erfc(z/sqrt2)
        # term = q*Sf(beta) + (1-q)*Sf(alpha)
        sb = 0.5 * pt.erfc(beta / pt.sqrt(2.0))
        sa = 0.5 * pt.erfc(alpha / pt.sqrt(2.0))
        term = q * sb + (1 - q) * sa
        # x = sqrt2 * erfcinv(2 * term)
        return pt.sqrt(2.0) * pt.erfcinv(2 * term)

    result_standard = ppf_standard(q, alpha, beta)
    result_survival = ppf_survival(q, alpha, beta)

    result = pt.switch(pt.gt(alpha, 0), result_survival, result_standard)

    result = mu + sigma * result

    return ppf_bounds_cont(result, q, lower, upper)


def isf(q, mu, sigma, lower, upper):
    return ppf(1.0 - q, mu, sigma, lower, upper)


def rvs(mu, sigma, lower, upper, size=None, random_state=None):
    u = pt.random.uniform(0, 1, size=size, rng=random_state)
    return ppf(u, mu, sigma, lower, upper)
