import pytensor.tensor as pt
from pytensor.tensor.special import betaln
from pytensor.tensor.math import betaincinv

from .helper import ppf_bounds_cont


def mean(alpha, beta):
    return alpha / (alpha + beta)

def mode(alpha, beta):
    alpha_b, beta_b = pt.broadcast_arrays(alpha, beta)
    result = pt.full_like(alpha_b, pt.nan)

    result = pt.where(pt.equal(alpha_b, 1) & pt.equal(beta_b, 1), 0.5, result)
    result = pt.where(pt.equal(alpha_b, 1) & (beta_b > 1), 0.0, result)
    result = pt.where(pt.equal(beta_b, 1) & (alpha_b > 1), 1.0, result)
    result = pt.where((alpha_b > 1) & (beta_b > 1),
                      (alpha_b - 1) / (alpha_b + beta_b - 2),
                      result)
    
    return result

def median(alpha, beta):
    return ppf(0.5, alpha, beta)

def var(alpha, beta):
    return (alpha * beta) / (
        pt.pow(alpha + beta, 2) * (alpha + beta + 1)
    )

def std(alpha, beta):
    return pt.sqrt(var(alpha, beta))

def skewness(alpha, beta):
    alpha_b, beta_b = pt.broadcast_arrays(alpha, beta)
    
    psc = alpha_b + beta_b
    result = pt.where(
        pt.eq(alpha_b, beta_b), 0.0,
        (2 * (beta_b - alpha_b) * pt.sqrt(psc + 1)) / (
            (psc + 2) * pt.sqrt(alpha_b * beta_b)
        )
    )
    return result


def kurtosis(alpha, beta):
    alpha_b, beta_b = pt.broadcast_arrays(alpha, beta)
    psc = alpha_b + beta_b
    prod = alpha_b * beta_b
    result = (6 * (pt.abs(alpha_b - beta_b) ** 2 * (psc + 1) - prod * (psc + 2))
    / (prod * (psc + 2) * (psc + 3)))
    return result

def entropy(alpha, beta):
    alpha_b, beta_b = pt.broadcast_arrays(alpha, beta)
    psc = alpha_b + beta_b
    return (
        betaln(alpha_b, beta_b)
        - (alpha_b - 1) * pt.psi(alpha_b)
        - (beta_b - 1) * pt.psi(beta_b)
        + (psc - 2) * pt.psi(psc)
    )

def cdf(x, alpha, beta):
    return pt.exp(logcdf(x, alpha, beta))

def isf(x, alpha, beta):
    return ppf(1 - x, alpha, beta)

def pdf(x, alpha, beta):
    return pt.exp(logpdf(x, alpha, beta))

def ppf(q, alpha, beta):
    return ppf_bounds_cont(betaincinv(alpha, beta, q), q, 0.0, 1.0)

def sf(x, alpha, beta):
    return pt.exp(logsf(x, alpha, beta))

def rvs(alpha, beta, size=None, random_state=None):
     return pt.random.beta(alpha, beta, rng=random_state, size=size)

def logcdf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0),
        -pt.inf,
        pt.switch(
            pt.lt(x, 1),
            pt.log(pt.betainc(alpha, beta, x)),
            0,
        ),
    )

def logpdf(x, alpha, beta):
    result = (
        pt.switch(pt.eq(alpha, 1.0), 0.0, (alpha - 1.0) * pt.log(x))
        + pt.switch(pt.eq(beta, 1.0), 0.0, (beta - 1.0) * pt.log1p(-x))
        - (pt.gammaln(alpha) + pt.gammaln(beta) - pt.gammaln(alpha + beta))
    )
    return pt.switch(pt.bitwise_and(pt.ge(x, 0.0), pt.le(x, 1.0)), result, -pt.inf)


def logsf(x, alpha, beta):
    return pt.switch(
        pt.lt(x, 0),
        0, 
        pt.switch(
            pt.lt(x, 1),
            pt.log(pt.betainc(beta, alpha, 1 - x)),  
            -pt.inf,
        ),
    )






