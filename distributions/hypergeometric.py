import pytensor.tensor as pt

from distributions.helper import cdf_bounds, discrete_entropy
from distributions.optimization import find_ppf_discrete


def _support_lower(M, n, N):
    return pt.maximum(0, N + n - M)


def _support_upper(M, n, N):
    return pt.minimum(n, N)


def _log_binomial(n, k):
    return pt.gammaln(n + 1) - pt.gammaln(k + 1) - pt.gammaln(n - k + 1)


def mean(M, n, N):
    return N * n / M


def mode(M, n, N):
    return pt.floor((N + 1) * (n + 1) / (M + 2))


def median(M, n, N):
    return ppf(0.5, M, n, N)


def var(M, n, N):
    return N * n * (M - n) * (M - N) / (M * M * (M - 1))


def std(M, n, N):
    return pt.sqrt(var(M, n, N))


def skewness(M, n, N):
    numerator = (M - 2 * n) * pt.sqrt(M - 1) * (M - 2 * N)
    denominator = pt.sqrt(N * n * (M - n) * (M - N)) * (M - 2)
    return numerator / denominator


def kurtosis(M, n, N):
    M = pt.cast(M, "float64")
    n = pt.cast(n, "float64")
    N = pt.cast(N, "float64")
    m = M - n
    num = M * M * (M - 1) * (M * (M + 1) - 6 * N * (M - N) - 6 * n * m) + 6 * n * N * (
        M - N
    ) * m * (5 * M - 6)
    den = n * N * (M - N) * m * (M - 2) * (M - 3)
    return num / den


def entropy(M, n, N):
    lower = _support_lower(M, n, N)
    upper = _support_upper(M, n, N) + 1
    return discrete_entropy(lower, upper, logpdf, M, n, N)


def pdf(x, M, n, N):
    return pt.exp(logpdf(x, M, n, N))


def logpdf(x, M, n, N):
    x = pt.floor(x)
    lower = _support_lower(M, n, N)
    upper = _support_upper(M, n, N)
    in_support = pt.and_(pt.ge(x, lower), pt.le(x, upper))
    result = _log_binomial(n, x) + _log_binomial(M - n, N - x) - _log_binomial(M, N)
    return pt.switch(in_support, result, -pt.inf)


def cdf(x, M, n, N):
    lower = _support_lower(M, n, N)
    upper = _support_upper(M, n, N)
    x = pt.as_tensor_variable(x)
    x_floor = pt.floor(x)
    safe_k = pt.clip(x_floor, 0, upper)
    safe_k = pt.switch(pt.isnan(x), 0, safe_k)
    k_vals = pt.arange(0, pt.cast(upper + 1, "int64"))
    log_pmf_vals = logpdf(k_vals, M, n, N)
    pmf_vals = pt.exp(log_pmf_vals)
    cumsum = pt.cumsum(pmf_vals)
    k_idx = pt.cast(safe_k, "int64")
    raw_cdf = cumsum[k_idx]
    raw_cdf = pt.switch(pt.isnan(x), pt.nan, raw_cdf)
    return cdf_bounds(raw_cdf, x, lower, upper)


def logcdf(x, M, n, N):
    return pt.log(cdf(x, M, n, N))


def sf(x, M, n, N):
    return 1.0 - cdf(x, M, n, N)


def logsf(x, M, n, N):
    return pt.log1p(-cdf(x, M, n, N))


def ppf(q, M, n, N):
    lower = _support_lower(M, n, N)
    upper = _support_upper(M, n, N)
    return find_ppf_discrete(q, lower, upper, cdf, M, n, N)


def isf(q, M, n, N):
    return ppf(1.0 - q, M, n, N)


def rvs(M, n, N, size=None, random_state=None):
    return pt.random.hypergeometric(n, M - n, N, size=size, rng=random_state)
