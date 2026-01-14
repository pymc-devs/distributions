import math

import pytensor.tensor as pt

from distributions.helper import ppf_bounds_cont, ppf_bounds_disc


def find_ppf(q, lower, upper, cdf, *params):
    """
    Compute the inverse CDF using the bisection method.

    Uses iterative expansion for infinite bounds.

    Note: We need to improve this method!!!
    """

    def func(x):
        return cdf(x, *params) - q

    factor = 10.0
    right = pt.switch(pt.isinf(upper), factor, upper)
    left = pt.switch(pt.isinf(lower), -factor, lower)

    for _ in range(10):
        f_left = func(left)
        should_expand = pt.gt(f_left, 0.0)
        new_left = left * factor
        new_right = left
        left = pt.switch(should_expand, new_left, left)
        right = pt.switch(should_expand, new_right, right)

    for _ in range(10):
        f_right = func(right)
        should_expand = pt.lt(f_right, 0.0)
        new_left = right
        new_right = right * factor
        left = pt.switch(should_expand, new_left, left)
        right = pt.switch(should_expand, new_right, right)

    for _ in range(50):
        mid = 0.5 * (left + right)
        f_mid = cdf(mid, *params) - q

        new_lower = pt.switch(pt.lt(f_mid, 0), mid, left)
        new_upper = pt.switch(pt.lt(f_mid, 0), right, mid)

        left = new_lower
        right = new_upper

    return ppf_bounds_cont(0.5 * (left + right), q, lower, upper)


def _is_scalar_param(param):
    """Check if a parameter is a scalar (0-dimensional) at graph-build time."""
    if hasattr(param, "ndim"):
        return param.ndim == 0
    # For Python scalars
    import numpy as np

    return np.ndim(param) == 0


def _should_use_bisection(lower, upper, params, max_direct_search_size=10_000):
    """Compile-time check to select PPF algorithm for discrete distributions.

    This function inspects bounds at graph-construction time to choose between:
    - Direct search: Fast for narrow bounded support (e.g., BetaBinomial, Binomial)
    - Bisection: Required for unbounded or wide support (e.g., Poisson, NegativeBinomial)

    The check happens at Python level during graph construction, not during
    PyTensor execution. This is intentional: a fully symbolic approach using
    pt.switch would evaluate both branches, causing performance issues.

    Parameters
    ----------
    lower : int, float, or PyTensor constant
        Lower bound of the distribution support
    upper : int, float, or PyTensor constant
        Upper bound of the distribution support
    params : tuple
        Distribution parameters - if any are non-scalar, bisection is required
        to handle broadcasting correctly.
    max_direct_search_size : int, default 10_000
        Maximum range size for direct search. Larger ranges use bisection.

    Returns
    -------
    bool
        True if bisection should be used, False for direct search.
    """
    # Check if any parameter is non-scalar (array) - direct search doesn't
    # handle broadcasting correctly, so fall back to bisection
    for param in params:
        if not _is_scalar_param(param):
            return True

    try:
        # Extract constant values at graph-build time
        if hasattr(lower, "data"):
            lower_val = float(lower.data)
        else:
            lower_val = float(lower)

        if hasattr(upper, "data"):
            upper_val = float(upper.data)
        else:
            upper_val = float(upper)
    except (TypeError, ValueError):
        # Symbolic (non-constant) bounds - use bisection as safe default
        return True

    # Check for infinite bounds
    if not (math.isfinite(lower_val) and math.isfinite(upper_val)):
        return True

    # Check if range exceeds threshold
    return (upper_val - lower_val) > max_direct_search_size


def find_ppf_discrete(q, lower, upper, cdf, *params):
    """
    Compute the inverse CDF for discrete distributions.

    For narrow bounded support, uses direct search over all values (fast).
    For unbounded or wide support, uses bisection method.
    """
    if _should_use_bisection(lower, upper, params):
        # Use bisection method for unbounded or wide ranges
        rounded_k = pt.round(find_ppf(q, lower, upper, cdf, *params))
        cdf_k = cdf(rounded_k, *params)
        rounded_k = pt.switch(pt.lt(cdf_k, q), rounded_k + 1, rounded_k)
        return ppf_bounds_disc(rounded_k, q, lower, upper)

    # Bounded case with narrow range: direct search over all values
    q = pt.as_tensor_variable(q)

    # Create array of all possible values in support
    k_vals = pt.arange(lower, upper + 1)

    # Compute CDF for all values - shape: (n_support,)
    cdf_vals = cdf(k_vals, *params)

    # Use a small tolerance for floating point comparison
    eps = 1e-10

    if q.ndim == 0:
        # Scalar case
        exceeds_q = pt.ge(cdf_vals, q - eps)
        first_idx = pt.argmax(exceeds_q)
        result = k_vals[first_idx]
    else:
        # Array case - need broadcasting
        exceeds_q = pt.ge(cdf_vals[:, None], q[None, :] - eps)
        first_idx = pt.argmax(exceeds_q, axis=0)
        result = k_vals[first_idx]

    return ppf_bounds_disc(result, q, lower, upper)


def von_mises_ppf(q, mu, kappa, cdf_func):
    """
    Compute the percent point function (inverse CDF) of von Mises distribution.

    Parameters
    ----------
    q : tensor
        Quantile values (between 0 and 1)
    mu : tensor
        Mean direction (location parameter)
    kappa : tensor
        Concentration parameter
    cdf_func : callable
        CDF function with signature cdf_func(x, mu, kappa) -> cdf_value

    Returns
    -------
    tensor
        PPF values (angles) in the range [-pi, pi]
    """
    left = -pt.pi * pt.ones_like(q)
    right = pt.pi * pt.ones_like(q)

    for _ in range(10):
        mid = 0.5 * (left + right)
        f_mid = cdf_func(mid, mu, kappa) - q

        left = pt.switch(f_mid < 0, mid, left)
        right = pt.switch(f_mid < 0, right, mid)

    result = 0.5 * (left + right)

    result = pt.switch(q < 0, pt.nan, result)
    result = pt.switch(q > 1, pt.nan, result)
    result = pt.switch(pt.eq(q, 0), -pt.inf, result)
    result = pt.switch(pt.eq(q, 1), pt.inf, result)
    return result
