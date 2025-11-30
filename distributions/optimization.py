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


def _should_use_bisection(lower, upper, max_direct_search_size=10_000):
    """Check if bisection should be used instead of direct search.

    Uses bisection if:
    - Upper bound is infinite
    - Range is too wide (> max_direct_search_size)
    - Values cannot be evaluated at graph-build time

    Evaluates bounds at graph-build time for constants.
    """
    try:
        lower_t = pt.as_tensor_variable(lower)
        upper_t = pt.as_tensor_variable(upper)
        lower_val = float(lower_t.eval())
        upper_val = float(upper_t.eval())

        # Check for infinite bounds
        if not (math.isfinite(lower_val) and math.isfinite(upper_val)):
            return True

        # Check if range is too wide
        if (upper_val - lower_val) > max_direct_search_size:
            return True

        return False
    except Exception:
        # If evaluation fails, use bisection as safe default
        return True


def find_ppf_discrete(q, lower, upper, cdf, *params):
    """
    Compute the inverse CDF for discrete distributions.

    For narrow bounded support, uses direct search over all values (fast).
    For unbounded or wide support, uses bisection method.
    """
    if _should_use_bisection(lower, upper):
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
