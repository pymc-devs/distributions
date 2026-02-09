import math

import pytensor.tensor as pt
from pytensor import scan
from pytensor.scan.utils import until

from pytensor_distributions.helper import ppf_bounds_cont, ppf_bounds_disc


def find_ppf(q, x0, lower, upper, cdf_func, pdf_func, *params, max_iter=100, tol=1e-8):
    x0 = x0 + pt.zeros_like(q)

    def step(x_prev):
        cdf_val = cdf_func(x_prev, *params)
        f_x = pt.maximum(pdf_func(x_prev, *params), 1e-10)
        delta = (cdf_val - q) / f_x

        max_step = pt.maximum(pt.abs(x_prev), 1.0)
        delta = pt.clip(delta, -max_step, max_step)
        x_new = x_prev - delta

        converged = pt.abs(x_new - x_prev) < tol
        x_new = pt.switch(converged, x_prev, x_new)

        all_converged = pt.all(converged)
        return x_new, until(all_converged)

    x_seq = scan(fn=step, outputs_info=pt.shape_padleft(x0), n_steps=max_iter, return_updates=False)

    return ppf_bounds_cont(x_seq[-1].squeeze(), q, lower, upper)


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
