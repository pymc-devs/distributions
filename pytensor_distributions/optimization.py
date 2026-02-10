import math

import pytensor.tensor as pt
from pytensor import scan
from pytensor.scan.utils import until

from pytensor_distributions.helper import ppf_bounds_cont, ppf_bounds_disc


def find_ppf(q, x0, lower, upper, cdf_func, pdf_func, *params, max_iter=100, tol=1e-8):
    x0 = x0 + pt.zeros_like(q)

    def step(x_prev):
        x_prev_squeezed = pt.squeeze(x_prev)

        cdf_val = cdf_func(x_prev_squeezed, *params)
        f_x = pt.maximum(pdf_func(x_prev_squeezed, *params), 1e-10)
        delta = (cdf_val - q) / f_x

        max_step = pt.maximum(pt.abs(x_prev_squeezed), 1.0)
        delta = pt.clip(delta, -max_step, max_step)
        x_new = x_prev_squeezed - delta

        converged = pt.abs(x_new - x_prev_squeezed) < tol
        x_new = pt.switch(converged, x_prev_squeezed, x_new)

        all_converged = pt.all(converged)
        return pt.shape_padleft(x_new), until(all_converged)

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


def find_ppf_discrete(q, x0, lower, upper, cdf_func, pmf_func, *params, max_iter=100, tol=1e-7):
    """Find PPF for discrete distributions."""
    x0 = pt.floor(x0) + pt.zeros_like(q)

    def step(x_prev):
        x_prev_squeezed = pt.squeeze(x_prev)
        x_int = pt.floor(x_prev_squeezed)

        cdf_val = cdf_func(x_int, *params)
        cdf_val_minus = cdf_func(x_int - 1, *params)

        found = (cdf_val >= q - tol) & (cdf_val_minus < q)
        pmf_val = pt.maximum(pmf_func(x_int, *params), 1e-10)
        delta = (cdf_val - q) / pmf_val

        delta_discrete = pt.switch(pt.abs(delta) < 0.5, pt.sign(cdf_val - q), pt.floor(delta))

        x_new = x_int - delta_discrete

        x_new = pt.clip(x_new, lower, upper)
        x_new = pt.switch(found, x_int, x_new)
        all_converged = pt.all(found)

        return pt.shape_padleft(x_new), until(all_converged)

    x_seq = scan(fn=step, outputs_info=pt.shape_padleft(x0), n_steps=max_iter, return_updates=False)

    return ppf_bounds_disc(x_seq[-1].squeeze(), q, lower, upper)
