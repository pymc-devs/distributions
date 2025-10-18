import pytensor.tensor as pt

from .helper import ppf_bounds_cont, ppf_bounds_disc
from .normal import ppf as normal_ppf


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


def find_ppf_discrete(q, lower, upper, cdf, *params):
    """
    Compute the inverse CDF using the bisection method.

    The continuous bisection method finds where CDF(x) ≈ q. For discrete distributions,
    we round to the nearest integer and then check if we need to adjust.
    """
    rounded_k = pt.round(find_ppf(q, lower, upper, cdf, *params))
    # return ppf_bounds_disc(rounded_k, q, lower, upper)
    cdf_k = cdf(rounded_k, *params)
    rounded_k = pt.switch(pt.lt(cdf_k, q), rounded_k + 1, rounded_k)
    return ppf_bounds_disc(rounded_k, q, lower, upper)
