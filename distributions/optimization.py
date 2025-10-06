import pytensor.tensor as pt


def find_ppf(q, params, lower, upper, cdf):
    """
    Compute the inverse CDF using the bisection method.
    """
    left = pt.switch(pt.isinf(lower), -10.0, lower + 1e-10)
    right = pt.switch(pt.isinf(upper), 10.0, upper - 1e-10)

    factor = 10.0
    f_left = cdf(left, *params) - q
    f_right = cdf(right, *params) - q

    left = pt.switch(f_left > 0, left - factor, left)
    right = pt.switch(f_right < 0, right + factor, right)

    for _ in range(50):
        mid = 0.5 * (left + right)
        f_mid = cdf(mid, *params) - q
        f_lower_curr = cdf(left, *params) - q

        new_lower = pt.switch(pt.lt(f_mid * f_lower_curr, 0), left, mid)
        new_upper = pt.switch(pt.lt(f_mid * f_lower_curr, 0), mid, right)

        left = new_lower
        right = new_upper

    return 0.5 * (left + right)
