import pytensor.tensor as pt


def cdf_bounds(cdf_val, x, lower, upper):
    return pt.switch(
        pt.or_(pt.lt(x, lower), pt.gt(x, upper)),
        0.0,
        pt.switch(pt.eq(x, lower), 0.0, pt.switch(pt.eq(x, upper), 1.0, cdf_val)),
    )


def ppf_bounds_cont(x_val, q, lower, upper):
    return pt.switch(
        pt.or_(pt.lt(q, 0), pt.gt(q, 1)),
        pt.nan,
        pt.switch(pt.eq(q, 0), lower, pt.switch(pt.eq(q, 1), upper, x_val)),
    )
