import pytensor.tensor as pt


def ppf_bounds_cont(value, q, lower, upper):
    q = pt.as_tensor_variable(q)
    return pt.switch(
        pt.and_(q >= lower, q <= upper),
        value,
        pt.nan,
    )