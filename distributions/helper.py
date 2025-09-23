import numpy as np
import pytensor.tensor as pt

eps = np.finfo(float).eps

def from_precision(precision):
    sigma = 1 / precision**0.5
    return sigma

def to_precision(sigma):
    precision = 1 / (eps + sigma**2)
    return precision


def all_not_none(*args):
    for arg in args:
        if arg is None:
            return False
    return True


def any_not_none(*args):
    for arg in args:
        if arg is not None:
            return True
    return False


def ppf_bounds_cont(value, q, lower, upper):
    q = pt.as_tensor_variable(q)
    return pt.switch(
        pt.and_(q >= lower, q <= upper),
        value,
        np.nan,
    )