from types import ModuleType

import pytensor.tensor as pt
from pytensor.tensor import TensorLike, TensorVariable


def logpdf(
    x: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    x = pt.as_tensor_variable(x)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_logpdf = dist.logpdf(x, *params)
    base_logcdf_lower = dist.logcdf(lower_val, *params)
    base_logsf_upper = dist.logsf(upper_val, *params)

    return pt.switch(
        pt.lt(x, lower_val),
        -pt.inf,  # x < lower
        pt.switch(
            pt.eq(x, lower_val),
            base_logcdf_lower,  # x = lower (point mass)
            pt.switch(
                pt.lt(x, upper_val),
                base_logpdf,  # lower < x < upper (continuous)
                pt.switch(
                    pt.eq(x, upper_val),
                    base_logsf_upper,  # x = upper (point mass)
                    -pt.inf,  # x > upper
                ),
            ),
        ),
    )


def pdf(
    x: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    return pt.exp(logpdf(x, dist, lower, upper, *params))


def cdf(
    x: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    x = pt.as_tensor_variable(x)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_cdf = dist.cdf(x, *params)

    return pt.switch(
        pt.lt(x, lower_val),
        0.0,  # x < lower
        pt.switch(
            pt.ge(x, upper_val),
            1.0,  # x >= upper
            base_cdf,  # lower <= x < upper
        ),
    )


def logcdf(
    x: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    x = pt.as_tensor_variable(x)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_logcdf = dist.logcdf(x, *params)

    return pt.switch(
        pt.lt(x, lower_val),
        -pt.inf,  # x < lower
        pt.switch(
            pt.ge(x, upper_val),
            0.0,  # x >= upper (log(1) = 0)
            base_logcdf,  # lower <= x < upper
        ),
    )


def sf(
    x: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    x = pt.as_tensor_variable(x)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_sf = dist.sf(x, *params)

    return pt.switch(
        pt.lt(x, lower_val),
        1.0,  # x < lower
        pt.switch(
            pt.ge(x, upper_val),
            0.0,  # x >= upper
            base_sf,  # lower <= x < upper
        ),
    )


def logsf(
    x: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    x = pt.as_tensor_variable(x)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_logsf = dist.logsf(x, *params)

    return pt.switch(
        pt.lt(x, lower_val),
        0.0,  # x < lower (log(1) = 0)
        pt.switch(
            pt.ge(x, upper_val),
            -pt.inf,  # x >= upper (log(0) = -inf)
            base_logsf,  # lower <= x < upper
        ),
    )


def rvs(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    size: tuple[int, ...] | None = None,
    random_state=None,
) -> TensorVariable:
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_samples = dist.rvs(*params, size=size, random_state=random_state)
    return pt.clip(base_samples, lower_val, upper_val)


def ppf(
    q: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    q = pt.as_tensor_variable(q)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_ppf = dist.ppf(q, *params)

    return pt.clip(base_ppf, lower_val, upper_val)


def isf(
    q: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    q = pt.as_tensor_variable(q)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_isf = dist.isf(q, *params)

    return pt.clip(base_isf, lower_val, upper_val)


def _censored_raw_moment(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    order: int = 1,
    n_points: int = 1000,
) -> TensorVariable:
    if lower is None:
        int_lower = dist.ppf(0.0001, *params)
    else:
        int_lower = pt.as_tensor_variable(lower)

    if upper is None:
        int_upper = dist.ppf(0.9999, *params)
    else:
        int_upper = pt.as_tensor_variable(upper)

    if lower is not None:
        lower_val = pt.as_tensor_variable(lower)
        cdf_lower = dist.cdf(lower_val, *params)
        lower_contribution = (lower_val**order) * cdf_lower
    else:
        lower_contribution = 0.0

    if upper is not None:
        upper_val = pt.as_tensor_variable(upper)
        sf_upper = dist.sf(upper_val, *params)
        upper_contribution = (upper_val**order) * sf_upper
    else:
        upper_contribution = 0.0

    x_vals = pt.linspace(int_lower, int_upper, n_points)
    pdf_vals = dist.pdf(x_vals, *params)
    integrand = (x_vals**order) * pdf_vals

    dx = (int_upper - int_lower) / (n_points - 1)
    interior_integral = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1]) + 0.5 * integrand[-1])

    return lower_contribution + interior_integral + upper_contribution


def mean(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    n_points: int = 1000,
) -> TensorVariable:
    return _censored_raw_moment(dist, lower, upper, *params, order=1, n_points=n_points)


def var(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    n_points: int = 1000,
) -> TensorVariable:
    mean_val = _censored_raw_moment(dist, lower, upper, *params, order=1, n_points=n_points)
    second_moment = _censored_raw_moment(dist, lower, upper, *params, order=2, n_points=n_points)
    return second_moment - mean_val**2


def std(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    n_points: int = 1000,
) -> TensorVariable:
    return pt.sqrt(var(dist, lower, upper, *params, n_points=n_points))


def median(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    return ppf(0.5, dist, lower, upper, *params)


def mode(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    base_mode = dist.mode(*params)

    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    clipped_mode = pt.clip(base_mode, lower_val, upper_val)

    pdf_at_mode = dist.pdf(clipped_mode, *params)

    if lower is not None:
        cdf_lower = dist.cdf(lower_val, *params)
    else:
        cdf_lower = pt.constant(0.0)

    if upper is not None:
        sf_upper = dist.sf(upper_val, *params)
    else:
        sf_upper = pt.constant(0.0)

    return pt.switch(
        pt.and_(pt.ge(cdf_lower, pdf_at_mode), pt.ge(cdf_lower, sf_upper)),
        lower_val,
        pt.switch(
            pt.and_(pt.gt(sf_upper, cdf_lower), pt.ge(sf_upper, pdf_at_mode)),
            upper_val,
            clipped_mode,
        ),
    )


def _censored_central_moment(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    order: int,
    mean_val: TensorVariable,
    n_points: int = 1000,
) -> TensorVariable:
    if lower is None:
        int_lower = dist.ppf(0.0001, *params)
    else:
        int_lower = pt.as_tensor_variable(lower)

    if upper is None:
        int_upper = dist.ppf(0.9999, *params)
    else:
        int_upper = pt.as_tensor_variable(upper)

    if lower is not None:
        lower_val = pt.as_tensor_variable(lower)
        cdf_lower = dist.cdf(lower_val, *params)
        lower_contribution = ((lower_val - mean_val) ** order) * cdf_lower
    else:
        lower_contribution = 0.0

    if upper is not None:
        upper_val = pt.as_tensor_variable(upper)
        sf_upper = dist.sf(upper_val, *params)
        upper_contribution = ((upper_val - mean_val) ** order) * sf_upper
    else:
        upper_contribution = 0.0

    x_vals = pt.linspace(int_lower, int_upper, n_points)
    pdf_vals = dist.pdf(x_vals, *params)
    integrand = ((x_vals - mean_val) ** order) * pdf_vals

    dx = (int_upper - int_lower) / (n_points - 1)
    interior_integral = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1]) + 0.5 * integrand[-1])

    return lower_contribution + interior_integral + upper_contribution


def skewness(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    n_points: int = 1000,
) -> TensorVariable:
    mean_val = _censored_raw_moment(dist, lower, upper, *params, order=1, n_points=n_points)
    var_val = var(dist, lower, upper, *params, n_points=n_points)
    third_central = _censored_central_moment(
        dist, lower, upper, *params, order=3, mean_val=mean_val, n_points=n_points
    )
    return third_central / (pt.sqrt(var_val) ** 3)


def kurtosis(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    n_points: int = 1000,
) -> TensorVariable:
    mean_val = _censored_raw_moment(dist, lower, upper, *params, order=1, n_points=n_points)
    var_val = var(dist, lower, upper, *params, n_points=n_points)
    fourth_central = _censored_central_moment(
        dist, lower, upper, *params, order=4, mean_val=mean_val, n_points=n_points
    )
    return fourth_central / (var_val**2) - 3


def entropy(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    n_points: int = 1000,
) -> TensorVariable:
    if lower is None:
        int_lower = dist.ppf(0.0001, *params)
    else:
        int_lower = pt.as_tensor_variable(lower)

    if upper is None:
        int_upper = dist.ppf(0.9999, *params)
    else:
        int_upper = pt.as_tensor_variable(upper)

    if lower is not None:
        lower_val = pt.as_tensor_variable(lower)
        cdf_lower = dist.cdf(lower_val, *params)
        lower_entropy = pt.switch(
            pt.gt(cdf_lower, 0.0),
            -cdf_lower * pt.log(cdf_lower),
            0.0,
        )
    else:
        lower_entropy = 0.0

    if upper is not None:
        upper_val = pt.as_tensor_variable(upper)
        sf_upper = dist.sf(upper_val, *params)
        upper_entropy = pt.switch(
            pt.gt(sf_upper, 0.0),
            -sf_upper * pt.log(sf_upper),
            0.0,
        )
    else:
        upper_entropy = 0.0

    x_vals = pt.linspace(int_lower, int_upper, n_points)
    logpdf_vals = dist.logpdf(x_vals, *params)
    pdf_vals = pt.exp(logpdf_vals)

    integrand = -pdf_vals * logpdf_vals

    dx = (int_upper - int_lower) / (n_points - 1)
    interior_entropy = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1]) + 0.5 * integrand[-1])

    return lower_entropy + interior_entropy + upper_entropy
