"""Censored distribution modifier.

This module provides functions to create censored versions of continuous distributions.
A censored distribution clips values outside the censoring bounds to the bounds themselves,
creating point masses at the lower and/or upper bounds.

The PDF of a censored distribution is:
- 0 for x < lower
- CDF(lower) for x = lower (point mass)
- base_pdf(x) for lower < x < upper
- SF(upper) for x = upper (point mass)
- 0 for x > upper

References
----------
- PyMC Censored Distribution:
  https://www.pymc.io/projects/docs/en/latest/_modules/pymc/distributions/censored.html
"""

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
    """
    Log-PDF of a censored distribution.

    Parameters
    ----------
    x : tensor
        Point at which to evaluate the log-PDF.
    dist : module
        Base distribution module (must have logpdf, logcdf, logsf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Log-PDF value at x.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> x = pt.dvector("x")
    >>> logpdf_val = censored.logpdf(x, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    PDF of a censored distribution.

    Parameters
    ----------
    x : tensor
        Point at which to evaluate the PDF.
    dist : module
        Base distribution module (must have logpdf, logcdf, logsf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        PDF value at x.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> x = pt.dvector("x")
    >>> pdf_val = censored.pdf(x, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    return pt.exp(logpdf(x, dist, lower, upper, *params))


def cdf(
    x: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    """
    CDF of a censored distribution.

    Parameters
    ----------
    x : tensor
        Point at which to evaluate the CDF.
    dist : module
        Base distribution module (must have cdf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        CDF value at x.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> x = pt.dvector("x")
    >>> cdf_val = censored.cdf(x, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    Log-CDF of a censored distribution.

    Parameters
    ----------
    x : tensor
        Point at which to evaluate the log-CDF.
    dist : module
        Base distribution module (must have logcdf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Log-CDF value at x.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> x = pt.dvector("x")
    >>> logcdf_val = censored.logcdf(x, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    Survival function (1 - CDF) of a censored distribution.

    Parameters
    ----------
    x : tensor
        Point at which to evaluate the survival function.
    dist : module
        Base distribution module (must have sf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Survival function value at x.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> x = pt.dvector("x")
    >>> sf_val = censored.sf(x, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    Log-survival function of a censored distribution.

    Parameters
    ----------
    x : tensor
        Point at which to evaluate the log-survival function.
    dist : module
        Base distribution module (must have logsf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Log-survival function value at x.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> x = pt.dvector("x")
    >>> logsf_val = censored.logsf(x, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    Generate random samples from a censored distribution.

    Samples are generated from the base distribution and then clipped
    to the censoring bounds.

    Parameters
    ----------
    dist : module
        Base distribution module (must have rvs).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.
    size : tuple of int or None
        Output shape. If None, the output shape is determined by broadcasting
        the distribution parameters.
    random_state : RandomState or None
        Random number generator state.

    Returns
    -------
    tensor
        Random samples clipped to [lower, upper].

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> samples = censored.rvs(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0, size=(100,))
    """
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
    """
    Percent point function (inverse CDF) of a censored distribution.

    Parameters
    ----------
    q : tensor
        Probability value(s) between 0 and 1.
    dist : module
        Base distribution module (must have ppf, cdf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Quantile value(s) clipped to [lower, upper].

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> q = pt.dvector("q")
    >>> ppf_val = censored.ppf(q, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    q = pt.as_tensor_variable(q)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_ppf = dist.ppf(q, *params)

    # Clip the result to the censoring bounds
    return pt.clip(base_ppf, lower_val, upper_val)


def isf(
    q: TensorLike,
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    """
    Inverse survival function of a censored distribution.

    Parameters
    ----------
    q : tensor
        Probability value(s) between 0 and 1.
    dist : module
        Base distribution module (must have isf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Quantile value(s) clipped to [lower, upper].

    Examples
    --------
    >>> from distributions import censored, normal
    >>> import pytensor.tensor as pt
    >>> q = pt.dvector("q")
    >>> isf_val = censored.isf(q, normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    q = pt.as_tensor_variable(q)
    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    base_isf = dist.isf(q, *params)

    # Clip the result to the censoring bounds
    return pt.clip(base_isf, lower_val, upper_val)


def _censored_raw_moment(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    order: int = 1,
    n_points: int = 1000,
) -> TensorVariable:
    """
    Compute raw moments of a censored distribution.

    Uses numerical integration for the continuous interior and adds
    point mass contributions at the bounds.

    Parameters
    ----------
    dist : module
        Base distribution module.
    lower : float or None
        Lower censoring bound.
    upper : float or None
        Upper censoring bound.
    *params : tensors
        Parameters for the base distribution.
    order : int
        Order of the moment (1 for mean, 2 for E[X^2], etc.).
    n_points : int
        Number of integration points.

    Returns
    -------
    tensor
        Raw moment of order `order`.
    """
    # Determine integration bounds
    # When bounds are None, use ppf to get reasonable finite bounds
    if lower is None:
        int_lower = dist.ppf(0.0001, *params)
    else:
        int_lower = pt.as_tensor_variable(lower)

    if upper is None:
        int_upper = dist.ppf(0.9999, *params)
    else:
        int_upper = pt.as_tensor_variable(upper)

    # Point mass contributions
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

    # Interior integral using trapezoidal rule
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
    """
    Mean of a censored distribution.

    The mean is computed as:
    E[X] = lower * CDF(lower) + integral(x * pdf(x), lower, upper) + upper * SF(upper)

    Parameters
    ----------
    dist : module
        Base distribution module (must have pdf, cdf, sf, ppf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.
    n_points : int
        Number of integration points for numerical integration.

    Returns
    -------
    tensor
        Mean value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> mean_val = censored.mean(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    return _censored_raw_moment(dist, lower, upper, *params, order=1, n_points=n_points)


def var(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
    n_points: int = 1000,
) -> TensorVariable:
    """
    Variance of a censored distribution.

    The variance is computed as: Var[X] = E[X^2] - E[X]^2

    Parameters
    ----------
    dist : module
        Base distribution module (must have pdf, cdf, sf, ppf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.
    n_points : int
        Number of integration points for numerical integration.

    Returns
    -------
    tensor
        Variance value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> var_val = censored.var(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    Compute the standard deviation of a censored distribution.

    Parameters
    ----------
    dist : module
        Base distribution module (must have pdf, cdf, sf, ppf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.
    n_points : int
        Number of integration points for numerical integration.

    Returns
    -------
    tensor
        Standard deviation value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> std_val = censored.std(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    return pt.sqrt(var(dist, lower, upper, *params, n_points=n_points))


def median(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    """
    Median of a censored distribution.

    The median is the value x such that CDF(x) = 0.5, which corresponds
    to ppf(0.5) for the censored distribution.

    Parameters
    ----------
    dist : module
        Base distribution module (must have ppf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Median value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> median_val = censored.median(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    return ppf(0.5, dist, lower, upper, *params)


def mode(
    dist: ModuleType,
    lower: TensorLike | None,
    upper: TensorLike | None,
    *params,
) -> TensorVariable:
    """
    Mode of a censored distribution.

    The mode is the value with highest probability density. For a censored
    distribution, this can be:
    - The lower bound (if CDF(lower) is the largest "density")
    - The upper bound (if SF(upper) is the largest "density")
    - The base distribution's mode (clipped to bounds)

    The comparison is between point mass probabilities at bounds and the
    PDF value at the base distribution's mode.

    Parameters
    ----------
    dist : module
        Base distribution module (must have mode, pdf, cdf, sf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.

    Returns
    -------
    tensor
        Mode value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> mode_val = censored.mode(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    # Get base distribution's mode (clipped to bounds)
    base_mode = dist.mode(*params)

    lower_val = -pt.inf if lower is None else pt.as_tensor_variable(lower)
    upper_val = pt.inf if upper is None else pt.as_tensor_variable(upper)

    # Clip the base mode to bounds
    clipped_mode = pt.clip(base_mode, lower_val, upper_val)

    # Get PDF at the clipped mode
    pdf_at_mode = dist.pdf(clipped_mode, *params)

    # Get point mass probabilities at bounds
    if lower is not None:
        cdf_lower = dist.cdf(lower_val, *params)
    else:
        cdf_lower = pt.constant(0.0)

    if upper is not None:
        sf_upper = dist.sf(upper_val, *params)
    else:
        sf_upper = pt.constant(0.0)

    # Compare and select the mode
    # Mode is at lower if cdf_lower >= pdf_at_mode AND cdf_lower >= sf_upper
    # Mode is at upper if sf_upper > cdf_lower AND sf_upper >= pdf_at_mode
    # Otherwise mode is at clipped_mode
    return pt.switch(
        pt.and_(pt.ge(cdf_lower, pdf_at_mode), pt.ge(cdf_lower, sf_upper)),
        lower_val,  # Mode at lower bound
        pt.switch(
            pt.and_(pt.gt(sf_upper, cdf_lower), pt.ge(sf_upper, pdf_at_mode)),
            upper_val,  # Mode at upper bound
            clipped_mode,  # Mode at base distribution's mode
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
    """
    Compute central moments of a censored distribution.

    Parameters
    ----------
    dist : module
        Base distribution module.
    lower : float or None
        Lower censoring bound.
    upper : float or None
        Upper censoring bound.
    *params : tensors
        Parameters for the base distribution.
    order : int
        Order of the central moment.
    mean_val : tensor
        Pre-computed mean value.
    n_points : int
        Number of integration points.

    Returns
    -------
    tensor
        Central moment of order `order`.
    """
    # Determine integration bounds
    if lower is None:
        int_lower = dist.ppf(0.0001, *params)
    else:
        int_lower = pt.as_tensor_variable(lower)

    if upper is None:
        int_upper = dist.ppf(0.9999, *params)
    else:
        int_upper = pt.as_tensor_variable(upper)

    # Point mass contributions
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

    # Interior integral using trapezoidal rule
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
    """
    Skewness of a censored distribution.

    Skewness is computed as E[(X - mean)^3] / std^3.

    Parameters
    ----------
    dist : module
        Base distribution module (must have pdf, cdf, sf, ppf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.
    n_points : int
        Number of integration points for numerical integration.

    Returns
    -------
    tensor
        Skewness value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> skewness_val = censored.skewness(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    Excess kurtosis of a censored distribution.

    Kurtosis is computed as E[(X - mean)^4] / var^2 - 3.

    Parameters
    ----------
    dist : module
        Base distribution module (must have pdf, cdf, sf, ppf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.
    n_points : int
        Number of integration points for numerical integration.

    Returns
    -------
    tensor
        Excess kurtosis value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> kurtosis_val = censored.kurtosis(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
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
    """
    Entropy of a censored distribution.

    The entropy is computed as:
    H = -CDF(L)*log(CDF(L)) - SF(U)*log(SF(U)) + integral(-pdf(x)*log(pdf(x)), L, U)

    This accounts for both the discrete point masses at the bounds and the
    continuous density in the interior.

    Parameters
    ----------
    dist : module
        Base distribution module (must have pdf, logpdf, cdf, sf, ppf).
    lower : float or None
        Lower censoring bound. None means no lower censoring (-inf).
    upper : float or None
        Upper censoring bound. None means no upper censoring (+inf).
    *params : tensors
        Parameters for the base distribution.
    n_points : int
        Number of integration points for numerical integration.

    Returns
    -------
    tensor
        Entropy value.

    Examples
    --------
    >>> from distributions import censored, normal
    >>> entropy_val = censored.entropy(normal, lower=-1.0, upper=1.0, mu=0.0, sigma=1.0)
    """
    # Determine integration bounds
    if lower is None:
        int_lower = dist.ppf(0.0001, *params)
    else:
        int_lower = pt.as_tensor_variable(lower)

    if upper is None:
        int_upper = dist.ppf(0.9999, *params)
    else:
        int_upper = pt.as_tensor_variable(upper)

    # Point mass entropy contributions: -p * log(p) for each point mass
    # Using the convention that 0 * log(0) = 0
    if lower is not None:
        lower_val = pt.as_tensor_variable(lower)
        cdf_lower = dist.cdf(lower_val, *params)
        # Use switch to handle the case when cdf_lower is 0
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
        # Use switch to handle the case when sf_upper is 0
        upper_entropy = pt.switch(
            pt.gt(sf_upper, 0.0),
            -sf_upper * pt.log(sf_upper),
            0.0,
        )
    else:
        upper_entropy = 0.0

    # Interior entropy: integral(-pdf(x) * log(pdf(x)), lower, upper)
    x_vals = pt.linspace(int_lower, int_upper, n_points)
    logpdf_vals = dist.logpdf(x_vals, *params)
    pdf_vals = pt.exp(logpdf_vals)

    # -pdf * log(pdf) = -pdf * logpdf
    integrand = -pdf_vals * logpdf_vals

    dx = (int_upper - int_lower) / (n_points - 1)
    interior_entropy = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1]) + 0.5 * integrand[-1])

    return lower_entropy + interior_entropy + upper_entropy
