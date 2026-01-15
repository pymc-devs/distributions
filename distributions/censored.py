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
