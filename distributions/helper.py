import pytensor.tensor as pt
from pytensor.scan import scan


def cdf_bounds(prob, x, lower, upper):
    """
    Apply bounds checking for cumulative distribution function (CDF).

    Parameters
    ----------
    prob : tensor
        The computed CDF probability value
    x : tensor
        Input value to evaluate CDF at
    lower : float
        Lower bound of the distribution support
    upper : float
        Upper bound of the distribution support

    Returns
    -------
    tensor
        CDF value with proper bounds: 0.0 for x < lower,
        1.0 for x >= upper, otherwise the computed prob value
    """
    x = pt.as_tensor_variable(x)
    return pt.switch(pt.lt(x, lower), 0.0, pt.switch(pt.ge(x, upper), 1.0, prob))


def ppf_bounds_cont(x_val, q, lower, upper):
    """
    Apply bounds checking for the inverse CDF of continuous distributions.

    Parameters
    ----------
    x_val : tensor
        The computed PPF value
    q : tensor
        Probability value (quantile) between 0 and 1
    lower : float
        Lower bound of the distribution support
    upper : float
        Upper bound of the distribution support

    Returns
    -------
    tensor
        PPF value with proper bounds: NaN for q outside [0,1],
        lower bound for q=0, upper bound for q=1, otherwise x_val
    """
    return pt.switch(
        pt.or_(pt.lt(q, 0), pt.gt(q, 1)),
        pt.nan,
        pt.switch(pt.eq(q, 0), lower, pt.switch(pt.eq(q, 1), upper, x_val)),
    )


def ppf_bounds_disc(x_val, q, lower, upper):
    """
    Apply bounds checking for the inverse CDF of discrete distributions.

    Parameters
    ----------
    x_val : tensor
        The computed PPF value
    q : tensor
        Probability value (quantile) between 0 and 1
    lower : int
        Lower bound of the distribution support
    upper : int
        Upper bound of the distribution support

    Returns
    -------
    tensor
        PPF value with proper bounds: NaN for q outside [0,1],
        lower-1 for q=0 (since discrete PPF at 0 is below support),
        upper for q=1, otherwise x_val
    """
    return pt.switch(
        pt.lt(q, 0),
        pt.nan,
        pt.switch(
            pt.gt(q, 1),
            pt.nan,
            pt.switch(pt.eq(q, 0), lower - 1, pt.switch(pt.eq(q, 1), upper, x_val)),
        ),
    )


def sf_bounds(x_val, q, lower, upper):
    return pt.switch(pt.lt(q, lower), 1.0, pt.switch(pt.gt(q, upper), 0.0, x_val))


def logdiffexp(a, b):
    """Return log(exp(a) - exp(b))."""
    return a + pt.log1mexp(b - a)


def continuous_entropy(min_x, max_x, logpdf_func, *params):
    """
    Compute entropy for continuous distributions using numerical integration.

    Parameters
    ----------
    min_x : float
        Minimum value for integration
    max_x : float
        Maximum value for integration
    logpdf_func : function
        Log probability density function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to logpdf_func

    Returns
    -------
    entropy : tensor
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    x_values = pt.linspace(min_x, max_x, 1000)
    x_broadcast = x_values.reshape((-1,) + (1,) * broadcast_shape.ndim)

    logpdf_vals = logpdf_func(x_broadcast, *params)
    pdf_vals = pt.exp(logpdf_vals)

    integrand = -pdf_vals * logpdf_vals

    dx = (max_x - min_x) / (1000 - 1)
    result = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1], axis=0) + 0.5 * integrand[-1])

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def discrete_entropy(min_x, max_x, logpdf, *params):
    """
    Compute entropy for discrete distributions by explicit summation.

    Parameters
    ----------
    min_x : int or float
        Minimum value to sum over
    max_x : int or float
        Maximum value to sum over
    logpdf : function
        Log probability mass function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to logpdf

    Returns
    -------
    entropy : tensor
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    k_vals = pt.arange(min_x, max_x)
    k_broadcast = k_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)

    log_probs = logpdf(k_broadcast, *params)

    result = pt.sum(-pt.exp(log_probs) * log_probs, axis=0)

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def discrete_moment(ppf, pdf, *params, order=1):
    """
    Compute central moments for discrete distributions by explicit summation.

    Parameters
    ----------
    ppf : function
        Percent-point function that takes (p, *params) as arguments
    pdf : function
        Probability mass function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to pdf
    order : int
        Order of the moment to compute

    Returns
    -------
    moment : tensor
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    min_x = ppf(0.0001, *params).min()
    max_x = ppf(0.9999, *params).max() + 1

    k_vals = pt.arange(min_x, max_x)
    k_broadcast = k_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)
    pdf_vals = pdf(k_broadcast, *params)

    result = pt.sum(k_broadcast * pdf_vals, axis=0)
    if order > 1:
        result = pt.sum((k_broadcast - result) ** order * pdf_vals, axis=0)

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def discrete_mean(ppf, pdf, *params):
    """Compute mean for discrete distributions."""
    return discrete_moment(ppf, pdf, *params, order=1)


def discrete_variance(ppf, pdf, *params):
    """Compute variance for discrete distributions."""
    return discrete_moment(ppf, pdf, *params, order=2)


def discrete_skewness(ppf, pdf, *params):
    """Compute skewness for discrete distributions."""
    variance = discrete_moment(ppf, pdf, *params, order=2)
    third_moment = discrete_moment(ppf, pdf, *params, order=3)
    return third_moment / (variance**1.5)


def discrete_kurtosis(ppf, pdf, *params):
    """Compute kurtosis for discrete distributions."""
    variance = discrete_moment(ppf, pdf, *params, order=2)
    fourth_moment = discrete_moment(ppf, pdf, *params, order=4)
    result = fourth_moment / (variance**2) - 3
    return result


def continuous_moment(lower, upper, logpdf, *params, order=1, mean_val=None, n_points=1000):
    """
    Compute raw or central moments for continuous distributions using numerical integration.

    Uses the trapezoidal rule for numerical integration.

    Parameters
    ----------
    lower : float
        Lower bound for integration
    upper : float
        Upper bound for integration
    logpdf : function
        Log probability density function that takes (x, *params) as arguments
    *params : tensor variables
        Distribution parameters to pass to logpdf
    order : int
        Order of the moment to compute
    mean_val : tensor, optional
        If provided, computes central moment around this mean.
        If None, computes raw moment.
    n_points : int
        Number of integration points

    Returns
    -------
    moment : tensor
    """
    if len(params) == 1:
        broadcast_shape = pt.as_tensor_variable(params[0])
    else:
        broadcast_shape = pt.broadcast_arrays(*params)[0]

    x_vals = pt.linspace(lower, upper, n_points)
    x_broadcast = x_vals.reshape((-1,) + (1,) * broadcast_shape.ndim)
    pdf_vals = pt.exp(logpdf(x_broadcast, *params))

    if mean_val is not None:
        # Central moment
        integrand = (x_broadcast - mean_val) ** order * pdf_vals
    else:
        # Raw moment
        integrand = x_broadcast**order * pdf_vals

    dx = (upper - lower) / (n_points - 1)
    result = dx * (0.5 * integrand[0] + pt.sum(integrand[1:-1], axis=0) + 0.5 * integrand[-1])

    return pt.squeeze(result) if broadcast_shape.ndim == 0 else result


def continuous_mean(lower, upper, logpdf, *params):
    """Compute mean for continuous distributions."""
    return continuous_moment(lower, upper, logpdf, *params, order=1)


def continuous_variance(lower, upper, logpdf, *params):
    """Compute variance for continuous distributions."""
    mean_val = continuous_moment(lower, upper, logpdf, *params, order=1)
    return continuous_moment(lower, upper, logpdf, *params, order=2, mean_val=mean_val)


def continuous_skewness(lower, upper, logpdf, *params):
    """Compute skewness for continuous distributions."""
    mean_val = continuous_moment(lower, upper, logpdf, *params, order=1)
    variance = continuous_moment(lower, upper, logpdf, *params, order=2, mean_val=mean_val)
    third_central = continuous_moment(lower, upper, logpdf, *params, order=3, mean_val=mean_val)
    return third_central / (pt.sqrt(variance) ** 3)


def continuous_kurtosis(lower, upper, logpdf, *params):
    """Compute excess kurtosis for continuous distributions."""
    mean_val = continuous_moment(lower, upper, logpdf, *params, order=1)
    variance = continuous_moment(lower, upper, logpdf, *params, order=2, mean_val=mean_val)
    fourth_central = continuous_moment(lower, upper, logpdf, *params, order=4, mean_val=mean_val)
    return fourth_central / (variance**2) - 3


def from_tau(tau):
    """Convert precision (tau) to standard deviation (sigma)."""
    sigma = 1 / pt.sqrt(tau)
    return sigma


def to_tau(sigma):
    """Convert standard deviation (sigma) to precision (tau)."""
    tau = pt.power(sigma, -2)
    return tau


def von_mises_cdf_series(kappa, x, p):
    """
    Compute von Mises CDF using series expansion.

    Parameters
    ----------
    kappa : tensor
        Concentration parameter
    x : tensor
        Angle centered at mu (after wrapping to [-pi, pi])
    p : tensor (int)
        Number of terms in series

    Returns
    -------
    tensor
        CDF value
    """

    def series_step(n, cumsum, kappa, x):
        bessel_ratio = pt.ive(n, kappa) / pt.ive(0, kappa)
        term = bessel_ratio * pt.sin(n * x) / n
        return cumsum + term

    n_values = pt.arange(1, p + 1, dtype="float64")
    init_sum = pt.zeros_like(x)

    results, _ = scan(
        fn=lambda n, cs: series_step(n, cs, kappa, x), sequences=[n_values], outputs_info=[init_sum]
    )
    series_sum = results[-1]
    result = 0.5 + x / (2 * pt.pi) + series_sum / pt.pi

    return result


def von_mises_cdf_normalapprox(kappa, x):
    """
    Compute von Mises CDF using normal approximation for large kappa.

    Parameters
    ----------
    kappa : tensor
        Concentration parameter (large values)
    x : tensor
        Angle centered at mu

    Returns
    -------
    tensor
        CDF value using normal approximation
    """
    b = pt.sqrt(2 / pt.pi) / pt.ive(0.0, kappa)
    z = b * pt.sin(x / 2.0)
    return 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))


def von_mises_cdf(x, mu, kappa):
    x_centered = x - mu
    ix = pt.round(x_centered / (2 * pt.pi))
    x_wrapped = x_centered - ix * 2 * pt.pi

    # Constants from scipy implementation
    CK = 50.0  # Threshold for switching between methods
    a1, a2, a3, a4 = 28.0, 0.5, 100.0, 5.0

    # Compute number of terms for series expansion
    p = pt.cast(pt.round(1 + a1 + a2 * kappa - a3 / (kappa + a4)), "int32")

    use_series = kappa < CK

    cdf_series = von_mises_cdf_series(kappa, x_wrapped, p)
    cdf_series = pt.clip(cdf_series, 0.0, 1.0)
    cdf_norm = von_mises_cdf_normalapprox(kappa, x_wrapped)

    result = pt.switch(use_series, cdf_series, cdf_norm)
    result = result + ix

    return result


def zi_mode(base_mode, logpdf, *params):
    """
    Compute mode for zero-inflated distributions.

    Compares probability at x=0 vs probability at the base distribution's mode,
    and returns whichever has higher probability.

    Parameters
    ----------
    base_mode : tensor
        The mode of the underlying (non-zero-inflated) distribution
    logpdf : function
        Log probability function of the ZI distribution that takes (x, *params)
    *params : tensor variables
        Parameters to pass to logpdf (typically psi, plus base distribution params)

    Returns
    -------
    tensor
        The mode value (either 0 or base_mode)
    """
    return pt.switch(logpdf(0, *params) >= logpdf(base_mode, *params), 0, base_mode)
