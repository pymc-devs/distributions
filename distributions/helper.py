import pytensor.tensor as pt


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


def from_tau(tau):
    """Convert precision (tau) to standard deviation (sigma)."""
    sigma = 1 / pt.sqrt(tau)
    return sigma


def to_tau(sigma):
    """Convert standard deviation (sigma) to precision (tau)."""
    tau = pt.power(sigma, -2)
    return tau
