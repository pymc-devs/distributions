"""Parent classes for all families."""

import warnings
from collections import namedtuple
import numpy as np

from preliz.internal.distribution_helper import valid_distribution, valid_scalar_params
from preliz.internal.optimization import find_hdi, find_mode
from preliz.internal.rcparams import rcParams


class Distribution:
    """
    Base class for distributions.

    Not intended for direct instantiation.
    """

    def __init__(self):
        self.is_frozen = False
        self.opt = None

    def __repr__(self):
        name = self.__class__.__name__
        if name in ["Truncated", "Censored", "Hurdle"]:
            name += self.dist.__class__.__name__
        if name == "Mixture":
            name = (
                "Mixture"
                + "".join(dict.fromkeys(dist.__class__.__name__ for dist in self.dist))
                + "\n"
            )

        if self.is_frozen:
            if "Mixture" in name:
                bolded_name = "\033[1m" + name.strip() + "\033[0m" + "\n"
            else:
                bolded_name = "\033[1m" + name + "\033[0m"

            description = "".join(
                f"{n}={v:.3g}, "
                if np.isscalar(v)
                else f"{n}=[" + "".join(f"{vi:.3g}, " for vi in v).strip(", ") + "], "
                for n, v in zip(self.param_names, self.params)
            ).strip(", ")
            return f"{bolded_name}({description})"
        else:
            return name

    @property
    def params_dict(self):
        if self.is_frozen:
            return dict(zip(self.param_names, self.params))
        else:
            return None

    def summary(self, mass=None, interval=None, fmt=".2f"):
        """
        Namedtuple with the mean, median, sd, and lower and upper bounds.

        Parameters
        ----------
        mass: float
            Probability mass for the equal-tailed interval. Defaults to None,
        which results in the value of rcParams["stats.ci_prob"] being used.
        interval : str or list-like
            Type of interval. Available options are highest density interval `"hdi"`,
            equal tailed interval `"eti"` or arbitrary interval defined by a list-like object
            with a pair of values. Defaults to the value in rcParams["stats.ci_kind"].
        fmt : str
            fmt used to represent results using f-string fmt for floats. Default to ".2f"
            i.e. 2 digits after the decimal point.
        """
        valid_distribution(self)

        if interval is None:
            interval = rcParams["stats.ci_kind"]

        if not isinstance(fmt, str):
            raise ValueError("Invalid format string.")

        if valid_scalar_params(self):
            name = self.__class__.__name__
            if name == "Truncated":
                name = "Truncated" + self.dist.__class__.__name__
            elif name == "Censored":
                name = "Censored" + self.dist.__class__.__name__
            elif name == "Mixture":
                name = "Mixture" + "".join(
                    dict.fromkeys(dist.__class__.__name__ for dist in self.dist)
                )

            attr = namedtuple(name, ["mean", "median", "std", "lower", "upper"])
            mean = float(f"{self.mean():{fmt}}")
            median = float(f"{self.median():{fmt}}")
            std = float(f"{self.std():{fmt}}")

            if isinstance(interval, tuple | list | np.ndarray):
                c_int = self.ppf(interval)
            elif interval == "hdi":
                c_int = self.hdi(mass, fmt=fmt)
            elif interval == "eti":
                c_int = self.eti(mass, fmt=fmt)

            if self.kind == "discrete":
                lower_tail = c_int[0]
                upper_tail = c_int[1]
            else:
                lower_tail = float(f"{c_int[0]:{fmt}}")
                upper_tail = float(f"{c_int[1]:{fmt}}")
            return attr(mean, median, std, lower_tail, upper_tail)
        else:
            return None

    def rvs(self, size=None, random_state=None):
        """Random sample.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates. Defaults to 1.
        random_state : {None, int, numpy.random.Generator, numpy.random.RandomState}
            Defaults to None
        """
        raise NotImplementedError

    def pdf(self, x):
        """Probability density/mass function.

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the pdf
        """
        raise NotImplementedError

    def cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the cdf
        """
        raise NotImplementedError

    def ppf(self, q):
        """Percent point function (inverse of cdf).

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the inverse of the cdf
        """
        raise NotImplementedError

    def logpdf(self, x):
        """Log probability density/mass function.

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the logpdf
        """
        raise NotImplementedError

    def logcdf(self, x):
        """Log cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the logcdf
        """
        return np.log(self.cdf(x))

    def sf(self, x):
        """Survival function (1 - cdf).

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the sf
        """
        return 1 - self.cdf(x)

    def logsf(self, x):
        """Log survival function log(1 - cdf).

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the logsf
        """
        return np.log(self.sf(x))

    def isf(self, x):
        """Inverse survival function (inverse of sf).

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the inverse of the sf
        """
        return self.ppf(1 - np.array(x))

    def entropy(self):
        """Entropy."""
        raise NotImplementedError

    def mean(self):
        """Mean."""
        raise NotImplementedError

    def median(self):
        """Median."""
        raise NotImplementedError

    def mode(self):
        """Mode."""
        raise NotImplementedError

    def std(self):
        """Standard deviation."""
        raise NotImplementedError

    def var(self):
        """Variance."""
        raise NotImplementedError

    def skewness(self):
        """Skewness."""
        raise NotImplementedError

    def kurtosis(self):
        """Excess kurtosis."""
        raise NotImplementedError

    def moments(self, types="mvsk"):
        """
        Compute moments of the distribution.

        It can also return the standard deviation

        Parameters
        ----------
        types : str
            The type of moments to compute. Default is 'mvsk'
            where 'm' = mean, 'v' = variance, 's' = skewness, and 'k' = kurtosis.
            To compute the standard deviation use 'd'
            Valid combinations are any subset of 'mvdsk'.
        """
        moments = []
        for m_t in types:
            if m_t not in "mdvsk":
                raise ValueError(
                    "The input string should only contain the letters "
                    "'m', 'd', 'v', 's', or 'k'."
                )
            if m_t == "m":
                moments.append(self.mean())
            elif m_t == "d":
                moments.append(self.std())
            elif m_t == "v":
                moments.append(self.var())
            elif m_t == "s":
                moments.append(self.skewness())
            elif m_t == "k":
                moments.append(self.kurtosis())

        return moments

    def eti(self, mass=None, fmt=".2f"):
        """Equal-tailed interval containing `mass`.

        Parameters
        ----------
        mass: float
            Probability mass in the interval. Defaults to None,
        which results in the value of rcParams["stats.ci_prob"] being used.
        fmt : str
            fmt used to represent results using f-string fmt for floats. Default to ".2f"
            i.e. 2 digits after the decimal point. Use `"none"` for no format.
        """
        valid_distribution(self)

        if mass is None:
            mass = rcParams["stats.ci_prob"]

        if not isinstance(fmt, str):
            raise ValueError("Invalid format string.")

        if valid_scalar_params(self):
            lower_tail, upper_tail = self.ppf([(1 - mass) / 2, 1 - (1 - mass) / 2])
            if self.kind == "continuous" and fmt != "none":
                lower_tail = float(f"{lower_tail:{fmt}}")
                upper_tail = float(f"{upper_tail:{fmt}}")
            elif self.kind == "discrete":
                lower_tail = int(lower_tail)
                upper_tail = int(upper_tail)

            return (lower_tail, upper_tail)
        else:
            return None

    def hdi(self, mass=None, fmt=".2f"):
        """Highest density interval containing `mass`.

        Parameters
        ----------
        mass: float
            Probability mass in the interval. Defaults to None,
        which results in the value of rcParams["stats.ci_prob"] being used.
        fmt : str
            fmt used to represent results using f-string fmt for floats. Default to ".2f"
            i.e. 2 digits after the decimal point. Use `"none"` for no format.
        """
        valid_distribution(self)

        if mass is None:
            mass = rcParams["stats.ci_prob"]

        if not isinstance(fmt, str):
            raise ValueError("Invalid format string.")

        if self.__class__.__name__ == "Mixture":
            warnings.warn("HDI may not be correct for multimodal distributions")

        if valid_scalar_params(self):
            lower_tail, upper_tail = find_hdi(self, mass)
            if self.kind == "continuous" and fmt != "none":
                lower_tail = float(f"{lower_tail:{fmt}}")
                upper_tail = float(f"{upper_tail:{fmt}}")
            return (lower_tail, upper_tail)
        else:
            return None

    def to_pymc(self, name=None, **kwargs):
        """
        Convert the PreliZ distribution to a PyMC distribution.

        name : str
            Name of PyMC distribution. Needed if inside Model context
        kwargs : PyMC distributions properties
            kwargs are used to specify properties such as shape or dims

        Returns
        -------
        PyMC distribution
        """
        try:
            import pymc.distributions as pm_dists
            from pymc.model import Model

            model = Model.get_context(error_if_none=False)

            if self.__class__.__name__ == "Hurdle":
                preliz_name = self.__class__.__name__ + self.dist.__class__.__name__
            else:
                preliz_name = self.__class__.__name__
            pymc_class = getattr(pm_dists, preliz_name)

            if model is None:
                if self.__class__.__name__ in ["Truncated", "Censored"]:
                    pymc_dist = pymc_class.dist(
                        self.dist.to_pymc(),
                        lower=self.params_dict["lower"],
                        upper=self.params_dict["upper"],
                        **kwargs,
                    )
                elif self.__class__.__name__ == "Mixture":
                    pymc_dist = pymc_class.dist(
                        self.weights,
                        [dist.to_pymc() for dist in self.dist],
                        **kwargs,
                    )
                else:
                    pymc_dist = pymc_class.dist(**self.params_dict, **kwargs)
            elif self.__class__.__name__ in ["Truncated", "Censored"]:
                pymc_dist = pymc_class(
                    name,
                    getattr(pm_dists, self.dist.__class__.__name__).dist(**self.dist.params_dict),
                    lower=self.params_dict["lower"],
                    upper=self.params_dict["upper"],
                    **kwargs,
                )
            elif self.__class__.__name__ == "Mixture":
                pymc_dist = pymc_class(
                    name,
                    self.weights,
                    [
                        getattr(pm_dists, dist.__class__.__name__).dist(**dist.params_dict)
                        for dist in self.dist
                    ],
                    **kwargs,
                )
            else:
                pymc_dist = pymc_class(name, **self.params_dict, **kwargs)

            return pymc_dist

        except ImportError:
            raise ImportError("This function requires PyMC") from None

    def to_bambi(self, **kwargs):
        """
        Convert the PreliZ distribution to a Bambi Prior.

        kwargs : PyMC distributions properties
            kwargs are used to specify properties such as shape or dims

        Returns
        -------
        Bambi Prior
        """
        try:
            from bambi import Prior

            return Prior(self.__class__.__name__, **self.params_dict, **kwargs)

        except ImportError:
            raise ImportError("This function requires Bambi") from None

    def _check_endpoints(self, lower, upper, raise_error=True):
        """
        Evaluate if the lower and upper values are in the support of the distribution.

        Parameters
        ----------
        lower : int or float
            lower endpoint
        upper : int or float
            upper endpoint
        raise_error : bool
            If True (default) it will raise ValueErrors, otherwise it will return True
            if the lower and upper endpoint are in the support of the distribution or
            False otherwise.
        """
        s_l, s_u = self.support

        if raise_error:
            domain_error = (
                f"The provided endpoints are outside the domain of the "
                f"{self.__class__.__name__} distribution"
            )

            if np.isfinite(s_l):
                if lower < s_l:
                    raise ValueError(domain_error)

            if np.isfinite(s_u):
                if upper > s_u:
                    raise ValueError(domain_error)
            if np.isfinite(s_l) and np.isfinite(s_u):
                if lower == s_l and upper == s_u:
                    raise ValueError(
                        "Given the provided endpoints, mass will be always 1. "
                        "Please provide other values"
                    )
            return None
        else:
            return lower >= s_l and upper <= s_u

    def _finite_endpoints(self, support):
        """
        Return finite endpoints even for unbounded distributions.

        Parameters
        ----------
        support : str or tuple
            Available string options are "full" or "restricted".
        """
        if isinstance(support, tuple):
            lower_ep, upper_ep = support
        else:
            if support not in ["restricted", "full"]:
                raise ValueError("Allowed values for the support are 'restricted' or 'full' ")

            lower_ep, upper_ep = self.support

            if not np.isfinite(lower_ep) or support == "restricted":
                lower_ep = self.ppf(0.0001)
            if not np.isfinite(upper_ep) or support == "restricted":
                upper_ep = self.ppf(0.9999)

        return lower_ep, upper_ep

    def xvals(self, support, n_points=None):
        """
        Provide x values in the support of the distribution.

        This is useful for example when plotting.

        Parameters
        ----------
        support : str
            Available options are `"full"` or `"restricted"`.
            If `"full"` the values will cover the entire support of the distribution if the boundary
            is finite, or the quantiles 0.0001 or 0.9999, if infinite.
            If `"restricted"` the values will cover the quantile 0.0001 to 0.9999.
        n_points : int
            Number of values to return. Defaults to 1000 for continuous distributions
            and 200 for discrete ones.
            For discrete distributions the returned values may be fewer
            than `n_points` if the actual number of discrete values in the support of the
            distribution is smaller than `n_points`.
        """
        lower_ep, upper_ep = self._finite_endpoints(support)

        if self.kind == "continuous":
            if n_points is None:
                n_points = 1000
            return _continuous_xvals(lower_ep, upper_ep, n_points)
        else:
            if n_points is None:
                n_points = 200
            return _discrete_xvals(lower_ep, upper_ep, n_points)


class Continuous(Distribution):
    """Base class for continuous distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "continuous"

    def mode(self):
        return find_mode(self)


class Discrete(Distribution):
    """Base class for discrete distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "discrete"


class DistributionTransformer(Distribution):
    """Base class for distributions that transform other distributions."""

    def __init__(self):
        super().__init__()
        if not isinstance(self.dist, list):
            self.kind = self.dist.kind


def _continuous_xvals(lower_ep, upper_ep, n_points):
    return np.linspace(lower_ep, upper_ep, n_points)


def _discrete_xvals(lower_ep, upper_ep, n_points):
    upper_ep = int(upper_ep)
    lower_ep = int(lower_ep)
    range_x = upper_ep - lower_ep
    if range_x <= n_points:
        x_vals = np.arange(lower_ep, upper_ep + 1, dtype=int)
    else:
        x_vals = np.linspace(lower_ep, upper_ep + 1, n_points, dtype=int)

    return x_vals