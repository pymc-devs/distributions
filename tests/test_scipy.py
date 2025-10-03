import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_almost_equal
from scipy import stats

from distributions import beta as Beta
from distributions import betascaled as BetaScaled
from distributions import normal as Normal



@pytest.mark.parametrize(
    "p_dist, sp_dist, p_params, sp_params, support",
    [
    (Beta, stats.beta, (pt.constant(2.), pt.constant(5.)), (2, 5), (0, 1)),
    (BetaScaled, stats.beta, (pt.constant(2.), pt.constant(5.), pt.constant(-1.), pt.constant(3.)), (2, 5, -1, 4), (-1, 3)),
    (Normal, stats.norm, (pt.constant(0.), pt.constant(2.)), (0, 2), (-np.inf, np.inf)),
    ],
)
def test_match_scipy(p_dist, sp_dist, p_params, sp_params, support):
    scipy_dist = sp_dist(*sp_params)

    # Entropy
    actual = p_dist.entropy(*p_params).eval()
    expected = scipy_dist.entropy()
    assert_almost_equal(actual, expected, decimal=4)

    # Random variates
    rng = pt.random.default_rng(1)
    actual_rvs = p_dist.rvs(*p_params, size=20, random_state=rng).eval()
    rng = np.random.default_rng(1)
    expected_rvs = scipy_dist.rvs(20, random_state=rng)
    assert_almost_equal(actual_rvs, expected_rvs)

    extended_vals = np.concatenate(
            [
                actual_rvs,
                support,
                [support[0] - 1],
                [support[0] - 2],
                [support[1] + 1],
                [support[1] + 2],
            ]
        )

    # PDF
    actual_pdf = p_dist.pdf(extended_vals, *p_params).eval()
    expected_pdf = scipy_dist.pdf(extended_vals)
    assert_almost_equal(actual_pdf, expected_pdf, decimal=4)

    # logPDF
    actual_logpdf = p_dist.logpdf(extended_vals, *p_params).eval()
    expected_logpdf = scipy_dist.logpdf(extended_vals)
    assert_almost_equal(actual_logpdf, expected_logpdf)

    # CDF
    actual_cdf = p_dist.cdf(extended_vals, *p_params).eval()
    expected_cdf = scipy_dist.cdf(extended_vals)
    assert_almost_equal(actual_cdf, expected_cdf, decimal=6)

    # logCDF
    actual_logcdf = p_dist.logcdf(extended_vals, *p_params).eval()
    expected_logcdf = scipy_dist.logcdf(extended_vals)
    assert_almost_equal(actual_logcdf, expected_logcdf, decimal=5)

    # SF
    actual_sf = p_dist.sf(extended_vals, *p_params).eval()
    expected_sf = scipy_dist.sf(extended_vals)
    assert_almost_equal(actual_sf, expected_sf, decimal=6)

    # logSF
    actual_logsf = p_dist.logsf(extended_vals, *p_params).eval()
    expected_logsf = scipy_dist.logsf(extended_vals)
    assert_almost_equal(actual_logsf, expected_logsf, decimal=4)

    # PPF
    x_vals = np.array([-1, 0, 0.25, 0.5, 0.75, 1, 2])
    actual_ppf = p_dist.ppf(x_vals, *p_params).eval()
    expected_ppf = scipy_dist.ppf(x_vals)
    assert_almost_equal(actual_ppf, expected_ppf)

    # ISF
    actual_isf = p_dist.isf(x_vals, *p_params).eval()
    expected_isf = scipy_dist.isf(x_vals)
    assert_almost_equal(actual_isf, expected_isf)

    # mean
    mean = p_dist.mean(*p_params).eval()
    expected_mean = scipy_dist.mean()
    assert_almost_equal(mean, expected_mean)

    # median
    median = p_dist.median(*p_params).eval()
    expected_median = scipy_dist.median()
    assert_almost_equal(median, expected_median)

    # mode
    finite_expected_pdf = np.where(np.isfinite(expected_pdf), expected_pdf, -np.inf)
    expected_mode = extended_vals[np.argmax(finite_expected_pdf)]
    actual_mode = p_dist.mode(*p_params).eval()
    assert_almost_equal(actual_mode, expected_mode, decimal=0)

    # standard deviation   
    std = p_dist.std(*p_params).eval()
    expected_std = scipy_dist.std()
    assert_almost_equal(std, expected_std)

    # variance
    var = p_dist.var(*p_params).eval()
    expected_var = scipy_dist.var()
    assert_almost_equal(var, expected_var)

    # skewness
    skewness = p_dist.skewness(*p_params).eval()
    expected_skewness = scipy_dist.stats(moments="s")
    assert_almost_equal(skewness, expected_skewness)

    # kurtosis
    kurtosis = p_dist.kurtosis(*p_params).eval()
    expected_kurtosis = scipy_dist.stats(moments="k")
    assert_almost_equal(kurtosis, expected_kurtosis)
