import numpy as np
import pytest
import pytensor.tensor as pt

from distributions import skew_studentt as SkewStudentT
from tests.helper_empirical import run_empirical_tests
from tests.helper_scipy import make_params


@pytest.mark.parametrize(
    "params",
    [
        [10.0, 2.0, 0.0, 1.0],  # Positive skew
        [10.0, -2.0, 0.0, 1.0],  # Negative skew
        [5.0, 3.0, 2.0, 2.0],  # Shifted and scaled
    ],
)
def test_skew_studentt_empirical(params):
    p_params = make_params(*params, dtype="float64")
    run_empirical_tests(
        SkewStudentT,
        p_params,
        support=[-np.inf, np.inf],
        name="skew_studentt",
        sample_size=100_000,
        mean_rtol=1e-2,
        var_rtol=2e-2,
        quantiles_rtol=2e-2,
    )


def test_skew_studentt_symmetric():
    """Test that alpha=0 gives symmetric t-distribution."""
    nu, alpha, mu, sigma = 10.0, 0.0, 0.0, 1.0
    p_params = make_params(nu, alpha, mu, sigma, dtype="float64")

    # Mean should be mu
    mean_val = SkewStudentT.mean(*p_params).eval()
    np.testing.assert_allclose(mean_val, 0.0, atol=1e-10)

    # Variance should be nu/(nu-2) * sigma^2
    var_val = SkewStudentT.var(*p_params).eval()
    expected_var = 10.0 / 8.0
    np.testing.assert_allclose(var_val, expected_var, rtol=1e-6)

    # PDF at 0 should match t-distribution
    pdf_at_zero = SkewStudentT.pdf(0.0, *p_params).eval()
    # t(0; nu=10) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2))
    from scipy import stats

    expected_pdf = stats.t.pdf(0.0, df=10)
    np.testing.assert_allclose(pdf_at_zero, expected_pdf, rtol=1e-4)


def test_skew_studentt_boundary_cdf():
    """Test CDF at boundary values."""
    nu, alpha, mu, sigma = 10.0, 2.0, 0.0, 1.0
    p_params = make_params(nu, alpha, mu, sigma, dtype="float64")

    # CDF at -inf should be 0
    cdf_neg_inf = SkewStudentT.cdf(-np.inf, *p_params).eval()
    np.testing.assert_equal(cdf_neg_inf, 0.0)

    # CDF at +inf should be 1
    cdf_pos_inf = SkewStudentT.cdf(np.inf, *p_params).eval()
    np.testing.assert_equal(cdf_pos_inf, 1.0)
