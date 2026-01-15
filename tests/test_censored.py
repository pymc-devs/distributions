"""Test censored distribution modifier."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad

from distributions import censored, exponential, normal


class TestCensoredNormal:
    """Tests for censored normal distribution."""

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),  # Two-sided censoring
            (None, 1.0),  # Right-censored only
            (-1.0, None),  # Left-censored only
            (None, None),  # No censoring (should match base distribution)
        ],
    )
    def test_pdf_integrates_to_one(self, lower, upper):
        """Test that the PDF integrates to 1.

        For censored distributions, the total probability is:
        - Point mass at lower: CDF(lower) of base distribution
        - Continuous interior: integral of base PDF from lower to upper
        - Point mass at upper: SF(upper) = 1 - CDF(upper) of base distribution

        Total = CDF(lower) + (CDF(upper) - CDF(lower)) + (1 - CDF(upper)) = 1
        """
        mu, sigma = 0.0, 1.0

        # Define integration bounds for the continuous part
        int_lower = lower if lower is not None else -10.0
        int_upper = upper if upper is not None else 10.0

        # For censored distributions, we need to integrate the continuous interior
        # and add the point masses separately
        eps = 1e-8

        # Integrate the continuous interior (excluding boundary points)
        if lower is not None and upper is not None:
            # Two-sided: integrate strictly between bounds
            continuous_integral, _ = quad(
                lambda x: normal.pdf(x, mu, sigma).eval(),
                int_lower + eps,
                int_upper - eps,
            )
            # Add point masses
            point_mass_lower = normal.cdf(lower, mu, sigma).eval()
            point_mass_upper = normal.sf(upper, mu, sigma).eval()
            result = continuous_integral + point_mass_lower + point_mass_upper
        elif lower is not None:
            # Left-censored only
            continuous_integral, _ = quad(
                lambda x: normal.pdf(x, mu, sigma).eval(),
                int_lower + eps,
                int_upper,
            )
            point_mass_lower = normal.cdf(lower, mu, sigma).eval()
            result = continuous_integral + point_mass_lower
        elif upper is not None:
            # Right-censored only
            continuous_integral, _ = quad(
                lambda x: normal.pdf(x, mu, sigma).eval(),
                int_lower,
                int_upper - eps,
            )
            point_mass_upper = normal.sf(upper, mu, sigma).eval()
            result = continuous_integral + point_mass_upper
        else:
            # No censoring
            result, _ = quad(
                lambda x: normal.pdf(x, mu, sigma).eval(),
                int_lower,
                int_upper,
            )

        assert_allclose(result, 1.0, rtol=1e-3)

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_samples_within_bounds(self, lower, upper):
        """Test that random samples are within censoring bounds."""
        mu, sigma = 0.0, 1.0
        rng = pt.random.default_rng(42)

        samples = censored.rvs(
            normal, lower, upper, mu, sigma, size=(10000,), random_state=rng
        ).eval()

        if lower is not None:
            assert np.all(samples >= lower), f"Samples below lower bound: {samples.min()}"
        if upper is not None:
            assert np.all(samples <= upper), f"Samples above upper bound: {samples.max()}"

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_cdf_bounds(self, lower, upper):
        """Test CDF behavior at boundaries."""
        mu, sigma = 0.0, 1.0

        if lower is not None:
            # CDF below lower bound should be 0
            cdf_below = censored.cdf(lower - 1.0, normal, lower, upper, mu, sigma).eval()
            assert_allclose(cdf_below, 0.0, atol=1e-10)

            # CDF at lower bound should equal base CDF at lower
            cdf_at_lower = censored.cdf(lower, normal, lower, upper, mu, sigma).eval()
            base_cdf_at_lower = normal.cdf(lower, mu, sigma).eval()
            assert_allclose(cdf_at_lower, base_cdf_at_lower, rtol=1e-6)

        if upper is not None:
            # CDF at or above upper bound should be 1
            cdf_at_upper = censored.cdf(upper, normal, lower, upper, mu, sigma).eval()
            assert_allclose(cdf_at_upper, 1.0, atol=1e-10)

            cdf_above = censored.cdf(upper + 1.0, normal, lower, upper, mu, sigma).eval()
            assert_allclose(cdf_above, 1.0, atol=1e-10)

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_cdf_monotonic(self, lower, upper):
        """Test that CDF is monotonically increasing."""
        mu, sigma = 0.0, 1.0

        x_lower = lower if lower is not None else -5.0
        x_upper = upper if upper is not None else 5.0
        x_vals = np.linspace(x_lower - 1, x_upper + 1, 100)

        cdf_vals = censored.cdf(x_vals, normal, lower, upper, mu, sigma).eval()
        diffs = np.diff(cdf_vals)

        assert np.all(diffs >= -1e-10), "CDF is not monotonic"

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_sf_cdf_complement(self, lower, upper):
        """Test that SF + CDF = 1."""
        mu, sigma = 0.0, 1.0
        x_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        cdf_vals = censored.cdf(x_vals, normal, lower, upper, mu, sigma).eval()
        sf_vals = censored.sf(x_vals, normal, lower, upper, mu, sigma).eval()

        assert_allclose(cdf_vals + sf_vals, 1.0, rtol=1e-6)

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_logpdf_pdf_consistency(self, lower, upper):
        """Test that exp(logpdf) == pdf."""
        mu, sigma = 0.0, 1.0
        x_vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])

        pdf_vals = censored.pdf(x_vals, normal, lower, upper, mu, sigma).eval()
        logpdf_vals = censored.logpdf(x_vals, normal, lower, upper, mu, sigma).eval()

        # Handle -inf in logpdf
        finite_mask = np.isfinite(logpdf_vals)
        assert_allclose(np.exp(logpdf_vals[finite_mask]), pdf_vals[finite_mask], rtol=1e-6)
        assert_allclose(pdf_vals[~finite_mask], 0.0, atol=1e-10)

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_logcdf_cdf_consistency(self, lower, upper):
        """Test that exp(logcdf) == cdf."""
        mu, sigma = 0.0, 1.0
        x_vals = np.array([-0.5, 0.0, 0.5])

        cdf_vals = censored.cdf(x_vals, normal, lower, upper, mu, sigma).eval()
        logcdf_vals = censored.logcdf(x_vals, normal, lower, upper, mu, sigma).eval()

        assert_allclose(np.exp(logcdf_vals), cdf_vals, rtol=1e-6)

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_logsf_sf_consistency(self, lower, upper):
        """Test that exp(logsf) == sf."""
        mu, sigma = 0.0, 1.0
        x_vals = np.array([-0.5, 0.0, 0.5])

        sf_vals = censored.sf(x_vals, normal, lower, upper, mu, sigma).eval()
        logsf_vals = censored.logsf(x_vals, normal, lower, upper, mu, sigma).eval()

        assert_allclose(np.exp(logsf_vals), sf_vals, rtol=1e-6)

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_ppf_within_bounds(self, lower, upper):
        """Test that PPF returns values within bounds."""
        mu, sigma = 0.0, 1.0
        q_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

        ppf_vals = censored.ppf(q_vals, normal, lower, upper, mu, sigma).eval()

        if lower is not None:
            assert np.all(ppf_vals >= lower), f"PPF below lower bound: {ppf_vals.min()}"
        if upper is not None:
            assert np.all(ppf_vals <= upper), f"PPF above upper bound: {ppf_vals.max()}"

    @pytest.mark.parametrize(
        "lower, upper",
        [
            (-1.0, 1.0),
            (None, 1.0),
            (-1.0, None),
        ],
    )
    def test_isf_within_bounds(self, lower, upper):
        """Test that ISF returns values within bounds."""
        mu, sigma = 0.0, 1.0
        q_vals = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

        isf_vals = censored.isf(q_vals, normal, lower, upper, mu, sigma).eval()

        if lower is not None:
            assert np.all(isf_vals >= lower), f"ISF below lower bound: {isf_vals.min()}"
        if upper is not None:
            assert np.all(isf_vals <= upper), f"ISF above upper bound: {isf_vals.max()}"


class TestCensoredExponential:
    """Tests for censored exponential distribution (one-sided)."""

    @pytest.mark.parametrize(
        "upper",
        [1.0, 2.0, 5.0],
    )
    def test_right_censored_samples(self, upper):
        """Test right-censored exponential samples are within bounds."""
        lam = 1.0
        rng = pt.random.default_rng(42)

        samples = censored.rvs(exponential, 0.0, upper, lam, size=(10000,), random_state=rng).eval()

        assert np.all(samples >= 0.0), f"Samples below 0: {samples.min()}"
        assert np.all(samples <= upper), f"Samples above upper bound: {samples.max()}"

    @pytest.mark.parametrize(
        "upper",
        [1.0, 2.0, 5.0],
    )
    def test_right_censored_cdf(self, upper):
        """Test right-censored exponential CDF."""
        lam = 1.0

        # CDF at upper should be 1
        cdf_at_upper = censored.cdf(upper, exponential, 0.0, upper, lam).eval()
        assert_allclose(cdf_at_upper, 1.0, atol=1e-10)

        # CDF inside bounds should match base distribution
        x_inside = upper / 2
        cdf_inside = censored.cdf(x_inside, exponential, 0.0, upper, lam).eval()
        base_cdf_inside = exponential.cdf(x_inside, lam).eval()
        assert_allclose(cdf_inside, base_cdf_inside, rtol=1e-6)


class TestCensoredEdgeCases:
    """Tests for edge cases in censored distributions."""

    def test_no_censoring_matches_base(self):
        """Test that no censoring (None bounds) matches base distribution."""
        mu, sigma = 0.0, 1.0
        x_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # PDF should match
        censored_pdf = censored.pdf(x_vals, normal, None, None, mu, sigma).eval()
        base_pdf = normal.pdf(x_vals, mu, sigma).eval()
        assert_allclose(censored_pdf, base_pdf, rtol=1e-6)

        # CDF should match
        censored_cdf = censored.cdf(x_vals, normal, None, None, mu, sigma).eval()
        base_cdf = normal.cdf(x_vals, mu, sigma).eval()
        assert_allclose(censored_cdf, base_cdf, rtol=1e-6)

    def test_pdf_zero_outside_bounds(self):
        """Test that PDF is 0 outside censoring bounds."""
        mu, sigma = 0.0, 1.0
        lower, upper = -1.0, 1.0

        pdf_below = censored.pdf(-2.0, normal, lower, upper, mu, sigma).eval()
        pdf_above = censored.pdf(2.0, normal, lower, upper, mu, sigma).eval()

        assert_allclose(pdf_below, 0.0, atol=1e-10)
        assert_allclose(pdf_above, 0.0, atol=1e-10)

    def test_logpdf_neginf_outside_bounds(self):
        """Test that logpdf is -inf outside censoring bounds."""
        mu, sigma = 0.0, 1.0
        lower, upper = -1.0, 1.0

        logpdf_below = censored.logpdf(-2.0, normal, lower, upper, mu, sigma).eval()
        logpdf_above = censored.logpdf(2.0, normal, lower, upper, mu, sigma).eval()

        assert logpdf_below == -np.inf
        assert logpdf_above == -np.inf

    def test_point_mass_at_bounds(self):
        """Test point masses at censoring bounds."""
        mu, sigma = 0.0, 1.0
        lower, upper = -1.0, 1.0

        # PDF at lower bound should equal CDF(lower) of base distribution
        pdf_at_lower = censored.pdf(lower, normal, lower, upper, mu, sigma).eval()
        base_cdf_at_lower = normal.cdf(lower, mu, sigma).eval()
        assert_allclose(pdf_at_lower, base_cdf_at_lower, rtol=1e-6)

        # PDF at upper bound should equal SF(upper) of base distribution
        pdf_at_upper = censored.pdf(upper, normal, lower, upper, mu, sigma).eval()
        base_sf_at_upper = normal.sf(upper, mu, sigma).eval()
        assert_allclose(pdf_at_upper, base_sf_at_upper, rtol=1e-6)

    def test_broadcasting(self):
        """Test that censoring works with broadcasted parameters."""
        mu = np.array([0.0, 1.0, 2.0])
        sigma = 1.0
        lower, upper = -1.0, 3.0
        x = 0.5

        pdf_vals = censored.pdf(x, normal, lower, upper, mu, sigma).eval()
        assert pdf_vals.shape == (3,)

    def test_sample_empirical_mean(self):
        """Test that samples have approximately correct mean (within bounds)."""
        mu, sigma = 0.0, 1.0
        lower, upper = -1.0, 1.0
        rng = pt.random.default_rng(123)

        samples = censored.rvs(
            normal, lower, upper, mu, sigma, size=(50000,), random_state=rng
        ).eval()

        # For symmetric censoring around 0, mean should be close to 0
        assert np.abs(samples.mean()) < 0.05, f"Sample mean {samples.mean()} too far from 0"
