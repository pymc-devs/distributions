"""Tests for helper functions."""

import numpy as np
import pytensor.tensor as pt
from numpy.testing import assert_allclose

from distributions.helper import zi_mode


class TestZiMode:
    """Tests for zi_mode helper function."""

    def test_zi_mode_returns_zero_when_zero_has_higher_prob(self):
        """Test that zi_mode returns 0 when logpdf(0) > logpdf(base_mode)."""

        # Create a mock logpdf where P(0) > P(base_mode)
        def mock_logpdf(x, psi, mu):
            # ZI-Poisson-like: high zero inflation (low psi) means P(0) is high
            return pt.switch(pt.eq(x, 0), pt.log(0.8), pt.log(0.05))

        base_mode = pt.constant(3.0)
        psi = pt.constant(0.3)
        mu = pt.constant(5.0)

        result = zi_mode(base_mode, mock_logpdf, psi, mu).eval()
        assert result == 0.0

    def test_zi_mode_returns_base_mode_when_base_mode_has_higher_prob(self):
        """Test that zi_mode returns base_mode when logpdf(base_mode) > logpdf(0)."""

        # Create a mock logpdf where P(base_mode) > P(0)
        def mock_logpdf(x, psi, mu):
            # Low zero inflation (high psi) means base_mode has higher prob
            return pt.switch(pt.eq(x, 0), pt.log(0.1), pt.log(0.3))

        base_mode = pt.constant(3.0)
        psi = pt.constant(0.9)
        mu = pt.constant(5.0)

        result = zi_mode(base_mode, mock_logpdf, psi, mu).eval()
        assert result == 3.0

    def test_zi_mode_returns_zero_when_equal_prob(self):
        """Test that zi_mode returns 0 when probabilities are equal (>= condition)."""

        # Create a mock logpdf where P(0) == P(base_mode)
        def mock_logpdf(x, psi, mu):
            return pt.log(0.2)  # Same probability for all x

        base_mode = pt.constant(5.0)
        psi = pt.constant(0.5)
        mu = pt.constant(5.0)

        result = zi_mode(base_mode, mock_logpdf, psi, mu).eval()
        assert result == 0.0  # Should return 0 due to >= condition

    def test_zi_mode_with_zi_poisson(self):
        """Test zi_mode with actual ZI-Poisson distribution."""
        from distributions import zi_poisson as ZIPoisson

        # High zero inflation (low psi) - mode should be 0
        psi_low = pt.constant(0.3)
        mu = pt.constant(5.0)
        mode_low_psi = ZIPoisson.mode(psi_low, mu).eval()
        assert mode_low_psi == 0.0

        # Low zero inflation (high psi) with high mu - mode should be base mode
        psi_high = pt.constant(0.99)
        mu_high = pt.constant(10.0)
        mode_high_psi = ZIPoisson.mode(psi_high, mu_high).eval()
        # For high psi, mode should be floor(mu) = 10
        assert mode_high_psi == 10.0

    def test_zi_mode_with_zi_binomial(self):
        """Test zi_mode with actual ZI-Binomial distribution."""
        from distributions import zi_binomial as ZIBinomial

        # High zero inflation (low psi) - mode should be 0
        psi_low = pt.constant(0.2)
        n = pt.constant(20, dtype="int64")
        p = pt.constant(0.5)
        mode_low_psi = ZIBinomial.mode(psi_low, n, p).eval()
        assert mode_low_psi == 0.0

        # Low zero inflation (high psi) - mode should be base mode
        psi_high = pt.constant(0.99)
        mode_high_psi = ZIBinomial.mode(psi_high, n, p).eval()
        # For Binomial(20, 0.5), mode = floor((n+1)*p) = floor(10.5) = 10
        assert mode_high_psi == 10.0

    def test_zi_mode_with_zi_negativebinomial(self):
        """Test zi_mode with actual ZI-Negative Binomial distribution."""
        from distributions import zi_negativebinomial as ZINegBinom

        # High zero inflation (low psi) - mode should be 0
        psi_low = pt.constant(0.2)
        n = pt.constant(5.0)
        p = pt.constant(0.3)
        mode_low_psi = ZINegBinom.mode(psi_low, n, p).eval()
        assert mode_low_psi == 0.0

        # Low zero inflation (high psi) - mode should be base mode
        psi_high = pt.constant(0.99)
        n_high = pt.constant(10.0)
        p_high = pt.constant(0.2)
        mode_high_psi = ZINegBinom.mode(psi_high, n_high, p_high).eval()
        # For NegBinom with n>=1: mode = floor((n-1)*(1-p)/p) = floor(9*0.8/0.2) = floor(36) = 36
        assert mode_high_psi == 36.0

    def test_zi_mode_vectorized(self):
        """Test that zi_mode works with array inputs."""
        from distributions import zi_poisson as ZIPoisson

        # Array of psi values
        psi_array = pt.constant(np.array([0.1, 0.5, 0.99]))
        mu = pt.constant(5.0)

        modes = ZIPoisson.mode(psi_array, mu).eval()

        # Low psi (0.1) -> mode should be 0
        assert modes[0] == 0.0
        # Medium psi (0.5) -> likely still 0 due to zero inflation
        assert modes[1] == 0.0
        # High psi (0.99) -> mode should be floor(mu) = 5
        assert modes[2] == 5.0

    def test_zi_mode_preserves_shape(self):
        """Test that zi_mode preserves input shape."""
        from distributions import zi_poisson as ZIPoisson

        psi = pt.constant(np.array([[0.9, 0.9], [0.9, 0.9]]))
        mu = pt.constant(np.array([[3.0, 5.0], [7.0, 10.0]]))

        modes = ZIPoisson.mode(psi, mu).eval()

        assert modes.shape == (2, 2)
        # With high psi, modes should be floor(mu)
        expected = np.array([[3.0, 5.0], [7.0, 10.0]])
        assert_allclose(modes, expected)
