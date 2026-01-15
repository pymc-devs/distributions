"""Test Categorical distribution against empirical samples."""

import numpy as np
import pytensor.tensor as pt
import pytest
from numpy.testing import assert_allclose
from scipy.stats import kurtosis, skew

from distributions import categorical as Categorical


def make_params(p, dtype="float64"):
    return (pt.as_tensor_variable(np.array(p, dtype=dtype)),)


@pytest.mark.parametrize(
    "probs",
    [
        [0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.2, 0.3, 0.4],
        [0.7, 0.2, 0.1],
        [0.01, 0.01, 0.98],
    ],
)
def test_categorical_empirical(probs):
    p_params = make_params(probs)
    k = len(probs)
    support = (0, k - 1)
    sample_size = 500_000
    param_info = f"\nCategorical params: {probs}"

    rng_p = pt.random.default_rng(42)
    rvs = Categorical.rvs(*p_params, size=sample_size, random_state=rng_p).eval()

    # Moments
    theoretical_mean = Categorical.mean(*p_params).eval()
    theoretical_var = Categorical.var(*p_params).eval()
    theoretical_std = Categorical.std(*p_params).eval()
    theoretical_skewness = Categorical.skewness(*p_params).eval()
    theoretical_kurtosis = Categorical.kurtosis(*p_params).eval()

    assert_allclose(
        theoretical_mean,
        rvs.mean(),
        rtol=1e-2,
        atol=1e-3,
        err_msg=f"Mean test failed with {param_info}",
    )
    assert_allclose(
        theoretical_var,
        rvs.var(),
        rtol=1e-2,
        atol=1e-3,
        err_msg=f"Variance test failed with {param_info}",
    )
    assert_allclose(
        theoretical_std,
        rvs.std(),
        rtol=1e-2,
        atol=1e-3,
        err_msg=f"Standard deviation test failed with {param_info}",
    )
    assert_allclose(
        theoretical_skewness,
        skew(rvs),
        rtol=1e-1,
        atol=1e-2,
        err_msg=f"Skewness test failed with {param_info}",
    )
    assert_allclose(
        theoretical_kurtosis,
        kurtosis(rvs),
        rtol=1e-1,
        atol=1e-2,
        err_msg=f"Kurtosis test failed with {param_info}",
    )

    # PDF/PMF
    for i in range(k):
        expected_pmf = probs[i] / sum(probs)
        actual_pmf = Categorical.pdf(i, *p_params).eval()
        assert_allclose(
            actual_pmf,
            expected_pmf,
            rtol=1e-10,
            err_msg=f"PMF test failed at x={i} with {param_info}",
        )

    # PDF outside support
    assert_allclose(
        Categorical.pdf(-1, *p_params).eval(),
        0.0,
        atol=1e-10,
        err_msg=f"PMF outside support (x=-1) should be 0 with {param_info}",
    )
    assert_allclose(
        Categorical.pdf(k, *p_params).eval(),
        0.0,
        atol=1e-10,
        err_msg=f"PMF outside support (x=k) should be 0 with {param_info}",
    )
    assert_allclose(
        Categorical.pdf(0.5, *p_params).eval(),
        0.0,
        atol=1e-10,
        err_msg=f"PMF at non-integer (x=0.5) should be 0 with {param_info}",
    )

    # CDF
    normalized_probs = np.array(probs) / sum(probs)
    cumsum_probs = np.cumsum(normalized_probs)
    for i in range(k):
        expected_cdf = cumsum_probs[i]
        actual_cdf = Categorical.cdf(i, *p_params).eval()
        assert_allclose(
            actual_cdf,
            expected_cdf,
            rtol=1e-10,
            err_msg=f"CDF test failed at x={i} with {param_info}",
        )

    # CDF bounds
    assert_allclose(
        Categorical.cdf(-1, *p_params).eval(),
        0.0,
        err_msg=f"CDF at x=-1 should be 0 with {param_info}",
    )
    assert_allclose(
        Categorical.cdf(k, *p_params).eval(),
        1.0,
        err_msg=f"CDF at x=k should be 1 with {param_info}",
    )

    # PPF
    q_vals = np.linspace(0.01, 0.99, 20)
    ppf_vals = Categorical.ppf(q_vals, *p_params).eval()
    for i, q in enumerate(q_vals):
        x = ppf_vals[i]
        cdf_at_x = Categorical.cdf(x, *p_params).eval()
        assert cdf_at_x >= q, f"PPF-CDF inverse failed at q={q} with {param_info}"

    # PPF bounds
    assert np.isnan(Categorical.ppf(-0.1, *p_params).eval())
    assert np.isnan(Categorical.ppf(1.1, *p_params).eval())

    # SF = 1 - CDF
    for i in range(k):
        sf_val = Categorical.sf(i, *p_params).eval()
        cdf_val = Categorical.cdf(i, *p_params).eval()
        assert_allclose(
            sf_val + cdf_val,
            1.0,
            rtol=1e-10,
            err_msg=f"SF + CDF != 1 at x={i} with {param_info}",
        )

    # Entropy
    expected_entropy = -np.sum(normalized_probs * np.log(normalized_probs + 1e-20))
    actual_entropy = Categorical.entropy(*p_params).eval()
    assert_allclose(
        actual_entropy,
        expected_entropy,
        rtol=1e-6,
        err_msg=f"Entropy test failed with {param_info}",
    )

    # Mode
    expected_mode = np.argmax(probs)
    actual_mode = Categorical.mode(*p_params).eval()
    assert_allclose(
        actual_mode,
        expected_mode,
        err_msg=f"Mode test failed with {param_info}",
    )


def test_categorical_unnormalized():
    """Test that unnormalized probabilities are handled correctly."""
    p_unnorm = make_params([2.0, 4.0, 4.0])
    p_norm = make_params([0.2, 0.4, 0.4])

    assert_allclose(
        Categorical.pdf(0, *p_unnorm).eval(),
        Categorical.pdf(0, *p_norm).eval(),
        rtol=1e-10,
    )
    assert_allclose(
        Categorical.pdf(1, *p_unnorm).eval(),
        Categorical.pdf(1, *p_norm).eval(),
        rtol=1e-10,
    )
    assert_allclose(
        Categorical.cdf(1, *p_unnorm).eval(),
        Categorical.cdf(1, *p_norm).eval(),
        rtol=1e-10,
    )
    assert_allclose(
        Categorical.entropy(*p_unnorm).eval(),
        Categorical.entropy(*p_norm).eval(),
        rtol=1e-10,
    )


def test_categorical_degenerate():
    """Test degenerate categorical (single outcome)."""
    p_params = make_params([1.0])
    k = 1

    assert_allclose(Categorical.mean(*p_params).eval(), 0.0)
    assert_allclose(Categorical.var(*p_params).eval(), 0.0)
    assert_allclose(Categorical.mode(*p_params).eval(), 0)
    assert_allclose(Categorical.entropy(*p_params).eval(), 0.0)
    assert_allclose(Categorical.pdf(0, *p_params).eval(), 1.0)
    assert_allclose(Categorical.cdf(0, *p_params).eval(), 1.0)


def test_categorical_two_outcomes():
    """Test binary categorical (Bernoulli-like)."""
    p = 0.3
    p_params = make_params([1 - p, p])

    # Mean should be p for binary categorical with outcomes 0, 1
    assert_allclose(Categorical.mean(*p_params).eval(), p, rtol=1e-10)

def test_categorical_batched():
    # Batched probabilities: (2, 3)
    p = np.array([
        [0.1, 0.2, 0.7],
        [0.3, 0.4, 0.3]
    ])
    # Batched x: (2,)
    x = np.array([2, 0])
    
    p_pt = pt.as_tensor_variable(p)
    x_pt = pt.as_tensor_variable(x)
    
    # Expected logpdf: [log(0.7), log(0.3)]
    expected_logpdf = np.log([0.7, 0.3])
    actual_logpdf = Categorical.logpdf(x_pt, p_pt).eval()
    
    assert_allclose(actual_logpdf, expected_logpdf)

def test_categorical_batched_moments():
    p = np.array([
        [0.1, 0.2, 0.7],
        [0.3, 0.4, 0.3]
    ])
    p_pt = pt.as_tensor_variable(p)
    
    # Mean: 0*0.1 + 1*0.2 + 2*0.7 = 1.6
    #       0*0.3 + 1*0.4 + 2*0.3 = 1.0
    expected_mean = np.array([1.6, 1.0])
    actual_mean = Categorical.mean(p_pt).eval()
    
    assert_allclose(actual_mean, expected_mean)
