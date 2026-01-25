"""Common utilities for testing distributions not available in scipy."""

import numpy as np
import pytensor.tensor as pt
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import kurtosis, skew


def run_empirical_tests(
    p_dist,
    p_params,
    support,
    name=None,
    sample_size=500_000,
    mean_rtol=1e-4,
    mean_atol=1e-4,
    var_rtol=1e-4,
    std_rtol=1e-4,
    skewness_rtol=1e-1,
    kurtosis_rtol=1e-1,
    quantiles_rtol=1e-4,
    quantiles_atol=1e-4,
    cdf_rtol=1e-4,
    pdf_cdf_rtol=1e-3,
    is_discrete=False,
):
    """Test a distribution against empirical samples for distributions not in scipy."""
    rng_p = pt.random.default_rng(1)
    rvs = p_dist.rvs(*p_params, size=sample_size, random_state=rng_p).eval()
    sample_x = rvs[:20]

    param_vals = [param.eval() if hasattr(param, "eval") else param for param in p_params]
    param_info = f"\n{name} params: {param_vals}"

    # Moments
    theoretical_mean = p_dist.mean(*p_params).eval()
    theoretical_var = p_dist.var(*p_params).eval()
    theoretical_std = p_dist.std(*p_params).eval()
    theoretical_skewness = p_dist.skewness(*p_params).eval()
    theoretical_kurtosis = p_dist.kurtosis(*p_params).eval()

    assert_allclose(
        theoretical_mean,
        rvs.mean(),
        rtol=mean_rtol,
        atol=mean_atol,
        err_msg=f"Mean test failed with {param_info}",
    )
    assert_allclose(
        theoretical_var,
        rvs.var(),
        rtol=var_rtol,
        atol=1e-4,
        err_msg=f"Variance test failed with {param_info}",
    )
    assert_allclose(
        theoretical_std,
        rvs.std(),
        rtol=std_rtol,
        atol=1e-4,
        err_msg=f"Standard deviation test failed with {param_info}",
    )
    assert_allclose(
        theoretical_skewness,
        skew(rvs),
        rtol=skewness_rtol,
        atol=1e-2,
        err_msg=f"Skewness test failed with {param_info}",
    )

    assert_allclose(
        theoretical_kurtosis,
        kurtosis(rvs),
        rtol=kurtosis_rtol,
        atol=1e-2,
        err_msg=f"Kurtosis test failed with {param_info}",
    )

    extended_vals = np.concatenate(
        [
            support,
            [support[0] - 1],
            [support[0] - 2],
            [support[1] + 1],
            [support[1] + 2],
        ]
    )
    if is_discrete:
        extended_vals = extended_vals[1:]

    # PPF
    q = np.linspace(0.01, 0.99, 50)
    theoretical_quantiles = p_dist.ppf(q, *p_params).eval()
    empirical_quantiles = np.quantile(rvs, q)
    assert_allclose(
        theoretical_quantiles,
        empirical_quantiles,
        rtol=quantiles_rtol,
        atol=quantiles_atol,
        err_msg=f"Quantiles test failed with {param_info}",
    )
    extended_expected_ppf = []
    if np.isfinite(support[0]):
        if not is_discrete:
            extended_expected_ppf.append(support[0])
    else:
        extended_expected_ppf.append(np.nan)
    if np.isfinite(support[1]):
        extended_expected_ppf.append(support[1])
    else:
        extended_expected_ppf.append(np.nan)
    extended_expected_ppf.extend([np.nan] * 4)

    assert_allclose(p_dist.ppf(extended_vals, *p_params).eval(), extended_expected_ppf)

    # CDF

    ## empirical CDF
    theoretical_cdf = p_dist.cdf(rvs[:20], *p_params).eval()
    for i, x in enumerate(sample_x):
        empirical_cdf = np.mean(rvs <= x)
        assert_allclose(
            theoretical_cdf[i],
            empirical_cdf,
            rtol=cdf_rtol,
            atol=1e-4,
            err_msg=f"CDF test failed at x={x} with {param_info}",
        )

    ## monotonicity
    x_sorted = np.sort(rvs)
    cdf_vals = p_dist.cdf(x_sorted, *p_params).eval()
    diffs = np.diff(cdf_vals)
    assert np.all(diffs >= -1e-4), "CDF is not monotonic"

    # CDF bounds
    if is_discrete:
        expected_cdf_values = [1.0, 0.0, 0.0, 1.0, 1.0]
    else:
        expected_cdf_values = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    assert_allclose(
        p_dist.cdf(extended_vals, *p_params).eval(),
        expected_cdf_values,
        err_msg=f"CDF bounds test failed with {param_info}",
    )

    # PDF
    if np.isfinite(support[0]):
        l_b = support[0]
    else:
        l_b = p_dist.ppf(0.001, *p_params)

    if np.isfinite(support[1]):
        u_b = support[1]
    else:
        u_b = p_dist.ppf(0.999, *p_params)

    ## non-negativity
    pdf_vals = p_dist.pdf(rvs, *p_params).eval()
    assert np.all(pdf_vals >= 0), "PDF has negative values"

    ## integrates to 1
    if is_discrete:
        result = pt.sum(p_dist.pdf(pt.arange(l_b, u_b + 1), *p_params)).eval()
    else:
        result, _ = quad(lambda x: p_dist.pdf(x, *p_params).eval(), l_b, u_b)
    assert np.abs(result - 1) < 0.01, f"PDF integral = {result}, should be 1"

    ## PDF is 0 outside support
    outside_vals = np.array([support[0] - 0.1, support[1] + 0.1])
    outside_pdf = p_dist.pdf(outside_vals, *p_params).eval()
    assert_allclose(
        outside_pdf,
        [0.0, 0.0],
        atol=1e-10,
        err_msg=f"PDF outside support should be 0 with {param_info}",
    )

    ## PDF-CDF inverse
    if is_discrete:
        x_vals = np.arange(
            p_dist.ppf(0.001, *p_params).eval(), p_dist.ppf(0.999, *p_params).eval() + 1
        )
        cdf_vals = p_dist.cdf(x_vals, *p_params).eval()
        pmf_vals = p_dist.pdf(x_vals, *p_params).eval()

        numerical_pmf = np.diff(cdf_vals, prepend=0)

        mask = np.abs(pmf_vals) > 1e-8
        if np.any(mask):
            rel_error = np.abs(numerical_pmf[mask] - pmf_vals[mask]) / (
                np.abs(pmf_vals[mask]) + 1e-10
            )
            assert np.all(rel_error < 1e-6), (
                f"PMF doesn't match CDF jumps. Max rel error: {np.max(rel_error)}"
            )

    else:
        x_mid = rvs[(rvs > support[0] + 1e-3) & (rvs < support[1] - 1e-3)]
        cdf_plus = p_dist.cdf(x_mid + 1e-5, *p_params).eval()
        cdf_minus = p_dist.cdf(x_mid - 1e-5, *p_params).eval()
        numerical_pdf = (cdf_plus - cdf_minus) / (2 * 1e-5)
        pdf_vals = p_dist.pdf(x_mid, *p_params).eval()

        mask = np.abs(pdf_vals) > 1e-4
        if np.any(mask):
            rel_error = np.abs(numerical_pdf[mask] - pdf_vals[mask]) / (
                np.abs(pdf_vals[mask]) + 1e-10
            )
            assert np.all(rel_error < pdf_cdf_rtol), (
                f"PDF doesn't match CDF derivative. Max rel error: {np.max(rel_error)}"
            )

    # PPF-CDF inverse
    x_vals = p_dist.ppf(q, *p_params).eval()
    p_recovered = p_dist.cdf(x_vals, *p_params).eval()

    if is_discrete:
        assert np.all(p_recovered >= q - 1e-10), "PPF-CDF inverse failed: CDF(ppf(q)) < q"
    else:
        abs_error = np.abs(p_recovered - q)
        assert np.all(abs_error < 1e-5), f"PPF-CDF inverse failed. Max error: {np.max(abs_error)}"

    # CDF-SF complement
    cdf_vals = p_dist.cdf(rvs, *p_params).eval()
    sf_vals = p_dist.sf(rvs, *p_params).eval()

    sum_vals = cdf_vals + sf_vals
    abs_error = np.abs(sum_vals - 1)
    assert np.all(abs_error < 1e-4), f"SF + CDF != 1. Max error: {np.max(abs_error)}"

    # ISF-PPF complement
    isf_vals = p_dist.isf(rvs, *p_params).eval()
    ppf_vals = p_dist.ppf(1 - rvs, *p_params).eval()

    if is_discrete:
        diff = np.abs(isf_vals - ppf_vals)
        diff = diff[~np.isnan(diff)]
        assert np.all(diff <= 1), (
            f"ISF and PPF(1-p) differ by more than 1 support step. Max diff: {np.max(diff)}"
        )
    else:
        rel_error = np.abs(isf_vals - ppf_vals) / (np.abs(ppf_vals) + 1e-10)
        assert np.all(rel_error < 1e-3), f"ISF != PPF(1-p). Max rel error: {np.max(rel_error)}"

    # Entropy
    logpdf_vals = p_dist.logpdf(rvs, *p_params).eval()
    logpdf_vals = logpdf_vals[np.isfinite(logpdf_vals)]

    mc_entropy = -np.mean(logpdf_vals)
    computed_entropy = p_dist.entropy(*p_params).eval()

    rel_error = np.abs(mc_entropy - computed_entropy) / (np.abs(computed_entropy) + 1e-10)
    assert rel_error < 0.1, f"Entropy mismatch. MC: {mc_entropy}, Computed: {computed_entropy}"
