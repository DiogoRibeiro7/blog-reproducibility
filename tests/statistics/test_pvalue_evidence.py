"""Independent checks of the article's Gaussian probability calculations."""

from math import sqrt
from random import Random
from statistics import NormalDist

import pytest

from blog_reproducibility.statistics.pvalue_evidence import (
    posterior_signal,
    rejection_probability,
    selected_studies,
    study_summary,
    two_sided_p,
)


def test_rejection_probability_matches_direct_density_integration() -> None:
    """Analytic rejection probability should match direct numerical integration."""
    cutoff = NormalDist().inv_cdf(0.975)
    normal = NormalDist(2.0, 1.0)
    samples = 16_000
    width = 2.0 * cutoff / samples
    inside = sum(normal.pdf(-cutoff + (index + 0.5) * width) for index in range(samples)) * width

    assert rejection_probability() == pytest.approx(1.0 - inside, abs=1e-8)
    assert two_sided_p(2.0) == pytest.approx(2.0 * NormalDist().cdf(-2.0))


def test_null_calibration_and_sign_symmetry() -> None:
    """The null rejection rate is alpha and two-sided power is sign-symmetric."""
    for alpha in (0.001, 0.01, 0.05, 0.2):
        assert rejection_probability(0.0, alpha) == pytest.approx(alpha)
        assert rejection_probability(2.0, alpha) == pytest.approx(
            rejection_probability(-2.0, alpha)
        )

    assert two_sided_p(2.0) == two_sided_p(-2.0)
    assert two_sided_p(0.0) == 1.0


def test_posterior_matches_direct_density_weighting() -> None:
    """Posterior model probability should equal direct Bayes density weighting."""
    for prior in (0.01, 0.1, 0.5):
        for z in (-2.0, 0.0, 2.0):
            signal = prior * NormalDist(2.0, 1.0).pdf(z)
            null = (1.0 - prior) * NormalDist(0.0, 1.0).pdf(z)
            assert posterior_signal(z, prior) == pytest.approx(signal / (signal + null))

    assert posterior_signal(-2.0) < 0.001 < posterior_signal(2.0)
    assert posterior_signal(2.0, 0.0) == 0.0
    assert posterior_signal(2.0, 1.0) == 1.0


def test_selected_counts_conserve_population_and_match_simulation() -> None:
    """Expected selected-study composition should agree with a seeded simulation."""
    row = selected_studies()
    counts = (
        row.signal_rejections,
        row.null_rejections,
        row.signal_nonrejections,
        row.null_nonrejections,
    )

    assert sum(counts) == pytest.approx(10_000.0)
    assert row.signal_rejections + row.signal_nonrejections == pytest.approx(1_000.0)

    rng = Random(20241107)
    cutoff = NormalDist().inv_cdf(0.975)
    selected = 0
    selected_signal = 0
    simulations = 100_000

    for _ in range(simulations):
        signal = rng.random() < 0.1
        z = rng.gauss(2.0 if signal else 0.0, 1.0)
        if abs(z) >= cutoff:
            selected += 1
            selected_signal += int(signal)

    observed = selected_signal / selected
    expected = row.signal_given_rejection
    standard_error = sqrt(expected * (1.0 - expected) / selected)
    assert abs(observed - expected) < 5.0 * standard_error


def test_replication_probability_matches_joint_probability() -> None:
    """Conditional repeat rejection should match the joint-probability calculation."""
    row = selected_studies()
    both = 0.1 * row.power**2 + 0.9 * 0.05**2
    first = 0.1 * row.power + 0.9 * 0.05

    assert row.repeat_rejection_given_first_rejection == pytest.approx(both / first)


def test_study_comparison_and_scale_invariance() -> None:
    """Nearly identical estimates can cross opposite sides of the threshold."""
    study_a = study_summary(0.20, 0.10)
    study_b = study_summary(0.19, 0.11)

    assert study_a.p_value < 0.05 < study_b.p_value

    difference = study_summary(
        study_a.estimate - study_b.estimate,
        sqrt(study_a.se**2 + study_b.se**2),
    )
    assert difference.ci95[0] < 0.0 < difference.ci95[1]
    assert difference.p_value > 0.9

    scaled = study_summary(2.0, 1.0)
    assert scaled.p_value == pytest.approx(study_a.p_value)
    assert scaled.ci95 == pytest.approx(tuple(10.0 * value for value in study_a.ci95))


def test_article_regression_values() -> None:
    """Key numerical values quoted in the article should remain stable."""
    selected = selected_studies()
    study_a = study_summary(0.20, 0.10)
    study_b = study_summary(0.19, 0.11)
    difference = study_summary(0.01, sqrt(0.10**2 + 0.11**2))

    assert two_sided_p(2.0) == pytest.approx(0.04550026389635844)
    assert selected.power == pytest.approx(0.516005273976175)
    assert selected.signal_given_rejection == pytest.approx(0.5341640339625117)
    assert selected.repeat_rejection_given_first_rejection == pytest.approx(0.29892325699491906)
    assert study_a.p_value == pytest.approx(0.04550026389635844)
    assert study_b.p_value == pytest.approx(0.08411869479791355)
    assert difference.p_value == pytest.approx(0.94636892512424)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -1.0, float("nan")])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    """Alpha must be finite and strictly between zero and one."""
    with pytest.raises(ValueError):
        rejection_probability(alpha=alpha)


@pytest.mark.parametrize("prior", [-0.1, 1.1, float("nan")])
def test_invalid_prior_is_rejected(prior: float) -> None:
    """Prior model probability must lie in the unit interval."""
    with pytest.raises(ValueError):
        posterior_signal(2.0, prior)


def test_invalid_standard_error_is_rejected() -> None:
    """Study summaries require a positive finite standard error."""
    with pytest.raises(ValueError):
        study_summary(0.2, 0.0)


def test_non_numeric_input_is_rejected() -> None:
    """Runtime validation rejects values outside the numerical API."""
    with pytest.raises(TypeError):
        two_sided_p("2")  # type: ignore[arg-type]
