"""Selection effects behind before-and-after testimonials.

A person is recruited because their baseline score crossed a threshold. Their
score has a stable part that varies between people and a noisy part that varies
between visits, so the baseline that got them selected is high partly by
accident, and the accident does not repeat. The follow-up therefore improves on
average with no intervention at all.

The moments after selection are exact. Writing ``a = (cutoff - mu) / sd`` and
``phi``, ``Phi`` for the standard normal density and tail, selection keeps a
fraction ``Phi(-a)``, and the inverse Mills ratio ``lambda = phi(a) / Phi(-a)``
gives the selected baseline mean ``mu + sd * lambda`` and the selected baseline
variance ``sd^2 (1 + a*lambda - lambda^2)``. The regression of follow-up on
baseline has slope ``between^2 / (between^2 + within^2 / k)``, which is the
shrinkage that produces the apparent improvement.

Lower scores are preferable throughout, so a negative ``effect`` is beneficial.
"""

from dataclasses import dataclass
from math import erfc, exp, pi, sqrt
from random import Random
from statistics import NormalDist

from blog_reproducibility.common.validation import count, non_negative, real

__all__ = [
    "ConditionalPrediction",
    "SelectedMoments",
    "TestimonialExamples",
    "conditional_prediction",
    "example_payload",
    "selected_moments",
    "simulate_pairs",
]


@dataclass(frozen=True, slots=True)
class SelectedMoments:
    """Exact moments of a cohort selected on its baseline score."""

    selected_fraction: float
    slope: float
    baseline_mean: float
    followup_mean: float
    mean_change: float
    selected_baseline_variance: float
    change_variance: float


@dataclass(frozen=True, slots=True)
class ConditionalPrediction:
    """The follow-up distribution expected for one person at a given baseline."""

    mean: float
    prediction_95: tuple[float, float]


@dataclass(frozen=True, slots=True)
class TestimonialExamples:
    """Every number the testimonial article reports."""

    selected_cohort: SelectedMoments
    interventions: dict[str, SelectedMoments]
    baseline_averaging: dict[str, SelectedMoments]
    conditional_at_70: ConditionalPrediction
    unselected_drop_at_least_10: float
    at_least_one_large_drop_among_20_independent_people: float


def selected_moments(
    *,
    cutoff: float = 65.0,
    mu: float = 50.0,
    between_sd: float = 8.0,
    within_sd: float = 6.0,
    baseline_count: int = 1,
    effect: float = 0.0,
) -> SelectedMoments:
    """Return the moments of a cohort whose baseline average reached ``cutoff``.

    ``baseline_count`` is how many baseline measurements are averaged before
    the selection decision. Averaging more of them removes measurement noise
    from the selection, which shrinks the apparent improvement without removing
    it: the stable component is still selected on.

    The follow-up is one new observation with ``effect`` added to it.
    """
    threshold = real(cutoff, name="cutoff")
    centre = real(mu, name="mu")
    between = non_negative(between_sd, name="between_sd")
    within = non_negative(within_sd, name="within_sd")
    shift = real(effect, name="effect")
    baselines = count(baseline_count, name="baseline_count", minimum=1)

    if between + within == 0.0:
        raise ValueError("The total variance must be positive")

    between_var = between**2
    within_var = within**2
    baseline_var = between_var + within_var / baselines
    baseline_sd = sqrt(baseline_var)
    slope = between_var / baseline_var

    standardized = (threshold - centre) / baseline_sd
    selected_fraction = 0.5 * erfc(standardized / sqrt(2.0))
    if selected_fraction == 0.0:
        raise ValueError("Selection probability underflows at this cutoff")

    mills = exp(-standardized * standardized / 2.0) / sqrt(2.0 * pi) / selected_fraction
    baseline_mean = centre + baseline_sd * mills
    followup_mean = centre + slope * (baseline_mean - centre) + shift

    selected_baseline_var = max(0.0, baseline_var * (1.0 + standardized * mills - mills**2))
    residual_var = max(0.0, between_var + within_var - between_var**2 / baseline_var)

    return SelectedMoments(
        selected_fraction=selected_fraction,
        slope=slope,
        baseline_mean=baseline_mean,
        followup_mean=followup_mean,
        mean_change=followup_mean - baseline_mean,
        selected_baseline_variance=selected_baseline_var,
        change_variance=(1.0 - slope) ** 2 * selected_baseline_var + residual_var,
    )


def simulate_pairs(
    *,
    n: int = 2_000,
    seed: int = 20260919,
    effect: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Draw ``n`` people, each with one stable component and two noisy visits.

    The generator is created here from ``seed``, so the draw never depends on
    global random state and repeats exactly.
    """
    people = count(n, name="n", minimum=1)
    shift = real(effect, name="effect")
    rng = Random(count(seed, name="seed"))

    pairs: list[tuple[float, float]] = []
    for _ in range(people):
        stable = rng.gauss(50.0, 8.0)
        pairs.append((stable + rng.gauss(0.0, 6.0), stable + rng.gauss(0.0, 6.0) + shift))
    return tuple(pairs)


def conditional_prediction(
    baseline: float = 70.0,
    *,
    correlation: float = 0.64,
    sd: float = 10.0,
    mu: float = 50.0,
) -> ConditionalPrediction:
    """Return the follow-up mean and 95% prediction interval at one baseline."""
    observed = real(baseline, name="baseline")
    rho = real(correlation, name="correlation")
    if not -1.0 < rho < 1.0:
        raise ValueError("correlation must lie in (-1, 1)")
    spread = non_negative(sd, name="sd")
    centre = real(mu, name="mu")

    distribution = NormalDist(centre + rho * (observed - centre), spread * sqrt(1 - rho * rho))
    return ConditionalPrediction(
        mean=distribution.mean,
        prediction_95=(distribution.inv_cdf(0.025), distribution.inv_cdf(0.975)),
    )


def example_payload() -> TestimonialExamples:
    """Return the numbers the article reports."""
    correlation, sd = 0.64, 10.0
    unselected_drop = NormalDist(0.0, sqrt(2 * sd * sd * (1 - correlation))).cdf(-10.0)

    return TestimonialExamples(
        selected_cohort=selected_moments(),
        interventions={str(effect): selected_moments(effect=effect) for effect in (-3.0, 0.0, 3.0)},
        baseline_averaging={str(k): selected_moments(baseline_count=k) for k in (1, 4, 16)},
        conditional_at_70=conditional_prediction(70.0, correlation=correlation, sd=sd),
        unselected_drop_at_least_10=unselected_drop,
        at_least_one_large_drop_among_20_independent_people=1 - (1 - unselected_drop) ** 20,
    )
