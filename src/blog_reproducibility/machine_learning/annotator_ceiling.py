"""Label noise and the accuracy ceiling, for the article on annotator agreement.

When the evaluation label is one annotator's judgement and that annotator flips
the true label with probability ``e``, a model that predicts the truth exactly
agrees with the labels on a share ``1 - e`` of items: that is the highest score
it can display. With an odd panel of ``m`` independent annotators and a majority
vote, the label is wrong only when more than half of them err at once, so the
ceiling becomes

    1 - P(Binomial(m, e) > m / 2).

Label noise also shrinks the measured gap between two models by ``1 - 2e``, so
the evaluation set needed to detect a fixed true gap grows by ``(1 - 2e)^-2``.
The article's sample-size table uses the normal approximation with a variance
of 1/4 per item, and is reproduced exactly here because it involves no draws.

The figure simulates 200,000 balanced binary items from a generator seeded at 0,
labels them with one, three, five and seven annotators at six error rates, and
scores the true labels against the majority vote. That simulation is reproduced
draw for draw. The article's tables come from a different stream of draws (one
generator shared across its code blocks), so they are not reproduced here; the
simulated ceilings are checked against the closed form instead.
"""

from dataclasses import dataclass
from math import comb
from typing import Final

import numpy as np
from scipy.stats import norm

from blog_reproducibility.common.validation import count, positive, probability

__all__ = [
    "ERROR_RATES",
    "ITEMS",
    "PANEL_SIZES",
    "POWER_ERROR_RATES",
    "SEED",
    "TRUE_GAP",
    "AnnotatorCeilingSummary",
    "CeilingCurve",
    "PowerRow",
    "ceiling",
    "example_payload",
    "items_for_power",
    "majority_error",
    "simulate_ceilings",
]

SEED: Final[int] = 0
ITEMS: Final[int] = 200_000
ERROR_RATES: Final[tuple[float, ...]] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
PANEL_SIZES: Final[tuple[int, ...]] = (1, 3, 5, 7)
# The article's sample-size table: two models five points apart, 80% power at 5%.
TRUE_GAP: Final[float] = 0.05
POWER_ERROR_RATES: Final[tuple[float, ...]] = (0.0, 0.05, 0.10, 0.20)


@dataclass(frozen=True, slots=True)
class CeilingCurve:
    """Measured and closed-form ceilings for one panel size across error rates."""

    annotators: int
    error_rates: tuple[float, ...]
    simulated: tuple[float, ...]
    closed_form: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PowerRow:
    """The measurable gap and evaluation set needed at one annotator error rate."""

    error_rate: float
    measurable_gap: float
    items_needed: float
    relative_to_clean: float


@dataclass(frozen=True, slots=True)
class AnnotatorCeilingSummary:
    """The figure's ceilings and the article's sample-size table."""

    curves: tuple[CeilingCurve, ...]
    power: tuple[PowerRow, ...]


def _odd_panel(annotators: int) -> int:
    size = count(annotators, name="annotators", minimum=1)
    if size % 2 == 0:
        raise ValueError("annotators must be odd so that a majority vote has no ties")
    return size


def majority_error(error_rate: float, annotators: int) -> float:
    """Probability that a majority of independent annotators give the wrong label."""
    error = probability(error_rate, name="error_rate")
    size = _odd_panel(annotators)
    return float(
        sum(
            comb(size, wrong) * error**wrong * (1 - error) ** (size - wrong)
            for wrong in range(size // 2 + 1, size + 1)
        )
    )


def ceiling(error_rate: float, annotators: int = 1) -> float:
    """Accuracy a perfect model appears to reach against majority-vote labels."""
    return 1.0 - majority_error(error_rate, annotators)


def items_for_power(
    error_rate: float,
    *,
    true_gap: float = TRUE_GAP,
    power: float = 0.8,
    alpha: float = 0.05,
) -> float:
    """Items needed to detect ``true_gap`` once label noise compresses it by ``1 - 2e``."""
    error = probability(error_rate, name="error_rate")
    if error >= 0.5:
        raise ValueError("error_rate must be below 0.5, or the labels carry no signal")
    gap = (1 - 2 * error) * positive(true_gap, name="true_gap")
    beta = probability(power, name="power", inclusive=False)
    size = probability(alpha, name="alpha", inclusive=False)
    z = float(norm.ppf(1 - size / 2) + norm.ppf(beta))
    return (z / gap) ** 2 * 0.25


def simulate_ceilings(
    seed: int = SEED,
    *,
    items: int = ITEMS,
    error_rates: tuple[float, ...] = ERROR_RATES,
    panel_sizes: tuple[int, ...] = PANEL_SIZES,
) -> tuple[CeilingCurve, ...]:
    """Score the true labels against majority votes of simulated annotators."""
    rng = np.random.default_rng(count(seed, name="seed"))
    size = count(items, name="items", minimum=1)
    errors = tuple(probability(error, name="error_rate") for error in error_rates)
    panels = tuple(_odd_panel(panel) for panel in panel_sizes)
    truth = (rng.random(size) < 0.5).astype(int)

    curves = []
    for panel in panels:
        measured = []
        for error in errors:
            votes = np.zeros(size)
            for _ in range(panel):
                votes += np.where(rng.random(size) < error, 1 - truth, truth)
            majority = (votes > panel / 2).astype(int)
            measured.append(float(np.mean(majority == truth)))
        curves.append(
            CeilingCurve(
                annotators=panel,
                error_rates=errors,
                simulated=tuple(measured),
                closed_form=tuple(ceiling(error, panel) for error in errors),
            )
        )
    return tuple(curves)


def example_payload() -> AnnotatorCeilingSummary:
    """Return the figure's ceilings and the article's sample-size table."""
    clean = items_for_power(0.0)
    power = tuple(
        PowerRow(
            error_rate=error,
            measurable_gap=(1 - 2 * error) * TRUE_GAP,
            items_needed=items_for_power(error),
            relative_to_clean=items_for_power(error) / clean,
        )
        for error in POWER_ERROR_RATES
    )
    return AnnotatorCeilingSummary(curves=simulate_ceilings(), power=power)
