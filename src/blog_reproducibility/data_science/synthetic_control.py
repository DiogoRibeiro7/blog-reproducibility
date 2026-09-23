"""Synthetic control, for the article on estimating single-unit interventions.

Twenty donors and one treated unit follow two common factors, a drifting trend
and a seasonal cycle, with unit-specific loadings and levels. The treated unit
loads more heavily on the trend than the average donor, and its outcome drops
by eight units from month 37.

Synthetic control fits non-negative weights summing to one so that a weighted
average of donors tracks the treated unit before the intervention; the gap
afterwards is the estimated effect. Inference is by permutation: every donor is
treated as if it had been the treated unit, and the treated unit's ratio of
post- to pre-intervention RMSPE is ranked among theirs.

The panel is the article's simulation with seed 0, reproduced draw for draw.
The weights come from SciPy's SLSQP solver, as in the article, so they can move
in the last digits between SciPy releases.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from blog_reproducibility.common.validation import count, real

__all__ = [
    "DONORS",
    "EFFECT",
    "POST_MONTHS",
    "PRE_MONTHS",
    "PlaceboResult",
    "SyntheticControlSummary",
    "example_payload",
    "placebo_gaps",
    "simulate_panel",
    "synthetic_weights",
]

DONORS: Final[int] = 20
PRE_MONTHS: Final[int] = 36
POST_MONTHS: Final[int] = 12
EFFECT: Final[float] = -8.0
TREATED_LOADINGS: Final[tuple[float, float]] = (1.2, 0.9)
# The placebo in time pretends the intervention came a year early.
FAKE_INTERVENTION: Final[int] = PRE_MONTHS - 12


@dataclass(frozen=True, slots=True)
class PlaceboResult:
    """Post/pre RMSPE ratios and the permutation p-value."""

    treated_ratio: float
    donor_median_ratio: float
    donor_max_ratio: float
    rank: int
    p_value: float


@dataclass(frozen=True, slots=True)
class SyntheticControlSummary:
    """Every number the article reports about the seeded panel."""

    donors_with_weight: int
    largest_weights: tuple[float, ...]
    pre_period_rmspe: float
    synthetic_control_effect: float
    before_after_effect: float
    difference_in_differences_effect: float
    placebo_in_space: PlaceboResult
    placebo_in_time_gap: float


def simulate_panel(
    seed: int = 0,
    *,
    donors: int = DONORS,
    pre: int = PRE_MONTHS,
    post: int = POST_MONTHS,
    effect: float = EFFECT,
) -> NDArray[np.float64]:
    """Outcomes of shape ``(donors + 1, pre + post)``; row 0 is the treated unit."""
    rng = np.random.default_rng(count(seed, name="seed"))
    units = count(donors, name="donors", minimum=1) + 1
    months = count(pre, name="pre", minimum=1) + count(post, name="post", minimum=1)
    trend = np.cumsum(rng.normal(0.3, 1.0, months)) + 100
    season = 10 * np.sin(np.arange(months) * 2 * np.pi / 12)
    factors = np.column_stack([trend, season])
    loadings = rng.uniform(0.3, 1.5, (units, 2))
    loadings[0] = TREATED_LOADINGS
    levels = rng.uniform(-20, 20, units)
    outcomes = levels[:, None] + loadings @ factors.T + rng.normal(0, 2.0, (units, months))
    outcomes[0, pre:] += real(effect, name="effect")
    return np.asarray(outcomes)


def synthetic_weights(
    treated: NDArray[np.float64], donors: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Convex weights on donors minimising mean squared error against the treated path."""
    if donors.ndim != 2 or treated.shape != (donors.shape[1],):
        raise ValueError("donors must be a matrix whose rows match the treated path")
    units = donors.shape[0]

    def loss(weights: NDArray[np.float64]) -> float:
        return float(np.mean((treated - weights @ donors) ** 2))

    result = minimize(
        loss,
        np.full(units, 1 / units),
        method="SLSQP",
        bounds=[(0, 1)] * units,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1},
        options={"maxiter": 500},
    )
    return np.asarray(result.x, dtype=np.float64)


def placebo_gaps(outcomes: NDArray[np.float64], pre: int = PRE_MONTHS) -> NDArray[np.float64]:
    """Each unit's outcome minus its synthetic control built from all other units."""
    gaps = []
    for unit in range(outcomes.shape[0]):
        others = np.delete(outcomes, unit, axis=0)
        weights = synthetic_weights(outcomes[unit, :pre], others[:, :pre])
        gaps.append(outcomes[unit] - weights @ others)
    return np.asarray(gaps)


def _rmspe(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(values**2)))


def placebo_in_space(gaps: NDArray[np.float64], pre: int = PRE_MONTHS) -> PlaceboResult:
    """Rank the treated unit's post/pre RMSPE ratio among the donors' ratios."""
    ratios = np.array([_rmspe(gap[pre:]) / _rmspe(gap[:pre]) for gap in gaps])
    rank = 1 + int(np.sum(ratios[1:] >= ratios[0]))
    return PlaceboResult(
        treated_ratio=float(ratios[0]),
        donor_median_ratio=float(np.median(ratios[1:])),
        donor_max_ratio=float(ratios[1:].max()),
        rank=rank,
        p_value=rank / ratios.size,
    )


def example_payload() -> SyntheticControlSummary:
    """Return the estimates and placebo checks the article reports."""
    outcomes = simulate_panel()
    pre = PRE_MONTHS
    treated, donors = outcomes[0], outcomes[1:]

    weights = synthetic_weights(treated[:pre], donors[:, :pre])
    gap = treated - weights @ donors
    before_after = float(treated[pre:].mean() - treated[:pre].mean())
    donor_change = float(donors[:, pre:].mean() - donors[:, :pre].mean())

    fake = FAKE_INTERVENTION
    early = synthetic_weights(treated[:fake], donors[:, :fake])
    time_gap = treated[:pre] - early @ donors[:, :pre]

    return SyntheticControlSummary(
        donors_with_weight=int(np.sum(weights > 0.01)),
        largest_weights=tuple(float(value) for value in np.sort(weights)[::-1][:4]),
        pre_period_rmspe=_rmspe(gap[:pre]),
        synthetic_control_effect=float(gap[pre:].mean()),
        before_after_effect=before_after,
        difference_in_differences_effect=before_after - donor_change,
        placebo_in_space=placebo_in_space(placebo_gaps(outcomes)),
        placebo_in_time_gap=float(time_gap[fake:].mean()),
    )
