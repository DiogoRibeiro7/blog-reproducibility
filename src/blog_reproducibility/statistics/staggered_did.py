"""Two-way fixed effects against group-time estimates, for the article on staggered rollouts.

Sixty units are observed over twenty periods. Fifteen adopt a treatment at
period 5, fifteen at 10, fifteen at 15, and fifteen never do. The outcome is

    Y_it = a_i + b_t + tau_it + e_it,

with unit effects ``a_i ~ N(0, 1)``, period effects rising from 0 to 2 plus
``N(0, 0.2^2)`` noise, and ``e_it ~ N(0, 1)``. In the figure the effect grows
with exposure, ``tau = 0.2 (k + 1)`` at ``k = t - g`` periods after adoption at
``g``; the article adds a constant effect, one larger for early adopters, and
one that grows for eight periods and then fades.

The two-way fixed effects coefficient is ``sum(D~ Y) / sum(D~^2)`` with ``D~``
the treatment dummy after removing unit and period means. The design is fixed,
so the unit and period effects cancel exactly and the coefficient's expectation
is ``sum(D~ tau) / sum(D~^2)``, with variance ``sigma^2 / sum(D~^2)``. For the
growing effect that expectation is 0.60, less than half the true average effect
on the treated, ``19 / 15 = 1.27``, because some of the comparisons it averages
use already-treated units as controls.

The group-time effect ``ATT(g, t)`` is the change in cohort ``g``'s mean outcome
from period ``g - 1`` to ``t`` minus the same change among the units not yet
treated at ``t`` (or never treated). It is a fixed linear combination of the
outcomes whose weights sum to zero over every unit and every period, so the
unit and period effects cancel, the estimate is unbiased for ``tau`` whenever
the controls are untreated, and its variance is ``sigma^2`` times the sum of the
squared weights. The event study averages ``ATT(g, g + k)`` over the cohorts
observed at exposure ``k``.

The figure simulates 200 panels from a generator seeded at 0 and is reproduced
draw for draw. The article's single panel from a generator seeded at 1 is the
same computation and is reproduced exactly, and so is its noise-free panel from
a generator seeded at 2. Its scenario table runs 500 panels per pattern from a
generator seeded at 0, the constant effect first, and its event study 200 more
panels after those, so neither is reproduced; their true effects are closed
forms and are pinned, and the tests check the rest against the closed forms
above.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from blog_reproducibility.common.validation import count, non_negative

__all__ = [
    "ADOPTION_PERIODS",
    "COHORT_SIZE",
    "EXPOSURES",
    "NEVER_TREATED",
    "NOISE_FREE_SEED",
    "NOISE_SD",
    "PATTERNS",
    "PERIODS",
    "PERIOD_SD",
    "PERIOD_TREND",
    "REPLICATIONS",
    "SEED",
    "SINGLE_PANEL_SEED",
    "UNITS",
    "UNIT_SD",
    "Control",
    "EventStudyRow",
    "ScenarioRow",
    "SinglePanel",
    "StaggeredSummary",
    "cohorts",
    "did_2x2",
    "effect_matrix",
    "event_study",
    "event_study_weights",
    "example_payload",
    "expected_twfe",
    "group_time_att",
    "group_time_average",
    "group_time_average_weights",
    "scenario_row",
    "simulate_event_study",
    "simulate_panel",
    "single_panel",
    "treatment_matrix",
    "true_att",
    "twfe",
    "twfe_sd",
]

Control = Literal["not_yet", "never"]

SEED: Final[int] = 0
SINGLE_PANEL_SEED: Final[int] = 1
NOISE_FREE_SEED: Final[int] = 2
UNITS: Final[int] = 60
PERIODS: Final[int] = 20
COHORT_SIZE: Final[int] = 15
ADOPTION_PERIODS: Final[tuple[int, ...]] = (5, 10, 15)
NEVER_TREATED: Final[int] = -1
REPLICATIONS: Final[int] = 200
EXPOSURES: Final[tuple[int, ...]] = tuple(range(10))
UNIT_SD: Final[float] = 1.0
PERIOD_TREND: Final[tuple[float, float]] = (0.0, 2.0)
PERIOD_SD: Final[float] = 0.2
NOISE_SD: Final[float] = 1.0
# The article's four effect patterns; the figure uses "growing".
PATTERNS: Final[tuple[str, ...]] = ("constant", "growing", "cohort", "fading")
_COHORT_EFFECTS: Final[dict[int, float]] = {5: 2.0, 10: 1.0, 15: 0.5}


@dataclass(frozen=True, slots=True)
class EventStudyRow:
    """The figure's group-time estimate at one exposure, beside the truth."""

    exposure: int
    true_effect: float
    estimate: float
    standard_error: float


@dataclass(frozen=True, slots=True)
class ScenarioRow:
    """True average effect on the treated and the regression's expectation for one pattern."""

    pattern: str
    true_att: float
    expected_twfe: float
    twfe_sd: float
    group_time_sd: float


@dataclass(frozen=True, slots=True)
class SinglePanel:
    """The article's single growing-effect panel and its noise-free two-by-two comparisons."""

    seed: int
    true_att: float
    twfe: float
    never_treated: float
    not_yet_treated: float
    clean_comparison: float
    forbidden_comparison: float


@dataclass(frozen=True, slots=True)
class StaggeredSummary:
    """The figure's event study and regression, with the closed forms and the article's panel."""

    replications: int
    rows: tuple[EventStudyRow, ...]
    true_att: float
    twfe: float
    expected_twfe: float
    twfe_standard_error: float
    scenarios: tuple[ScenarioRow, ...]
    single_panel: SinglePanel


def cohorts() -> NDArray[np.int64]:
    """Adoption period of each unit, ``-1`` for the never treated."""
    return np.array(
        [g for g in ADOPTION_PERIODS for _ in range(COHORT_SIZE)] + [NEVER_TREATED] * COHORT_SIZE,
        dtype=np.int64,
    )


def _grid() -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    t = np.arange(PERIODS)[None, :]
    g = cohorts()[:, None]
    return t, g, (g >= 0) & (t >= g)


def treatment_matrix() -> NDArray[np.float64]:
    """One where a unit has adopted by that period, zero elsewhere."""
    treated: NDArray[np.float64] = _grid()[2].astype(float)
    return treated


def effect_matrix(pattern: str = "growing") -> NDArray[np.float64]:
    """True treatment effect of every unit in every period under one of the article's patterns."""
    t, g, treated = _grid()
    k = t - g
    if pattern == "growing":
        values = 0.2 * (t - g + 1)
    elif pattern == "constant":
        values = np.ones_like(k, dtype=float)
    elif pattern == "cohort":
        by_cohort = np.select([g == c for c in _COHORT_EFFECTS], list(_COHORT_EFFECTS.values()))
        values = np.broadcast_to(by_cohort, k.shape).astype(float)
    elif pattern == "fading":
        values = np.where(k < 8, 0.3 * (k + 1), 0.3 * 8 - 0.6 * (k - 7))
    else:
        raise ValueError(f"pattern must be one of {PATTERNS}")
    effect: NDArray[np.float64] = np.where(treated, values, 0.0)
    return effect


def simulate_panel(
    rng: np.random.Generator, pattern: str = "growing", *, noise_sd: float = NOISE_SD
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Draw one panel: unit effects, period effects, then the noise, as the website does.

    Returns the outcome, the treatment dummy and the true effect. A noise-free
    panel (``noise_sd = 0``) still consumes the noise draws, as the article's does.
    """
    sigma = non_negative(noise_sd, name="noise_sd")
    tau = effect_matrix(pattern)
    unit_fe = rng.normal(0, UNIT_SD, UNITS)
    time_fe = np.linspace(*PERIOD_TREND, PERIODS) + rng.normal(0, PERIOD_SD, PERIODS)
    outcome: NDArray[np.float64] = (
        unit_fe[:, None] + time_fe[None, :] + tau + rng.normal(0, sigma, (UNITS, PERIODS))
    )
    return outcome, treatment_matrix(), tau


def _panel(outcome: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(outcome, dtype=np.float64)
    if values.shape != (UNITS, PERIODS):
        raise ValueError(f"outcome must be a {UNITS} by {PERIODS} panel")
    if not np.all(np.isfinite(values)):
        raise ValueError("outcome must be finite")
    return values


def _demean(values: NDArray[np.float64]) -> NDArray[np.float64]:
    within: NDArray[np.float64] = (
        values - values.mean(1, keepdims=True) - values.mean(0, keepdims=True) + values.mean()
    )
    return within


def _twfe(outcome: NDArray[np.float64], treatment: NDArray[np.float64]) -> float:
    y, d = _demean(outcome), _demean(treatment)
    return float((d * y).sum() / (d**2).sum())


def twfe(outcome: ArrayLike, treatment: ArrayLike) -> float:
    """Coefficient on the treatment dummy with unit and period fixed effects."""
    y = _panel(outcome)
    d = np.asarray(treatment, dtype=np.float64)
    if d.shape != y.shape:
        raise ValueError("treatment must have the outcome's shape")
    if float((_demean(d) ** 2).sum()) == 0.0:
        raise ValueError("treatment must vary beyond the unit and period effects")
    return _twfe(y, d)


def _controls(cohort_of: NDArray[np.int64], period: int, control: Control) -> NDArray[np.bool_]:
    if control == "not_yet":
        mask: NDArray[np.bool_] = (cohort_of == NEVER_TREATED) | (cohort_of > period)
    elif control == "never":
        mask = cohort_of == NEVER_TREATED
    else:
        raise ValueError("control must be 'not_yet' or 'never'")
    return mask


def _cell(cohort: int, period: int) -> tuple[int, int]:
    g = count(cohort, name="cohort", minimum=1)
    t = count(period, name="period")
    if g not in ADOPTION_PERIODS:
        raise ValueError(f"cohort must be one of {ADOPTION_PERIODS}")
    if not g <= t < PERIODS:
        raise ValueError("period must lie between the cohort's adoption and the panel's end")
    return g, t


def _att(y: NDArray[np.float64], g: int, t: int, control: Control) -> float:
    cohort_of = cohorts()
    tr = cohort_of == g
    co = _controls(cohort_of, t, control)
    return float((y[tr, t] - y[tr, g - 1]).mean() - (y[co, t] - y[co, g - 1]).mean())


def group_time_att(
    outcome: ArrayLike, cohort: int, period: int, *, control: Control = "not_yet"
) -> float:
    """Cohort ``g``'s change from ``g - 1`` to ``t`` minus the same change among controls."""
    g, t = _cell(cohort, period)
    return _att(_panel(outcome), g, t, control)


def _event_study(
    y: NDArray[np.float64], exposures: tuple[int, ...], control: Control
) -> list[float]:
    estimates = []
    for k in exposures:
        values = [_att(y, g, g + k, control) for g in ADOPTION_PERIODS if g + k < PERIODS]
        if not values:
            raise ValueError("no cohort is observed at that exposure")
        estimates.append(float(np.mean(values)))
    return estimates


def event_study(
    outcome: ArrayLike, exposures: tuple[int, ...] = EXPOSURES, *, control: Control = "not_yet"
) -> tuple[float, ...]:
    """Group-time effects averaged over the cohorts observed at each exposure."""
    steps = tuple(count(k, name="exposure") for k in exposures)
    return tuple(_event_study(_panel(outcome), steps, control))


def group_time_average(outcome: ArrayLike, *, control: Control = "not_yet") -> float:
    """Every post-adoption group-time effect, weighted by cohort size: the overall estimate."""
    y = _panel(outcome)
    cells = [(g, t) for g in ADOPTION_PERIODS for t in range(g, PERIODS)]
    atts = [_att(y, g, t, control) for g, t in cells]
    return float(np.average(atts, weights=[COHORT_SIZE] * len(cells)))


def did_2x2(outcome: ArrayLike, treated: int, control: int, pre: int, post: int) -> float:
    """Change in one cohort minus the change in another between two periods (``-1``: never)."""
    y = _panel(outcome)
    cohort_of = cohorts()
    groups = set(ADOPTION_PERIODS) | {NEVER_TREATED}
    if treated not in groups or control not in groups or treated == control:
        raise ValueError("treated and control must be two different cohorts")
    first, second = count(pre, name="pre"), count(post, name="post")
    if not first < second < PERIODS:
        raise ValueError("pre must come before post, within the panel")
    tr, co = cohort_of == treated, cohort_of == control
    return float(
        (y[tr, second].mean() - y[tr, first].mean()) - (y[co, second].mean() - y[co, first].mean())
    )


def _att_weights(g: int, t: int, control: Control) -> NDArray[np.float64]:
    cohort_of = cohorts()
    tr = cohort_of == g
    co = _controls(cohort_of, t, control)
    weights = np.zeros((UNITS, PERIODS))
    weights[tr, t] += 1 / tr.sum()
    weights[tr, g - 1] -= 1 / tr.sum()
    weights[co, t] -= 1 / co.sum()
    weights[co, g - 1] += 1 / co.sum()
    return weights


def event_study_weights(exposure: int, *, control: Control = "not_yet") -> NDArray[np.float64]:
    """Weights ``W`` with ``event study = sum(W Y)`` at one exposure."""
    k = count(exposure, name="exposure")
    parts = [_att_weights(g, g + k, control) for g in ADOPTION_PERIODS if g + k < PERIODS]
    if not parts:
        raise ValueError("no cohort is observed at that exposure")
    weights: NDArray[np.float64] = sum(parts, np.zeros((UNITS, PERIODS))) / len(parts)
    return weights


def group_time_average_weights(*, control: Control = "not_yet") -> NDArray[np.float64]:
    """Weights ``W`` with ``overall group-time estimate = sum(W Y)``."""
    cells = [(g, t) for g in ADOPTION_PERIODS for t in range(g, PERIODS)]
    total = sum((_att_weights(g, t, control) for g, t in cells), np.zeros((UNITS, PERIODS)))
    weights: NDArray[np.float64] = total / len(cells)
    return weights


def true_att(pattern: str = "growing") -> float:
    """Average true effect over the treated unit-periods."""
    return float(effect_matrix(pattern)[treatment_matrix() == 1].mean())


def expected_twfe(pattern: str = "growing") -> float:
    """Expectation of the regression coefficient: ``sum(D~ tau) / sum(D~^2)``."""
    d = _demean(treatment_matrix())
    return float((d * effect_matrix(pattern)).sum() / (d**2).sum())


def twfe_sd(noise_sd: float = NOISE_SD) -> float:
    """Spread of the regression coefficient across panels: ``sigma / sqrt(sum(D~^2))``."""
    d = _demean(treatment_matrix())
    return non_negative(noise_sd, name="noise_sd") / sqrt(float((d**2).sum()))


def scenario_row(pattern: str) -> ScenarioRow:
    """The truth and the regression's expectation and spread under one effect pattern."""
    return ScenarioRow(
        pattern=pattern,
        true_att=true_att(pattern),
        expected_twfe=expected_twfe(pattern),
        twfe_sd=twfe_sd(),
        group_time_sd=NOISE_SD * sqrt(float((group_time_average_weights() ** 2).sum())),
    )


def simulate_event_study(
    seed: int = SEED, *, replications: int = REPLICATIONS
) -> tuple[NDArray[np.float64], float, float]:
    """The figure's loop: event-study estimates, mean regression coefficient, mean true effect."""
    rng = np.random.default_rng(count(seed, name="seed"))
    draws = count(replications, name="replications", minimum=1)
    estimates = np.zeros(len(EXPOSURES))
    coefficients, effects = [], []
    for _ in range(draws):
        y, d, tau = simulate_panel(rng)
        coefficients.append(_twfe(y, d))
        effects.append(tau[d == 1].mean())
        for k, value in zip(EXPOSURES, _event_study(y, EXPOSURES, "not_yet"), strict=True):
            estimates[k] += value / draws
    return estimates, float(np.mean(coefficients)), float(np.mean(effects))


def single_panel() -> SinglePanel:
    """The article's panel from seed 1 and its noise-free panel from seed 2."""
    y, d, tau = simulate_panel(np.random.default_rng(SINGLE_PANEL_SEED))
    noise_free, _, _ = simulate_panel(np.random.default_rng(NOISE_FREE_SEED), noise_sd=0.0)
    return SinglePanel(
        seed=SINGLE_PANEL_SEED,
        true_att=float(tau[d == 1].mean()),
        twfe=_twfe(y, d),
        never_treated=group_time_average(y, control="never"),
        not_yet_treated=group_time_average(y, control="not_yet"),
        clean_comparison=did_2x2(noise_free, 10, NEVER_TREATED, 9, 14),
        forbidden_comparison=did_2x2(noise_free, 15, 5, 14, 19),
    )


def example_payload() -> StaggeredSummary:
    """Return the figure's event study (200 panels), the closed forms and the article's panel."""
    estimates, coefficient, effect = simulate_event_study()
    rows = tuple(
        EventStudyRow(
            exposure=k,
            true_effect=0.2 * (k + 1),
            estimate=float(estimates[k]),
            standard_error=NOISE_SD
            * sqrt(float((event_study_weights(k) ** 2).sum()) / REPLICATIONS),
        )
        for k in EXPOSURES
    )
    return StaggeredSummary(
        replications=REPLICATIONS,
        rows=rows,
        true_att=effect,
        twfe=coefficient,
        expected_twfe=expected_twfe(),
        twfe_standard_error=twfe_sd() / sqrt(REPLICATIONS),
        scenarios=tuple(scenario_row(pattern) for pattern in PATTERNS),
        single_panel=single_panel(),
    )
