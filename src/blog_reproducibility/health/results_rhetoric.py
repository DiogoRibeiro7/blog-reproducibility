"""Retention arithmetic and the testimonial-wall calculation.

Two things are computed for the article on results as rhetoric.

The first is arithmetic on the retention and weight-loss figures Finley et al.
(2007) published for 60,164 clients of a commercial programme. The better the
reported result, the fewer clients it describes.

The second is exact: the expected mean of the best ``k`` results among ``n``
clients whose individual results are standard normal. That is what a wall of
testimonials shows. A client with result ``x`` is on the wall when at most
``k - 1`` of the others did better, so the expected total on the wall is

    n * integral x * phi(x) * P(Binomial(n - 1, P(X > x)) <= k - 1) dx,

evaluated on a fixed grid. No approximation of the order statistics is used:
the article's point is how large the selected mean is even with no effect at
all, so the number should not depend on an approximation being good.
"""

from dataclasses import dataclass
from math import exp, lgamma, log, log1p
from random import Random
from statistics import NormalDist
from typing import Final

from blog_reproducibility.common.validation import count, positive, real

__all__ = [
    "FINLEY_COHORT_LOSS",
    "FINLEY_ENROLLED",
    "FINLEY_RETAINED",
    "KROGSBOLL",
    "SCALE",
    "WALL",
    "FinleySummary",
    "RhetoricSummary",
    "clients_needed",
    "example_payload",
    "finley_summary",
    "wall_mean",
    "wall_mean_by_simulation",
]

STANDARD: Final[NormalDist] = NormalDist()

# Finley CE, Barlow CE, Greenway FL, Rock CL, Rolls BJ, Blair SN. Retention rates
# and weight loss in a commercial weight loss program. Int J Obes
# 2007;31(2):292-298. Figures as given in the abstract.
FINLEY_ENROLLED: Final[int] = 60_164
FINLEY_RETAINED: Final[dict[int, float]] = {0: 100.0, 4: 73.0, 13: 42.0, 26: 22.0, 52: 6.6}
# Mean and SD of weight lost, % of initial body weight, among clients still
# attending at that week.
FINLEY_COHORT_LOSS: Final[dict[int, tuple[float, float]]] = {
    13: (8.3, 3.3),
    26: (12.6, 5.1),
    52: (15.6, 7.5),
}
FINLEY_EARLY_LEAVERS_LOSS: Final[tuple[float, float]] = (1.1, 1.6)
# Krogsboll LT, Hrobjartsson A, Gotzsche PC. Spontaneous improvement in randomised
# clinical trials. BMC Med Res Methodol 2009;9:1. Change from baseline in standard
# deviations, 37 three-armed trials.
KROGSBOLL: Final[dict[str, float]] = {
    "no treatment": 0.24,
    "placebo": 0.44,
    "active treatment": 1.01,
}

WALL: Final[int] = 20
# An SD of 5.1% of body weight, taken from the 26-week cohort, gives the standard
# deviation a size so the wall can be quoted in percentage points.
SCALE: Final[float] = FINLEY_COHORT_LOSS[26][1]


@dataclass(frozen=True, slots=True)
class FinleySummary:
    """What the published retention figures imply in client counts."""

    left_within_four_weeks_percent: float
    left_before_a_year_percent: float
    still_attending_at_a_year: int
    left_within_four_weeks: int
    enrolled_per_client_still_attending: float


@dataclass(frozen=True, slots=True)
class RhetoricSummary:
    """Every number the article reports."""

    finley: FinleySummary
    wall_by_client_count: dict[int, float]
    wall_with_effect_1sd_1000_clients: float
    wall_with_effect_half_sd_1000_clients: float
    clients_at_which_no_effect_matches: int
    share_of_the_wall_that_is_selection: float
    wall_as_percent_of_body_weight: dict[int, float]
    share_of_change_the_treatment_explains: float
    before_after_change_over_treatment_effect: float


def finley_summary() -> FinleySummary:
    """Turn the published retention percentages into client counts."""
    return FinleySummary(
        left_within_four_weeks_percent=round(100 - FINLEY_RETAINED[4], 1),
        left_before_a_year_percent=round(100 - FINLEY_RETAINED[52], 1),
        still_attending_at_a_year=round(FINLEY_ENROLLED * FINLEY_RETAINED[52] / 100),
        left_within_four_weeks=round(FINLEY_ENROLLED * (100 - FINLEY_RETAINED[4]) / 100),
        enrolled_per_client_still_attending=round(100 / FINLEY_RETAINED[52], 1),
    )


def _log_binomial_pmf(successes: int, trials: int, probability: float) -> float:
    """Log of the binomial probability mass, computed with log-gamma for stability."""
    return (
        lgamma(trials + 1)
        - lgamma(successes + 1)
        - lgamma(trials - successes + 1)
        + successes * log(probability)
        + (trials - successes) * log1p(-probability)
    )


def wall_mean(
    clients: int,
    wall: int = WALL,
    effect: float = 0.0,
    *,
    step: float = 0.002,
) -> float:
    """Expected mean of the best ``wall`` results among ``clients``, in SDs.

    Individual results are ``Normal(effect, 1)``, larger being better. The
    effect enters as a pure shift, so a programme that works moves the wall by
    exactly its effect and no more.
    """
    number = count(clients, name="clients", minimum=1)
    places = count(wall, name="wall", minimum=1)
    shift = real(effect, name="effect")
    spacing = positive(step, name="step")

    if number < places:
        raise ValueError("the wall cannot hold more results than there are clients")
    if number == places:
        return shift

    total = 0.0
    position = -9.0
    while position < 9.0:
        tail = 1 - STANDARD.cdf(position) if position < 0 else STANDARD.cdf(-position)
        expected_better = (number - 1) * tail
        # Beyond this the binomial weight is below 1e-30 and adds nothing.
        if expected_better < places + 40 * places**0.5 + 40:
            if tail <= 0.0:
                on_wall = 1.0
            elif tail >= 1.0:
                # Everyone else did better, and there are more of them than places.
                on_wall = 0.0
            else:
                on_wall = sum(
                    exp(_log_binomial_pmf(better, number - 1, tail))
                    for better in range(min(places, number))
                )
            total += position * STANDARD.pdf(position) * min(on_wall, 1.0) * spacing
        position += spacing
    return shift + number * total / places


def wall_mean_by_simulation(
    clients: int,
    wall: int = WALL,
    effect: float = 0.0,
    *,
    repeats: int = 400,
    seed: int = 20260919,
) -> float:
    """Estimate the same quantity by drawing walls, as an independent check.

    The generator is built from ``seed`` here, so the estimate repeats and never
    depends on global random state.
    """
    number = count(clients, name="clients", minimum=1)
    places = count(wall, name="wall", minimum=1)
    shift = real(effect, name="effect")
    draws = count(repeats, name="repeats", minimum=1)

    if number < places:
        raise ValueError("the wall cannot hold more results than there are clients")

    rng = Random(count(seed, name="seed"))
    total = 0.0
    for _ in range(draws):
        results = sorted(rng.gauss(shift, 1.0) for _ in range(number))
        total += sum(results[-places:]) / places
    return total / draws


def clients_needed(target: float, effect: float = 0.0, wall: int = WALL) -> int:
    """Smallest client count, to two significant figures, whose wall reaches ``target``.

    ``wall_mean`` rises with the number of clients, so the search is a bisection.
    """
    goal = real(target, name="target")
    shift = real(effect, name="effect")
    places = count(wall, name="wall", minimum=1)

    low, high = places, 10**8
    while high - low > max(1, low // 1000):
        middle = (low + high) // 2
        if wall_mean(middle, places, shift) >= goal:
            high = middle
        else:
            low = middle

    digits = len(str(high)) - 2
    return round(high, -digits) if digits > 0 else high


def example_payload() -> RhetoricSummary:
    """Return the numbers the article reports."""
    sizes = (200, 1_000, 10_000, 100_000, 1_000_000)
    walls = {size: round(wall_mean(size), 2) for size in sizes}
    with_effect = round(wall_mean(1_000, effect=1.0), 2)
    active = KROGSBOLL["active treatment"]
    own_effect = active - KROGSBOLL["placebo"]

    return RhetoricSummary(
        finley=finley_summary(),
        wall_by_client_count=walls,
        wall_with_effect_1sd_1000_clients=with_effect,
        wall_with_effect_half_sd_1000_clients=round(wall_mean(1_000, effect=0.5), 2),
        clients_at_which_no_effect_matches=clients_needed(with_effect),
        share_of_the_wall_that_is_selection=round(walls[1_000] / with_effect, 2),
        wall_as_percent_of_body_weight={
            size: round(walls[size] * SCALE, 1) for size in (1_000, 100_000)
        },
        share_of_change_the_treatment_explains=round(own_effect / active, 2),
        before_after_change_over_treatment_effect=round(active / own_effect, 1),
    )
