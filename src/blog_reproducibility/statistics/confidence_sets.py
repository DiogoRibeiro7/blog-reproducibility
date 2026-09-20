"""Test inversion for the article on confidence sets that are not intervals.

The working model is ``Y ~ N((theta^2 - 1)^2, sigma^2)`` with ``sigma = 0.08``.
Inverting the two-sided z test at level 0.05 accepts

    {theta : |y - (theta^2 - 1)^2| < z * sigma},

which has a closed form. Depending on the observation it is empty, two
intervals, three, or four, so the set is computed as a tuple of components
rather than as a pair of endpoints.

Three further pieces of the article are here because each one is a claim with a
number attached: the convex hull and what it costs, Fieller's set for a ratio of
means, and the projection over a nuisance parameter where a local optimiser
returns a maximum that is not the supremum.

Everything is standard library. The local optimiser is written out rather than
taken from a library, because the article's point is precisely that it stops in
the basin it started in, and that behaviour should be inspectable.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import erfc, sqrt
from random import Random
from statistics import NormalDist

from blog_reproducibility.common.validation import count, positive, probability, real

__all__ = [
    "ConfidenceSetExamples",
    "CoverageSummary",
    "FiellerSet",
    "Interval",
    "ProfileSearch",
    "SIGMA",
    "Z",
    "accepted_set",
    "BisectionFloor",
    "bisection_floor",
    "contains",
    "convex_hull",
    "coverage_summary",
    "example_payload",
    "fieller_set",
    "grid_miss_probability",
    "local_minimum",
    "mean_function",
    "measure",
    "p_value",
    "profile_projection_bound",
    "profile_search",
]

SIGMA = 0.08
Z = NormalDist().inv_cdf(0.975)

Interval = tuple[float, float]


@dataclass(frozen=True, slots=True)
class FiellerSet:
    """Fieller's 95% set for a ratio of means, with its shape named.

    ``kind`` is ``"interval"`` when the denominator is significantly non-zero,
    ``"exclusive"`` when the set is the complement of a bounded interval, and
    ``"real line"`` when no value of the ratio is rejected.
    """

    kind: str
    bounds: Interval | None


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Simulated coverage and size of the set against its convex hull."""

    true_theta: float
    trials: int
    set_coverage: float
    hull_coverage: float
    mean_set_measure: float
    mean_hull_length: float
    component_counts: dict[int, float]


@dataclass(frozen=True, slots=True)
class ProfileSearch:
    """Where a local search started, where it stopped, and what it decided."""

    start: float
    minimizer: float
    statistic: float
    p_value: float
    rejects: bool


@dataclass(frozen=True, slots=True)
class BisectionFloor:
    """Where bisection towards a set boundary runs out of doubles."""

    halvings: int
    final_width: float
    boundary: float


@dataclass(frozen=True, slots=True)
class GridMissRow:
    """Chance that one grid step misses each component width."""

    step: float
    outer_component: float
    inner_component: float
    y_004_component: float


@dataclass(frozen=True, slots=True)
class ConfidenceSetExamples:
    """Every number the confidence-sets article reports."""

    accepted_sets: dict[str, tuple[Interval, ...]]
    hull_for_004: Interval | None
    hull_length_ratio_for_004: float
    p_value_at_zero_for_004: float
    p_value_at_one_for_004: float
    component_widths_for_050: tuple[float, ...]
    grid_miss: tuple[GridMissRow, ...]
    bisection: BisectionFloor
    fieller: dict[str, FiellerSet]
    profile_searches: tuple[ProfileSearch, ...]
    profile_projection_bound_right: float
    profile_projection_bound_left: float
    coverage: tuple[CoverageSummary, ...]


def mean_function(theta: float) -> float:
    """Return the model mean ``(theta^2 - 1)^2``."""
    candidate = real(theta, name="theta")
    return (candidate**2 - 1.0) ** 2


def p_value(theta: float, y: float, *, sigma: float = SIGMA) -> float:
    """Two-sided p-value of the candidate ``theta`` against the observation ``y``."""
    observation = real(y, name="y")
    spread = positive(sigma, name="sigma")
    return erfc(abs(observation - mean_function(theta)) / spread / sqrt(2.0))


def accepted_set(y: float, *, sigma: float = SIGMA, z: float = Z) -> tuple[Interval, ...]:
    """Return the accepted set as sorted, disjoint, open intervals.

    An empty tuple means every candidate was rejected. That happens with
    probability 0.025 at the true values ``theta = ±1`` and is the procedure's
    stated error rate, not a malfunction to be repaired.
    """
    observation = real(y, name="y")
    spread = positive(sigma, name="sigma")
    critical = positive(z, name="z")

    low = max(observation - critical * spread, 0.0)
    high = observation + critical * spread
    if high <= 0.0:
        return ()

    # The accepted values of |theta^2 - 1| lie in (a, b), which gives one or two
    # intervals for theta^2 and then up to four for theta.
    a, b = sqrt(low), sqrt(high)
    squares: list[Interval] = [(1.0 + a, 1.0 + b)]
    if max(1.0 - b, 0.0) < 1.0 - a:
        squares.append((max(1.0 - b, 0.0), 1.0 - a))

    pieces: list[Interval] = []
    for lower, upper in squares:
        if lower == 0.0:
            pieces.append((-sqrt(upper), sqrt(upper)))
        else:
            pieces.append((-sqrt(upper), -sqrt(lower)))
            pieces.append((sqrt(lower), sqrt(upper)))

    merged: list[Interval] = []
    for lower, upper in sorted(pieces):
        # The two branches touch when a = 0, where the set is connected.
        if merged and lower <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(upper, merged[-1][1]))
        else:
            merged.append((lower, upper))
    return tuple(merged)


def convex_hull(pieces: tuple[Interval, ...]) -> Interval | None:
    """Return the smallest interval containing every component, or ``None`` if empty."""
    if not pieces:
        return None
    return (min(low for low, _ in pieces), max(high for _, high in pieces))


def measure(pieces: tuple[Interval, ...]) -> float:
    """Return the total length of the components."""
    return sum(high - low for low, high in pieces)


def contains(pieces: tuple[Interval, ...], theta: float) -> bool:
    """Return whether any component contains ``theta``."""
    return any(low <= theta <= high for low, high in pieces)


def fieller_set(
    numerator: float,
    denominator: float,
    *,
    se_numerator: float = 0.3,
    se_denominator: float = 0.3,
    z: float = Z,
) -> FiellerSet:
    """Return Fieller's set for ``numerator / denominator``, with its shape.

    The set solves a quadratic inequality in the ratio whose leading
    coefficient is ``denominator^2 - z^2 * se_denominator^2``. When the
    denominator is not significantly different from zero that coefficient turns
    negative or vanishes, and the set stops being an interval. Estimates are
    treated as independent, which is the case the article discusses.
    """
    a = real(numerator, name="numerator")
    b = real(denominator, name="denominator")
    sa = positive(se_numerator, name="se_numerator")
    sb = positive(se_denominator, name="se_denominator")
    critical = positive(z, name="z")

    quadratic = b * b - critical**2 * sb * sb
    linear = -2.0 * a * b
    constant = a * a - critical**2 * sa * sa
    discriminant = linear * linear - 4.0 * quadratic * constant

    if discriminant <= 0.0:
        # No real root: either every ratio is accepted or none is.
        return FiellerSet(kind="real line", bounds=None)

    root = sqrt(discriminant)
    first = (-linear - root) / (2.0 * quadratic)
    second = (-linear + root) / (2.0 * quadratic)
    lower, upper = min(first, second), max(first, second)

    if quadratic > 0.0:
        return FiellerSet(kind="interval", bounds=(lower, upper))
    return FiellerSet(kind="exclusive", bounds=(lower, upper))


def grid_miss_probability(width: float, step: float) -> float:
    """Probability that a grid of the given step contains no point of a component.

    With an offset unrelated to the problem, a component of width ``w`` is
    missed with probability ``1 - w / h`` when ``h > w``, and never when the
    step is at least as fine as the component.
    """
    component = positive(width, name="width")
    spacing = positive(step, name="step")
    if spacing <= component:
        return 0.0
    return 1.0 - component / spacing


def local_minimum(
    objective: Callable[[float], float],
    start: float,
    *,
    step: float = 0.05,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Return the local minimiser found by descending from ``start``.

    The search walks downhill with a doubling step until the objective rises,
    which brackets a minimum, and then narrows the bracket by golden section.
    It is deliberately local: given a start in the wrong basin it converges
    to the wrong basin, which is the behaviour the article is about.
    """
    if not callable(objective):
        raise TypeError("objective must be callable")
    origin = real(start, name="start")
    walk = positive(step, name="step")
    limit = positive(tolerance, name="tolerance")
    iterations = count(max_iterations, name="max_iterations", minimum=1)

    def value(point: float) -> float:
        return real(objective(point), name="objective(x)")

    here = value(origin)
    direction = 1.0 if value(origin + walk) < here else -1.0
    if value(origin + direction * walk) >= here:
        return origin  # Already at a local minimum on this scale.

    # Walk downhill with a doubling step until the objective rises.
    previous, current = origin, origin + direction * walk
    span = walk
    for _ in range(iterations):
        span *= 2.0
        nxt = current + direction * span
        if value(nxt) >= value(current):
            break
        previous, current = current, nxt
    else:  # pragma: no cover - the objectives here always turn back up
        return current

    low, high = sorted((previous, current + direction * span))

    # Golden-section search narrows the bracket without needing a derivative.
    ratio = (sqrt(5.0) - 1.0) / 2.0
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    left_value, right_value = value(left), value(right)
    for _ in range(iterations):
        if high - low <= limit:
            break
        if left_value < right_value:
            high, right, right_value = right, left, left_value
            left = high - ratio * (high - low)
            left_value = value(left)
        else:
            low, left, left_value = left, right, right_value
            right = low + ratio * (high - low)
            right_value = value(right)
    return (low + high) / 2.0


def _nuisance_mean(lam: float) -> float:
    """The nuisance function ``g(lambda) = (lambda^2 - 1)^2 - 0.3 lambda``."""
    return (lam**2 - 1.0) ** 2 - 0.3 * lam


def profile_search(
    start: float,
    *,
    psi: float = 0.40,
    y: float = 0.0,
    sigma: float = 0.06,
    z: float = Z,
) -> ProfileSearch:
    """Run the local search for one target value from one starting point."""
    target = real(psi, name="psi")
    observation = real(y, name="y")
    spread = positive(sigma, name="sigma")
    critical = positive(z, name="z")

    def statistic(lam: float) -> float:
        return ((observation - target - _nuisance_mean(lam)) / spread) ** 2

    minimizer = local_minimum(statistic, start)
    value = statistic(minimizer)
    return ProfileSearch(
        start=real(start, name="start"),
        minimizer=minimizer,
        statistic=value,
        p_value=erfc(sqrt(value) / sqrt(2.0)),
        rejects=value > critical**2,
    )


def profile_projection_bound(*, start: float = 1.2, sigma: float = 0.06, z: float = Z) -> float:
    """Return the largest target value the projection accepts from one start.

    The supremum of the p-value is attained where ``psi + g(lambda)`` comes
    closest to the observation, so the bound is ``-g(lambda*) + z * sigma`` at
    the minimiser the search reaches. Starting in the shallow basin returns a
    smaller bound, and the set it produces is too narrow: the error from a
    local optimiser in this step is anti-conservative.
    """
    spread = positive(sigma, name="sigma")
    critical = positive(z, name="z")
    minimizer = local_minimum(_nuisance_mean, real(start, name="start"))
    return -_nuisance_mean(minimizer) + critical * spread


def coverage_summary(
    true_theta: float,
    *,
    trials: int = 20_000,
    sigma: float = SIGMA,
    seed: int = 20260912,
) -> CoverageSummary:
    """Simulate coverage and size of the set and of its convex hull.

    The generator is built from ``seed`` here, so the result does not depend on
    global random state and repeats exactly.
    """
    theta = real(true_theta, name="true_theta")
    draws = count(trials, name="trials", minimum=1)
    spread = positive(sigma, name="sigma")
    rng = Random(count(seed, name="seed"))

    mean = mean_function(theta)
    set_hits = hull_hits = 0
    total_measure = total_length = 0.0
    counts: dict[int, int] = {}

    for _ in range(draws):
        pieces = accepted_set(rng.gauss(mean, spread), sigma=spread)
        counts[len(pieces)] = counts.get(len(pieces), 0) + 1
        total_measure += measure(pieces)

        if contains(pieces, theta):
            set_hits += 1
        hull = convex_hull(pieces)
        if hull is not None:
            total_length += hull[1] - hull[0]
            if hull[0] <= theta <= hull[1]:
                hull_hits += 1

    return CoverageSummary(
        true_theta=theta,
        trials=draws,
        set_coverage=set_hits / draws,
        hull_coverage=hull_hits / draws,
        mean_set_measure=total_measure / draws,
        mean_hull_length=total_length / draws,
        component_counts={size: number / draws for size, number in sorted(counts.items())},
    )


def bisection_floor(
    *,
    low: float = 0.5,
    high: float = 1.0,
    y: float = 0.04,
    sigma: float = SIGMA,
    alpha: float = 0.05,
    max_halvings: int = 200,
) -> BisectionFloor:
    """Bisect towards a set boundary until the midpoint stops moving.

    The loop ends because adjacent doubles have no representable midpoint,
    which is a stall rather than convergence.
    """
    left = real(low, name="low")
    right = real(high, name="high")
    observation = real(y, name="y")
    spread = positive(sigma, name="sigma")
    level = probability(alpha, name="alpha", inclusive=False)
    limit = count(max_halvings, name="max_halvings", minimum=1)

    if (p_value(left, observation, sigma=spread) > level) == (
        p_value(right, observation, sigma=spread) > level
    ):
        raise ValueError("The bracket must straddle a decision boundary")

    halvings = 0
    for _ in range(limit):
        middle = (left + right) / 2.0
        if middle in (left, right):
            break
        halvings += 1
        if (p_value(middle, observation, sigma=spread) > level) == (
            p_value(left, observation, sigma=spread) > level
        ):
            left = middle
        else:
            right = middle
    return BisectionFloor(halvings=halvings, final_width=right - left, boundary=left)


def example_payload() -> ConfidenceSetExamples:
    """Return the numbers the article reports."""
    pieces_004 = accepted_set(0.04)
    hull_004 = convex_hull(pieces_004)
    four = accepted_set(0.50)
    outer = four[0][1] - four[0][0]
    inner = four[1][1] - four[1][0]
    single_004 = measure(pieces_004) / 2.0

    return ConfidenceSetExamples(
        accepted_sets={f"{y:+.2f}": accepted_set(y) for y in (-0.20, 0.04, 0.50, 1.00, 1.20)},
        hull_for_004=hull_004,
        hull_length_ratio_for_004=(
            (hull_004[1] - hull_004[0]) / measure(pieces_004) if hull_004 else 0.0
        ),
        p_value_at_zero_for_004=p_value(0.0, 0.04),
        p_value_at_one_for_004=p_value(1.0, 0.04),
        component_widths_for_050=tuple(high - low for low, high in four),
        grid_miss=tuple(
            GridMissRow(
                step=step,
                outer_component=grid_miss_probability(outer, step),
                inner_component=grid_miss_probability(inner, step),
                y_004_component=grid_miss_probability(single_004, step),
            )
            for step in (0.05, 0.10, 0.20, 0.50)
        ),
        bisection=bisection_floor(),
        fieller={
            "denominator_2.0": fieller_set(1.0, 2.0),
            "denominator_0.5": fieller_set(1.0, 0.5),
            "numerator_0.2_denominator_0.3": fieller_set(0.2, 0.3),
        },
        profile_searches=tuple(profile_search(start) for start in (-1.2, 0.0, 1.2)),
        profile_projection_bound_right=profile_projection_bound(start=1.2),
        profile_projection_bound_left=profile_projection_bound(start=-1.2),
        coverage=tuple(coverage_summary(theta) for theta in (1.0, 0.0, 1.3)),
    )
