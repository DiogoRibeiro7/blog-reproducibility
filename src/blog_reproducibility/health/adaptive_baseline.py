"""An exponentially weighted personal baseline, for a healthcare draft.

A device that flags "unusual for you" has to decide what usual is, and it has to
keep deciding as the person changes. The simplest rule keeps a running baseline
and moves it a fraction ``alpha`` of the way towards each new observation.

The draft's point is what that fraction costs. A large ``alpha`` adapts quickly
and stops flagging a change almost immediately, because the baseline chases the
new level. A small one keeps flagging for weeks. Neither is right in general,
and the rule scores *before* it updates, so the first observation after a step
is always compared against the old baseline.

Missing observations produce no score and no update. That is deliberate: a gap
in the record is not evidence about the person, and treating it as a zero score
would drag the baseline towards nothing.

The recursion has an exact solution. After ``n`` steps of a constant observation
``y`` from a baseline of zero, the baseline is ``y * (1 - (1 - alpha)^n)``, and
the tests check the recursion against it.
"""

from dataclasses import dataclass

from blog_reproducibility.common.validation import probability, real

__all__ = [
    "BaselineRow",
    "adaptive_baseline",
    "example_payload",
    "exact_baseline",
]


@dataclass(frozen=True, slots=True)
class BaselineRow:
    """One observation: the baseline before it, its score, and the baseline after.

    ``score`` and ``observation`` are ``None`` when the observation is missing.
    """

    observation: float | None
    before: float
    score: float | None
    after: float


def adaptive_baseline(
    observations: tuple[float | None, ...],
    alpha: float,
    *,
    initial: float = 0.0,
) -> tuple[BaselineRow, ...]:
    """Score each observation against the current baseline, then update it."""
    rate = probability(alpha, name="alpha")
    baseline = real(initial, name="initial")

    rows: list[BaselineRow] = []
    for observation in observations:
        before = baseline
        if observation is None:
            score = None
        else:
            value = real(observation, name="observation")
            score = value - before
            baseline += rate * score
        rows.append(
            BaselineRow(
                observation=observation,
                before=before,
                score=score,
                after=baseline,
            )
        )
    return tuple(rows)


def exact_baseline(steps: int, alpha: float, level: float, *, initial: float = 0.0) -> float:
    """Closed form for the baseline after a constant observation repeated.

    The recursion ``b <- b + alpha (y - b)`` is geometric, so after ``n`` steps
    the baseline is ``y + (b0 - y)(1 - alpha)^n``.
    """
    if steps < 0:
        raise ValueError("steps must be zero or greater")
    rate = probability(alpha, name="alpha")
    target = real(level, name="level")
    start = real(initial, name="initial")

    return target + (start - target) * (1 - rate) ** steps


def example_payload() -> dict[str, tuple[BaselineRow, ...]]:
    """Return the four learning rates the draft compares over thirty steps."""
    observations: tuple[float | None, ...] = (4.0,) * 30
    return {str(alpha): adaptive_baseline(observations, alpha) for alpha in (0.0, 0.02, 0.1, 0.25)}
