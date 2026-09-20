"""Selection through bottlenecks, for the antibiotic resistance article.

The article's point is that selection changes a composition without anything
growing. A course of antibiotics kills most of a bacterial population; the
resistant fraction is killed less. Both counts fall, and the resistant *share*
rises anyway.

The example starts from 99,900 sensitive and 100 resistant bacteria. Each
bottleneck leaves 1% of the sensitive population and 80% of the resistant one,
so the odds of resistance multiply by exactly 80 every time. Nothing here
models mutation, transfer, or regrowth between courses; the article makes its
argument with selection alone, and adding the rest would obscure it.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "RESISTANT_SURVIVAL",
    "SENSITIVE_SURVIVAL",
    "STARTING_RESISTANT",
    "STARTING_SENSITIVE",
    "ResistanceSummary",
    "SelectionRow",
    "example_payload",
    "selection_rows",
]

STARTING_SENSITIVE: Final[float] = 99_900.0
STARTING_RESISTANT: Final[float] = 100.0
SENSITIVE_SURVIVAL: Final[float] = 0.01
RESISTANT_SURVIVAL: Final[float] = 0.80


@dataclass(frozen=True, slots=True)
class SelectionRow:
    """The population after a number of selective bottlenecks."""

    bottleneck: int
    sensitive: float
    resistant: float
    resistant_share: float

    @property
    def total(self) -> float:
        """The whole population, which falls at every step."""
        return self.sensitive + self.resistant


@dataclass(frozen=True, slots=True)
class ResistanceSummary:
    """Every number the article reports."""

    rows: tuple[SelectionRow, ...]
    odds_multiplier: float


def selection_rows(
    bottlenecks: int = 3,
    *,
    sensitive_survival: float = SENSITIVE_SURVIVAL,
    resistant_survival: float = RESISTANT_SURVIVAL,
) -> tuple[SelectionRow, ...]:
    """Expected counts through a series of selective bottlenecks.

    The first row is the starting population, so ``bottlenecks`` rounds give
    ``bottlenecks + 1`` rows.
    """
    steps = count(bottlenecks, name="bottlenecks", minimum=0)
    sensitive_rate = probability(sensitive_survival, name="sensitive_survival")
    resistant_rate = probability(resistant_survival, name="resistant_survival")

    sensitive, resistant = STARTING_SENSITIVE, STARTING_RESISTANT
    rows: list[SelectionRow] = []
    for step in range(steps + 1):
        rows.append(
            SelectionRow(
                bottleneck=step,
                sensitive=sensitive,
                resistant=resistant,
                resistant_share=resistant / (sensitive + resistant),
            )
        )
        sensitive *= sensitive_rate
        resistant *= resistant_rate
    return tuple(rows)


def example_payload() -> ResistanceSummary:
    """Return the numbers the article reports."""
    return ResistanceSummary(
        rows=selection_rows(),
        odds_multiplier=RESISTANT_SURVIVAL / SENSITIVE_SURVIVAL,
    )
