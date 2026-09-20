"""What a run of heads means, for the article on randomness owing no reversal.

Two calculations. The first is how often a run of a given length appears
somewhere in a record of tosses, which is what makes streaks look surprising
when they are not: overlapping windows give a long record many chances.

The second is the article's point. A streak licenses no prediction on its own,
because what it implies depends entirely on the mechanism that produced it:

* a coin known to be fair is unmoved by any history;
* draws from a bag without replacement are pushed the other way, because heads
  already taken are heads no longer available;
* a coin of unknown bias is pushed the same way, because the streak is evidence
  about which coin it is.

The run probability is exact. It is computed by carrying the distribution over
"how many trailing heads so far, given the run has not yet appeared" forward one
toss at a time, so no approximation or simulation enters.
"""

from dataclasses import dataclass
from typing import Final

from blog_reproducibility.common.validation import count, probability

__all__ = [
    "BAG_SIZE",
    "BIASED_COINS",
    "MechanismRow",
    "StreakSummary",
    "example_payload",
    "next_head_probabilities",
    "run_probability",
]

# The bag starts with five heads and five tails, drawn without replacement.
BAG_SIZE: Final[int] = 10
# The unknown coin is equally likely to be one of these two.
BIASED_COINS: Final[tuple[float, float]] = (0.25, 0.75)


@dataclass(frozen=True, slots=True)
class MechanismRow:
    """What each mechanism predicts after a run of heads."""

    heads_so_far: int
    fair_coin: float
    bag_without_replacement: float
    unknown_coin: float


@dataclass(frozen=True, slots=True)
class StreakSummary:
    """Every number the article reports."""

    mechanisms: tuple[MechanismRow, ...]
    four_head_run_probability: dict[str, float]


def run_probability(
    tosses: int,
    run_length: int,
    heads_probability: float = 0.5,
) -> float:
    """Probability that a run of heads of that length appears somewhere.

    Windows overlap, so this is not a count of disjoint blocks. The state is the
    number of trailing heads, carried forward only while the target run has not
    yet occurred; whatever mass is left at the end never produced the run.
    """
    number = count(tosses, name="tosses", minimum=0)
    length = count(run_length, name="run_length", minimum=1)
    heads = probability(heads_probability, name="heads_probability")

    states = [1.0] + [0.0] * (length - 1)
    for _ in range(number):
        states = [sum(states) * (1 - heads)] + [mass * heads for mass in states[:-1]]
    return max(0.0, min(1.0, 1 - sum(states)))


def next_head_probabilities(heads: int) -> MechanismRow:
    """Chance of heads next under three mechanisms, after a run of heads."""
    run = count(heads, name="heads", minimum=0)
    if run > 5:
        raise ValueError("heads must be at most 5")

    bag = (BAG_SIZE // 2 - run) / (BAG_SIZE - run)

    # Equal prior on the two coins; the posterior weight follows the likelihood
    # of the run, so the streak is evidence about which coin is in play.
    low, high = BIASED_COINS
    unknown = (low ** (run + 1) + high ** (run + 1)) / (low**run + high**run)

    return MechanismRow(
        heads_so_far=run,
        fair_coin=0.5,
        bag_without_replacement=bag,
        unknown_coin=unknown,
    )


def example_payload() -> StreakSummary:
    """Return the numbers the article reports."""
    return StreakSummary(
        mechanisms=tuple(next_head_probabilities(heads) for heads in range(5)),
        four_head_run_probability={
            str(tosses): run_probability(tosses, 4) for tosses in (4, 20, 50, 100)
        },
    )
