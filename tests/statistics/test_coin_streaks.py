"""Check the streak calculations against enumerated records.

Both quantities are checked by writing out every possible sequence and summing
its probability, which does not use the recursion or the posterior formula the
model evaluates.
"""

from itertools import pairwise, product

import pytest

from blog_reproducibility.statistics.coin_streaks import (
    BIASED_COINS,
    example_payload,
    next_head_probabilities,
    run_probability,
)

SUMMARY = example_payload()


def test_run_probability_matches_enumerated_records() -> None:
    """Every record of up to eight tosses is enumerated and weighted."""
    for tosses in (0, 1, 4, 8):
        for length in (1, 3, 4):
            for heads in (0.3, 0.5):
                total = 0.0
                for sequence in product((0, 1), repeat=tosses):
                    text = "".join(str(value) for value in sequence)
                    if "1" * length in text:
                        total += heads ** sum(sequence) * (1 - heads) ** (tosses - sum(sequence))
                assert run_probability(tosses, length, heads) == pytest.approx(total, abs=1e-12)


def test_run_probability_boundaries() -> None:
    """A record shorter than the run can never contain it; a certain coin always does."""
    assert run_probability(0, 1) == 0.0
    assert run_probability(3, 4) == 0.0
    assert run_probability(4, 4) == pytest.approx(0.5**4)
    assert run_probability(10, 4, 1.0) == 1.0
    assert run_probability(10, 4, 0.0) == 0.0

    # A longer record has more chances, never fewer.
    values = [run_probability(tosses, 4) for tosses in (4, 10, 20, 50, 100)]
    assert all(left < right for left, right in pairwise(values))


def test_unknown_coin_prediction_matches_enumerated_sequences() -> None:
    """The posterior prediction is recomputed from the sequence probabilities."""
    sequences = list(product((0, 1), repeat=5))
    low, high = BIASED_COINS

    for heads in range(5):
        denominator = numerator = 0.0
        for sequence in sequences:
            total_heads = sum(sequence)
            weight = sum(
                0.5 * bias**total_heads * (1 - bias) ** (5 - total_heads) for bias in (low, high)
            )
            if all(sequence[index] == 1 for index in range(heads)):
                denominator += weight
                if sequence[heads] == 1:
                    numerator += weight

        assert next_head_probabilities(heads).unknown_coin == pytest.approx(numerator / denominator)


def test_the_three_mechanisms_move_in_different_directions() -> None:
    """A fair coin is unmoved, a bag is pushed down, an unknown coin is pushed up."""
    rows = [next_head_probabilities(heads) for heads in range(5)]

    assert all(row.fair_coin == 0.5 for row in rows)
    assert all(
        left.bag_without_replacement > right.bag_without_replacement
        for left, right in pairwise(rows)
    )
    assert all(left.unknown_coin < right.unknown_coin for left, right in pairwise(rows))

    # They all start from the same place, before any streak is observed.
    assert rows[0].fair_coin == rows[0].bag_without_replacement == rows[0].unknown_coin == 0.5
    # And the unknown coin never passes the more biased of the two candidates.
    assert all(row.unknown_coin < max(BIASED_COINS) for row in rows)


def test_the_bag_empties_as_the_streak_runs() -> None:
    """Five heads drawn from five leaves none, so the next draw cannot be heads."""
    assert next_head_probabilities(5).bag_without_replacement == 0.0
    assert next_head_probabilities(4).bag_without_replacement == pytest.approx(1 / 6)


def test_published_numbers() -> None:
    """Every value the article prints should come back unchanged."""
    rows = SUMMARY.mechanisms

    assert [round(row.bag_without_replacement, 6) for row in rows] == [
        0.5,
        0.444444,
        0.375,
        0.285714,
        0.166667,
    ]
    assert [round(row.unknown_coin, 6) for row in rows] == [
        0.5,
        0.625,
        0.7,
        0.732143,
        0.743902,
    ]
    assert SUMMARY.four_head_run_probability == {
        "4": pytest.approx(0.0625),
        "20": pytest.approx(0.478019, abs=5e-7),
        "50": pytest.approx(0.827414, abs=5e-7),
        "100": pytest.approx(0.972715, abs=5e-7),
    }

    # The article's point: in a hundred tosses the run is nearly certain.
    assert SUMMARY.four_head_run_probability["100"] > 0.97


def test_invalid_inputs() -> None:
    """Negative records, zero-length runs, and impossible streaks are rejected."""
    with pytest.raises(ValueError):
        run_probability(-1, 4)
    with pytest.raises(ValueError):
        run_probability(10, 0)
    with pytest.raises(ValueError):
        run_probability(10, 4, 1.5)
    with pytest.raises(ValueError):
        next_head_probabilities(6)
    with pytest.raises(ValueError):
        next_head_probabilities(-1)
    with pytest.raises(TypeError):
        next_head_probabilities(True)
