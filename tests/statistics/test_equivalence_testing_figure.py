"""Tests for the equivalence-testing figure rendering."""

from pathlib import Path

from blog_reproducibility.statistics.equivalence_testing import (
    TEST_CASES,
    TRUE_DIFFERENCES,
    ArticleNumbers,
    EquivalenceSummary,
    OutcomeRow,
)
from blog_reproducibility.statistics.equivalence_testing_figure import render_equivalence_figure

# Rendering needs only the shares; recomputing them would rerun 84,000 comparisons.
SUMMARY = EquivalenceSummary(
    rows=tuple(
        OutcomeRow(
            true_difference=delta,
            test_cases=n,
            t_test_significant=min(1.0, n / 800),
            equivalent=min(1.0, n / 1600),
            non_inferior=min(1.0, n / 1200),
            exact_t_test_significant=min(1.0, n / 800),
            exact_equivalent=min(1.0, n / 1600),
            exact_non_inferior=min(1.0, n / 1200),
        )
        for delta in TRUE_DIFFERENCES
        for n in TEST_CASES
    ),
    article=ArticleNumbers(2.85, 308.3, 389.6, 1233.2, 282.6, 0.797, 310),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_equivalence_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "equivalence_testing_power.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
