"""Tests for the bandit regret figure rendering."""

from pathlib import Path

from blog_reproducibility.machine_learning.bandit_regret import BanditRegretSummary
from blog_reproducibility.machine_learning.bandit_regret_figure import (
    render_bandit_regret_figure,
)

# Rendering needs only the curves; recomputing them runs 100 bandits of 20,000 users.
USERS = tuple(range(0, 20_001, 500))
SUMMARY = BanditRegretSummary(
    users=USERS,
    thompson=tuple(25 * user / (user + 1_000) for user in USERS),
    quarter_then_exploit=tuple(0.015 * min(user, 5_000) for user in USERS),
    even_split=tuple(0.015 * user for user in USERS),
)


def test_renderer_writes_the_expected_png(tmp_path: Path) -> None:
    """The article figure should render as a non-empty PNG file."""
    artifact = render_bandit_regret_figure(output_dir=tmp_path, summary=SUMMARY)

    assert artifact.path == tmp_path / "bandit_cumulative_regret.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.path.stat().st_size > 10_000
    assert (artifact.width, artifact.height) == (1152, 672)
