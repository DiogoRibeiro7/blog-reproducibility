"""Tests for shared plotting utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from blog_reproducibility.common.plotting import PALETTE, save_figure, use_house_style


def test_house_style_uses_expected_colour_cycle() -> None:
    """The shared style should expose the validated categorical palette."""
    use_house_style()
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    assert tuple(cycle) == PALETTE


def test_save_figure_writes_png_and_reports_nominal_dimensions(tmp_path: Path) -> None:
    """Saving a figure should create a PNG in the requested output directory."""
    use_house_style()
    figure, axis = plt.subplots(figsize=(4, 3))
    axis.plot([0, 1], [0, 1])

    artifact = save_figure(figure, slug="smoke_test", output_dir=tmp_path)

    assert artifact.path == tmp_path / "smoke_test.png"
    assert artifact.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact.width == 640
    assert artifact.height == 480


@pytest.mark.parametrize("slug", ["", ".", "..", "../escape", "nested/name"])
def test_save_figure_rejects_unsafe_slugs(tmp_path: Path, slug: str) -> None:
    """Figure slugs must not be able to escape the requested directory."""
    figure = plt.figure()

    with pytest.raises(ValueError):
        save_figure(figure, slug=slug, output_dir=tmp_path)

    plt.close(figure)
