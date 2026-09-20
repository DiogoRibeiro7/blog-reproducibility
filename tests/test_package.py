"""Package-level smoke tests."""

from importlib.metadata import version
from importlib.resources import files

from blog_reproducibility import __version__


def test_package_exposes_version() -> None:
    """The public version should follow the installed distribution metadata."""
    assert __version__ == version("blog-reproducibility")


def test_package_contains_plotting_style_and_typing_marker() -> None:
    """Consumers need the plotting style and PEP 561 marker after installation."""
    package = files("blog_reproducibility")
    assert package.joinpath("py.typed").is_file()
    assert "figure.dpi" in package.joinpath("common/house.mplstyle").read_text(encoding="utf-8")
