"""Package-level smoke tests."""

from blog_reproducibility import __version__


def test_package_exposes_version() -> None:
    """The public package version should match the initial project version."""
    assert __version__ == "0.1.0"
