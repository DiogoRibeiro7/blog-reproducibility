"""Turn model results into something ``json.dumps`` accepts.

The models return frozen dataclasses so the schema is visible and typed. The
figure scripts print those results, so one conversion lives here rather than in
every script.
"""

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from blog_reproducibility.common.plotting import FigureArtifact


def to_jsonable(value: object) -> Any:
    """Convert dataclasses, tuples, and paths into JSON-serialisable values.

    Tuples become lists, so a dataclass field holding an interval prints as a
    two-element array rather than failing.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value


def describe_figures(artifacts: tuple[FigureArtifact, ...]) -> list[dict[str, object]]:
    """Describe generated figures for a script's JSON output."""
    return [
        {"path": str(artifact.path), "width": artifact.width, "height": artifact.height}
        for artifact in artifacts
    ]
