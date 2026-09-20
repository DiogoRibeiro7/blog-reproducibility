"""Tests for the JSON conversion used by the figure scripts."""

import json
from dataclasses import dataclass
from pathlib import Path

from blog_reproducibility.common.plotting import FigureArtifact
from blog_reproducibility.common.reporting import describe_figures, to_jsonable


@dataclass(frozen=True, slots=True)
class Inner:
    """A nested result with a tuple field."""

    bounds: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Outer:
    """A result holding another result."""

    name: str
    inner: Inner
    rows: tuple[Inner, ...]


def test_nested_dataclasses_and_tuples_survive_json_dumps() -> None:
    """Tuples become lists so a dataclass of intervals serialises."""
    value = Outer(name="x", inner=Inner((0.0, 1.0)), rows=(Inner((2.0, 3.0)),))
    converted = to_jsonable(value)

    assert converted == {
        "name": "x",
        "inner": {"bounds": [0.0, 1.0]},
        "rows": [{"bounds": [2.0, 3.0]}],
    }
    assert json.loads(json.dumps(converted)) == converted


def test_dicts_and_paths_are_converted() -> None:
    """Dictionary keys become strings and paths become their text form."""
    converted = to_jsonable({1: Path("build") / "figures", "a": (1, 2)})

    assert set(converted) == {"1", "a"}
    assert converted["a"] == [1, 2]
    assert isinstance(converted["1"], str)


def test_dates_and_timestamps_become_iso_strings() -> None:
    """A release vintage depends on its time zone, so the offset is kept."""
    from datetime import date, datetime

    moment = datetime.fromisoformat("2025-04-30T12:30:00+00:00")
    assert to_jsonable(moment) == "2025-04-30T12:30:00+00:00"
    assert to_jsonable(date(2025, 4, 30)) == "2025-04-30"
    assert json.loads(json.dumps(to_jsonable({"at": moment}))) == {
        "at": "2025-04-30T12:30:00+00:00"
    }


def test_scalars_pass_through_unchanged() -> None:
    """Values JSON already understands are returned as they are."""
    assert to_jsonable(3) == 3
    assert to_jsonable("text") == "text"
    assert to_jsonable(None) is None
    assert to_jsonable(Inner) is Inner


def test_describe_figures_reports_path_and_dimensions() -> None:
    """A figure description carries the path and the nominal pixel size."""
    artifact = FigureArtifact(path=Path("build/figures/a.png"), width=640, height=480)

    assert describe_figures((artifact,)) == [
        {"path": str(Path("build/figures/a.png")), "width": 640, "height": 480}
    ]
