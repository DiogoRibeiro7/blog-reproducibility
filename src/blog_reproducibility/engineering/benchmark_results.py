"""Loading and describing recorded benchmark results.

The two data-engineering articles rest on benchmarks that take minutes, write
gigabytes of temporary files, and need DuckDB, pandas, NumPy and PyArrow. They
cannot run in a test suite, and they would not give the same numbers twice on
different hardware in any case.

So the benchmarks are separated from the claims. The harnesses live under
`scripts/benchmarks/` and are run by hand. What they produce is a JSON record,
kept under `data/engineering/`, that carries both the measurements and the
machine they were taken on. The models here read that record, and the tests
check that the statements the articles make still hold in it.

The absolute timings are not reproducible and are not treated as if they were.
The ratios between them are stable across machines, and those are what the
articles quote and what the tests bound.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from blog_reproducibility.common.validation import count

__all__ = [
    "DATA_DIR",
    "MachineDescription",
    "load_results",
    "machine_description",
]

DATA_DIR: Final[Path] = Path(__file__).resolve().parents[3] / "data" / "engineering"


@dataclass(frozen=True, slots=True)
class MachineDescription:
    """The machine a benchmark record was produced on.

    This travels with every result because the absolute numbers mean nothing
    without it. Anyone rerunning the harness should expect different seconds and
    similar ratios.
    """

    python: str
    platform: str
    cpus: int
    duckdb: str | None = None
    sqlite: str | None = None


def load_results(name: str, *, directory: Path = DATA_DIR) -> dict[str, Any]:
    """Load one recorded benchmark result by file stem."""
    if not name or Path(name).name != name:
        raise ValueError("name must be a bare file stem")

    path = directory / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No benchmark record at {path}. "
            "Run the matching harness under scripts/benchmarks/ to produce one."
        )

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return loaded


def machine_description(results: dict[str, Any]) -> MachineDescription:
    """Return the recorded machine, which every result file carries."""
    machine = results.get("machine")
    if not isinstance(machine, dict):
        raise ValueError("The benchmark record has no machine description")

    return MachineDescription(
        python=str(machine["python"]),
        platform=str(machine["platform"]),
        cpus=count(machine["cpus"], name="cpus", minimum=1),
        duckdb=str(machine["duckdb"]) if "duckdb" in machine else None,
        sqlite=str(machine["sqlite"]) if "sqlite" in machine else None,
    )
