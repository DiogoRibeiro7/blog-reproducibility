"""Report the recorded database benchmark and render its figures.

The benchmark itself is under `scripts/benchmarks/`; this reads what it wrote.
"""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.common.reporting import describe_figures, to_jsonable
from blog_reproducibility.engineering.database_benchmarks import example_payload
from blog_reproducibility.engineering.database_benchmarks_figure import render_database_figures


def main() -> None:
    """Print the derived ratios and optionally render both figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT_DIR / "engineering",
        help="Directory for generated PNG output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the recorded results without writing figures.",
    )
    args = parser.parse_args()

    payload = to_jsonable(example_payload())
    if not args.dry_run:
        payload["figures"] = describe_figures(render_database_figures(output_dir=args.output_dir))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
