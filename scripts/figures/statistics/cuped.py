"""Reproduce the CUPED variance-reduction simulation, sample sizes and figure."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.common.reporting import describe_figures, to_jsonable
from blog_reproducibility.statistics.cuped import example_payload
from blog_reproducibility.statistics.cuped_figure import render_cuped_figure


def main() -> None:
    """Print the numerical payload and optionally render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT_DIR / "statistics",
        help="Directory for generated PNG output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print calculations without writing the figure.",
    )
    args = parser.parse_args()

    summary = example_payload()
    payload = to_jsonable(summary)
    if not args.dry_run:
        payload["figures"] = describe_figures(
            (render_cuped_figure(output_dir=args.output_dir, rows=summary.rows),)
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
