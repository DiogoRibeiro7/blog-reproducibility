"""Reproduce the before-and-after testimonial calculations and figures."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.common.reporting import describe_figures, to_jsonable
from blog_reproducibility.statistics.testimonial_selection import example_payload
from blog_reproducibility.statistics.testimonial_selection_figure import (
    render_testimonial_figures,
)


def main() -> None:
    """Print the numerical payload and optionally render both figures."""
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
        help="Print calculations without writing figures.",
    )
    args = parser.parse_args()

    payload = to_jsonable(example_payload())
    if not args.dry_run:
        payload["figures"] = describe_figures(
            render_testimonial_figures(output_dir=args.output_dir)
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
