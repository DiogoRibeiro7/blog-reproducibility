"""Reproduce the p-value evidence calculations and figures."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.statistics.pvalue_evidence import example_payload
from blog_reproducibility.statistics.pvalue_evidence_figure import (
    render_pvalue_evidence_figures,
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

    payload = example_payload()
    if not args.dry_run:
        artifacts = render_pvalue_evidence_figures(output_dir=args.output_dir)
        payload["figures"] = [
            {
                "path": str(artifact.path),
                "width": artifact.width,
                "height": artifact.height,
            }
            for artifact in artifacts
        ]

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
