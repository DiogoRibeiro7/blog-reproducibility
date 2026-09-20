"""Generate the deterministic sequential CUSUM worked-example figure."""

import argparse
import json
from pathlib import Path

from blog_reproducibility.common.plotting import DEFAULT_FIGURE_OUTPUT_DIR
from blog_reproducibility.time_series.sequential_cusum import seeded_mean_shift_example
from blog_reproducibility.time_series.sequential_cusum_figure import (
    render_sequential_cusum_figure,
)


def main() -> None:
    """Print the reproducibility payload and optionally render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_OUTPUT_DIR / "time_series",
        help="Directory for generated PNG output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deterministic numerical payload without writing a figure.",
    )
    args = parser.parse_args()

    example = seeded_mean_shift_example()
    payload: dict[str, object] = {
        "first_changed_index": example.first_changed_index,
        "alarm_index": example.alarm_index,
        "target_shift": example.target_shift,
        "threshold": example.threshold,
        "scores": example.scores,
        "data": example.data,
    }

    if not args.dry_run:
        artifact = render_sequential_cusum_figure(output_dir=args.output_dir)
        payload["figure"] = {
            "path": str(artifact.path),
            "width": artifact.width,
            "height": artifact.height,
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
