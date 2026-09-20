# Figure entry points

Run these commands from the repository root after `poetry sync`:

```console
poetry run python scripts/figures/time_series/sequential_cusum.py
poetry run python scripts/figures/statistics/pvalue_evidence.py
```

Both scripts print a JSON calculation payload and write PNGs under
`build/figures/<domain>/`. Use `--dry-run` for calculations only or
`--output-dir PATH` to override the directory. Reported width and height are nominal
canvas dimensions; the house style's tight bounding box can change saved pixel dimensions.

For all registered figures and a provenance report, use:

```console
poetry run python -m scripts.reproduce
```

Keep numerical models importable and testable under `src/blog_reproducibility/`.
These scripts are thin adapters to models and renderers. See
[adding an article](../../docs/adding-an-article.md) for the script contract.
Publication-ready copies remain in the website repository.
