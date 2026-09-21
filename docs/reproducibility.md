# Reproducing results

## Run from a known environment

Check out the commit or release you want to reproduce. Install Poetry 2.2.1 for the
same dependency manager used in CI, select an installed Python 3.11–3.14 interpreter,
and run `poetry sync`. The committed lock file selects exact versions and hashes
for compatible dependencies; different Python versions may select different
versions through environment markers.

Run commands from the repository root:

```console
poetry run python -m scripts.reproduce --list
poetry run python -m scripts.reproduce --check
poetry run pytest
poetry run python -m scripts.reproduce
```

Use `--article IDENTIFIER` to run one article or `--output-dir PATH` to change the
output root. Use `--domain statistics` to reproduce a subject, or
`--list --search pvalue --json` to search and export metadata without rendering.
The [article catalog](articles.md) lists the current collection. Relative custom
paths are resolved from the current directory. Each run validates the manifest,
its inputs against `data/SHA256SUMS`, and the output paths before rendering.
Files blocking output directories, directories occupying figure or report paths,
and conflicting file/directory requirements cause an immediate error, leaving
previous figures and reports untouched. The runner renders into a fresh temporary
directory using the non-interactive `Agg` backend, checks the expected PNG files,
and copies successful outputs to the requested directory. Each article has its own
staging directory, including when two articles share an entry point. A failed
renderer returns a nonzero exit code and cannot satisfy output checks with files
left by an earlier run.

## Outputs

Outputs mirror each article's paths in the manifest. For example:

```text
build/figures/
├── reproduction.json
├── statistics/
│   ├── science_pvalue_selected_studies.png
│   └── science_pvalue_study_comparison.png
└── time_series/
    └── sequential_cusum_worked_example.png
```

`reproduction.json` records the selected articles, UTC time, Git revision and dirty
status, Python and platform versions, installed packages, SHA-256 hashes of source
and input files, and SHA-256 hashes of the generated figures. Figure paths are
relative to the directory containing the report. Source hashes describe the local
files, including uncommitted edits. A report is evidence of a run, not a source
archive: retain the corresponding checkout to reproduce those edits.

The runner hashes sources and data before rendering and verifies them again before
publishing outputs. If those files change during the run, it fails without writing
a success report. Source archives report no Git revision, even if extracted inside
another repository.

Registered unpublished drafts are included in default reproduction runs. They
have the same source and output checks as published articles. The catalog and
JSON listings identify their publication status; they have no published permalink.

Each invocation replaces the report for that output directory. Its figure list
covers only that invocation; unrelated figures from earlier runs may remain in
the directory. Use a separate `--output-dir` when comparing runs. A failed rerun
removes the old report once rendering starts so it cannot be mistaken for a new
successful run. Validation failures leave prior outputs untouched.

The source distribution includes repository scripts and metadata. The wheel
contains the importable scientific package and plotting resources; clone the
repository or unpack a source distribution to use `scripts.reproduce`.

The engineering renderers currently locate their recorded inputs relative to the
source checkout. Use the repository or source distribution for these examples;
the wheel does not bundle those records. Reproducing their figures reads the
existing JSON records and does not rerun the measurement harnesses. See
[benchmark provenance](../data/engineering/README.md) before collecting new timings.

## Verify saved figures

Check an existing run or downloaded artifact without rendering it again:

```console
poetry run python -m scripts.verify_reproduction build/figures/reproduction.json
```

Keep the report and its figure directories together when copying or extracting
outputs. Paths are resolved relative to the report, so the command works from
another directory and does not need the original checkout or its data. The verifier
uses only Python's standard library and can also run directly:

```console
python scripts/verify_reproduction.py path/to/download/reproduction.json
```

Verification reads each recorded figure and compares its SHA-256 checksum with the
report. It returns a nonzero exit code for missing or changed figures, malformed
reports, unsupported report versions, duplicate figure paths, and paths that leave
the report directory. It changes no files. Unlisted files are ignored, since a
directory may hold figures from other runs. This verifies recorded figure bytes;
it does not validate the report's source or environment metadata, or rerun the
scientific calculations. CI verifies its saved figures before uploading artifacts.

## Save a reproducibility snapshot

Bundle a run with the code and data needed to reproduce it:

```console
poetry build
poetry run python -m scripts.reproduce
poetry run python -m scripts.package_snapshot
```

The command creates `build/snapshots/reproducibility-snapshot.zip`. The `dist/`
directory must contain exactly one wheel and one source archive. Both are checked
against the checkout; the wheel's project name and version must also match.
The report's input hashes must match the current source and data files. Rebuild
the distributions and rerun reproduction after changing those inputs.

The archive contains the distributions, report, recorded figures, license,
standalone figure verifier, a README, and `SHA256SUMS` for every other archived
file. Figures left over from other runs are excluded. After extraction, run
`python verify_reproduction.py figures/reproduction.json` to check the figures;
the bundled README explains how to check the complete checksum inventory and
reproduce from the source archive.

Use `--report PATH` to bundle another run from the same sources and
`--output PATH.zip` to choose a destination. Existing snapshots are never
overwritten. CI includes this archive in its `distributions-and-figures` artifact;
retain a copy or attach it to a reviewed release for long-term use.

## Numerical guarantees

- The CUSUM example uses `random.Random(42)`, 100 observations, and a mean shift at
  zero-based index 60. The published first alarm is at index 64.
- The p-value examples evaluate analytic Gaussian expressions. They use synthetic
  scenarios and do not fetch external datasets.
- Regression tests check published values with numerical tolerances. Independent
  calculations and invariants check more than agreement with stored output.
- Generated PNGs are checked for successful rendering. Fonts, operating systems,
  Matplotlib, and its dependencies can change layout or image bytes. Checksums
  identify an artifact; they are not a promise of identical pixels across systems.
- Random-number algorithms and floating-point behavior can differ across future
  Python versions. Record the interpreter as well as the seed and lock file.

For external inputs, follow [the data policy](../data/README.md). A changing API or
unversioned source cannot provide an exact historical snapshot; record those
limitations alongside the article's data metadata.

## Troubleshooting

If unrelated pytest plugins or missing imports appear, use `poetry run` and verify
the interpreter with `poetry env info`. Run `poetry sync` after changing Python or
checking out a different dependency lock file. The test suite and reproduction
runner select `Agg`, so a graphical desktop is not required.

To inspect calculations independently of rendering, run either article script in
[`scripts/figures`](../scripts/figures/README.md) with `--dry-run`.
