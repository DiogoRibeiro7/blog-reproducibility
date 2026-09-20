# Data

This directory is reserved for small, redistributable inputs and metadata required to reproduce article calculations.

Large, proprietary, sensitive, or externally maintained datasets should not be copied here merely for convenience. Instead, record their source, version or retrieval date, expected schema, and checksums where possible.

Derived outputs should normally be generated into `build/` and should not be committed unless they are small reference artefacts required for regression testing.

The current CUSUM and p-value examples use synthetic data, so their manifest
`data_inputs` lists are empty.

For each future external input, keep adjacent metadata recording:

- source URL and provider;
- dataset version or retrieval date;
- licence and redistribution terms;
- expected columns, types, units, and missing-value conventions;
- transformations applied before analysis;
- SHA-256 checksum for immutable files;
- instructions for obtaining restricted or large inputs locally.

List required local inputs under `data_inputs` in `articles/manifest.yml`. Manifest
validation checks that these files exist under `data/`, and reproduction reports
hash the files present in this directory. Document changing data sources and any
limits they place on reproducing historical results.

## Verify recorded inputs

[`SHA256SUMS`](SHA256SUMS) records the raw-byte SHA-256 digest of every distinct
`data_inputs` file registered in the article manifest, including drafts. Verify it
from the repository root with:

```console
poetry run python -m scripts.check_data
```

The reproduction command, CI manifest check, and commit hook also verify this
inventory. Changed bytes and missing or stale inventory entries fail validation
before rendering. Run reports additionally record the files used for each run.
Git's `.gitattributes` keeps text input line endings at LF across platforms.

For an intentional data correction, new input, or fresh benchmark measurement,
review the values and update the adjacent provenance documentation first. Then run:

```console
poetry run python -m scripts.check_data --update
```

Review and commit the data, manifest, provenance, and checksum changes together.
The update command does not change data files. The inventory detects differences
from a reviewed snapshot; it does not establish whether the source data is correct.
