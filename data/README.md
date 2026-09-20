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
