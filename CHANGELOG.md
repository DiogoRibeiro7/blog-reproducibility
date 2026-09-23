# Changelog

Changes affecting reproducibility, development, and supported articles are recorded
here. The migration roadmap is maintained separately in [ROADMAP.md](ROADMAP.md).

## Unreleased

### Added

- An economics domain with the Lorenz curve and Gini coefficient, Solow steady
  state, and Monte Carlo GDP fan chart from the website's `generate_figures.py`,
  each with an explicit seed and checks against closed forms.
- A mathematics domain with the central limit theorem, Kaplan-Meier, Metropolis,
  queueing, and distance-concentration figures, and penalised change-point
  partitioning under time series. The queueing, change-point, and distance
  models reproduce every table those articles print.
- scikit-learn as a runtime dependency, for the distance-concentration classifiers.
- A data-science domain with the Savitzky-Golay, spline, kernel density, 2D
  histogram outlier, synthetic control, and null-rate monitoring figures. The
  synthetic-control model reproduces every estimate and placebo check its article
  prints.
- SciPy as an explicit runtime dependency, used directly for filtering, splines,
  and constrained optimisation.

### Changed

- Version-tag pushes publish the GitHub release once all CI checks pass, rather
  than leaving a draft for manual publication.

## 0.1.0 - 2026-09-23

### Added

- A portable checksum-inventory verifier bundled with snapshots and exercised
  after CI extracts the completed archive.
- Pinned GitHub Actions workflow linting in CI and commit hooks, including checks
  when local action definitions change.
- Version-tag validation against package metadata and dated changelog notes,
  with draft GitHub releases after all CI checks pass.
- A portable snapshot archive with verified distributions, figures, report,
  standalone verifier, and checksum inventory, produced locally and by CI.
- A read-only command to verify saved figure checksums, also run before CI artifact uploads.
- A committed SHA-256 inventory for registered data inputs, checked before reproduction.
- A data verification command with explicit checksum updates for reviewed data changes.
- Draft manifest validation and publication status in the catalog and JSON listings.
- A generated catalog covering the full article manifest, checked by CI and hooks.
- Reproduction filters by domain and search term, with JSON listings for automation.
- Per-article staging and input-change detection to keep provenance accurate.
- Validation and catalog links for optional manual benchmark harnesses.
- A committed dependency lock file and isolated Poetry environment configuration.
- Manifest schema and validation of article source, test, input, and output paths.
- One command to reproduce registered figures with environment metadata and checksums.
- Contributor, architecture, reproducibility, article migration, and maintenance guides.
- Issue and pull request templates, citation metadata, and dependency update configuration.
- Tests for manifest contracts, reproduction failures, provenance, and script output.
- CI coverage for Python 3.14, Windows, and macOS, plus isolated distribution checks.

### Changed

- Distribution checks rebuild a wheel from the source archive and verify both
  wheels in separate clean installations, including their versions and figures.
- Distribution checks compare every package file and source reproduction input
  with the checkout, rejecting missing or changed files before installation.
- Reproduction rejects conflicting output paths before rendering or removing prior reports.
- Tooling now accepts omitted `data_inputs` for articles without external files.
- Coverage follows figure scripts into subprocesses; manual measurement harnesses
  are excluded from strict typing and coverage, while remaining linted.
- Package metadata uses the standard `project` table; runtime version information
  comes from installed distribution metadata.
- Plotting resources and the typing marker are explicitly included in both package formats.
- CI installs locked dependencies, enforces a 90% coverage floor, and retains artifacts.
- Commit hooks run targeted quality checks; the complete test suite runs before pushing.
