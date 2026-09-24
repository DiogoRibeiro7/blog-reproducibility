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
- A machine-learning domain with eleven figures across seven articles: drift
  monitoring, multiple comparisons in alerting, retraining frequency, annotator
  agreement, the winner's curse in model selection, bandits against A/B tests,
  and proxy metrics under optimisation pressure.
- The five machine-learning figures that train scikit-learn models: censored
  labels, feature selection leaking into cross-validation, learning-curve
  extrapolation, permutation importance with correlated features, and test-set
  size in model ranking.

- Eleven experiment-design figures under statistics: unequal allocation, CUPED,
  trigger dilution, percentile metrics, switchback periods, sequential stopping
  boundaries, novelty effects, marketplace interference, ratio metrics, sample
  ratio mismatch, and cluster randomisation.

- Nine inference figures under statistics: design analysis (type S and M
  errors), equivalence testing, small-count intervals, empirical Bayes shrinkage,
  the block bootstrap, exact post-selection intervals, and the bias-variance and
  regularisation-path figures.

- Ten causal-inference and observational-data figures under statistics:
  non-compliance, regression discontinuity, staggered difference-in-differences,
  propensity scores, negative controls, Berkson's paradox, regression to the
  mean, the ecological fallacy, left truncation, and measurement error.

- 12 operational-data and measurement figures under statistics:
  acceptance sampling, extreme values, run length, recurrent events, Benford's law,
  digit heaping, capture-recapture, post-stratification weighting, quantile
  regression, intermittent demand, week-over-week noise, and factorial designs.

### Changed

- The intermittent-demand figure's title no longer says only one method is wrong
  about the rate (undebiased Croston is 3.5 percent high too); the censored-labels
  figure draws its true curve dashed and on top, so the hazard model no longer
  hides it.
- The censored-labels model builds customer-month rows only from fully observed
  months; counting the month the extract falls in as survived biased every
  monthly hazard down by about 1.2 points. The lasso figure's penalty axis now
  decreases from left to right, as its label says.
- Three figures now show what their articles claim: the splines figure uses a sharp
  peak on a trend, the sample-ratio-mismatch figure plots the signed bias rather
  than the absolute error, and the capture-recapture figure applies the
  (t - 1)/t factor in Chao's three-pass bound.
- The migration from the website is finished: the 67 articles migrated from its
  `generate_figures.py` link to their figure generators here, the file is gone
  from the website, and their manifest entries record the cleanup as complete.
- Pull requests run the test matrix on Linux only; Windows and macOS run on merges
  to `main`, version tags, and manual runs, so releases are still checked on all
  three platforms.
- Tests run in parallel with pytest-xdist, on every core in CI and on four
  workers in the pre-push hook, with one
  BLAS, OpenMP, and joblib thread per worker. Each article is reproduced by its
  own test, so the reproductions run in parallel too; the report-provenance test
  covers five articles that exercise every kind of manifest entry.
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
