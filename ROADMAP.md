# Roadmap

This repository is being extracted from `DiogoRibeiro7/DiogoRibeiro7.github.io` so the website can remain a publication system while computational work lives in a dedicated reproducibility project.

## Phase 0 — Repository foundation

- [x] Define the publication/reproducibility boundary.
- [x] Use `main` as the only long-lived branch.
- [x] Add Poetry packaging.
- [x] Add Ruff, mypy, pytest, coverage, and pre-commit.
- [x] Add CI across supported Python versions.
- [x] Add an article-to-code manifest.

## Phase 1 — Inventory and mapping

- [x] Inventory every Python file currently under `assets/viz/` in the website repository.
- [x] Inventory article-specific tests currently under `tests/`.
- [x] Inventory `code/` notebooks and standalone examples.
- [x] Map every computational file to one or more published articles.
- [x] Identify generated artefacts that must remain in the website repository.
- [x] Identify dead, duplicate, and superseded scripts before migration.

The result is [docs/migration-inventory.md](docs/migration-inventory.md), which
classifies every file, maps each scientific source to its articles and figures,
and records how far each one has moved. The website inventory covers work for:

- aspartame dose calculations;
- confidence sets and coverage;
- p-value/evidence demonstrations;
- poll and selection effects;
- results rhetoric;
- scientific communication examples;
- hormone, inflammation, microbiome, parasite, and screening models;
- sequential change-point/CUSUM examples;
- quantum-observer examples;
- data-lake and database benchmarks;
- shared plotting/house-style utilities;
- one 3,826-line figure monolith holding 78 generators.

Two files were classified as obsolete rather than migrated: `code/Untitled.ipynb`
and `code/michelson_morley.py` both hold the same box plot of ten numbers marked
`# Hypothetical data`, with no article behind them.

## Phase 2 — Shared foundations

- [x] Migrate plotting style utilities into `blog_reproducibility.common`.
- [x] Separate numerical models from plotting functions for the first migrated article.
- [x] Define deterministic random-number handling for the first migrated simulation.
- [x] Define a common output directory contract for generated figures.
- [x] Add small typed helpers only where repeated code justifies them.

## Phase 3 — Domain migration

### Statistics

- [x] Confidence-set examples: inversion, hull cost, Fieller sets, profile projection.
- [x] Coverage demonstrations — the confidence-set coverage simulation.
- [x] P-value/evidence model, tests, and figure renderers.
- [x] Poll-selection models.
- [x] Results-rhetoric models — migrated under `health.results_rhetoric`.
- [x] Testimonial selection and regression to the mean.

### Health and evidence communication

- [x] Aspartame dose model.
- [x] Hormone-testing model.
- [x] Inflammation-marker model.
- [x] Leaky-gut model.
- [x] Microbiome-testing model.
- [x] Parasite-testing model.
- [x] Screening/survival model.
- [x] Testimonial model — migrated under `statistics.testimonial_selection`,
      because the mechanism is selection on a baseline rather than anything clinical.

### Time series

- [ ] Sequential CUSUM/change-point example.
  - [x] Numerical model and deterministic simulation.
  - [x] Reproducibility tests and article manifest.
  - [x] Figure renderer and shared plotting style.
  - [x] Website cleanup and article-link rewiring.

### Physics and scientific communication

- [x] Michelson–Morley example — the website file was scratch with hypothetical
      data, so this is a new analytic model of the predicted fringe shift rather
      than a migration.
- [x] Quantum-observer model.
- [x] Solar geometry and seasonal heat storage.
- [x] Remaining science-communication numerical examples: climate evidence,
      coin streaks, antibiotic selection, concentration and risk.

### Data engineering

- [ ] Data-lake benchmarks.
- [ ] Database benchmarks.
- [ ] Preserve benchmark provenance and environment metadata.

## Phase 4 — Figure generation

- [x] Establish article/domain figure entry points for migrated articles.
- [x] Write generated output to `build/figures/`.
- [x] Make migrated figure generation commands deterministic where applicable.
- [ ] Record the source module and article identifier in figure metadata where practical.
- [ ] Keep publication-ready rendered assets in the website repository.

## Phase 5 — Reproducibility tests

- [ ] Move article-specific numerical tests from the website repository.
  - [x] Sequential CUSUM numerical tests migrated.
  - [x] P-value evidence numerical tests migrated.
- [ ] Preserve tests that verify numbers quoted in published articles.
  - [x] Sequential CUSUM published values are explicitly regression-tested.
  - [x] P-value evidence article values are explicitly regression-tested.
- [x] Add invariant/property-style tests for the first migrated numerical model.
- [ ] Add regression tests for benchmark summaries.
- [ ] Keep website-layout, Markdown, theme, and link tests in the website repository.

## Phase 6 — Website rewiring

- [ ] Update the website `/code/` page to point here.
- [ ] Add reproducibility links to relevant articles.
- [ ] Remove migrated scientific Python from the website repository.
- [ ] Remove scientific Python test dependencies from the website CI.
- [ ] Keep rendered figures and article content in the website repository.
- [ ] Verify that no article URL changes during the migration.

## Phase 7 — Hardening

- [x] Add a manifest schema and validation tests for local references and outputs.
- [x] Lock dependencies and verify installed package resources.
- [x] Add a manifest-driven reproduction command with environment and checksum reports.
- [x] Document contributor, article migration, and maintenance workflows.
- [ ] Add provenance metadata for external datasets.
- [ ] Add checksums for immutable small inputs.
- [ ] Document reproducibility limitations for external APIs or changing datasets.
- [ ] Add release tags for coherent reproducibility snapshots.
