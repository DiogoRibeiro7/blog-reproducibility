# blog-reproducibility

[![CI](https://github.com/DiogoRibeiro7/blog-reproducibility/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/blog-reproducibility/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%E2%80%93%203.14-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**The computational evidence behind [Diogo Ribeiro's articles](https://diogoribeiro7.github.io/).**

Inspect the numerical models, verify published values, and regenerate the figures.
Each supported article connects its source code, tests, inputs, and outputs through a
[validated manifest](articles/manifest.yml).

## Quick start

Use Python 3.11–3.14 and [Poetry](https://python-poetry.org/docs/#installation)
2.2 or newer within the 2.x series. CI uses Poetry 2.2.1. If you use pipx:

```console
pipx install poetry==2.2.1
git clone https://github.com/DiogoRibeiro7/blog-reproducibility.git
cd blog-reproducibility
poetry sync
poetry run pytest
poetry run python -m scripts.reproduce
```

These commands work in PowerShell, Bash, and Zsh. Run repository commands from the
project root. Poetry creates an isolated `.venv/` and installs exact dependency
versions from `poetry.lock`.

The reproduction command generates every registered PNG figure and a
`build/figures/reproduction.json` report containing the Git revision, working-tree
status, Python and package versions, input hashes, and output checksums. The current
computations use local inputs and require no downloads after installation.

## Reproduce an article

Browse the complete [article catalog](docs/articles.md) for models, scripts, tests,
data inputs, and reproduction commands across all migrated domains. These two
examples are a quick starting point:

| Article | Identifier | Computational evidence |
| --- | --- | --- |
| [Sequential change-point detection](https://diogoribeiro7.github.io/data-science/advanced_sequential_changepoint/) | `advanced-sequential-changepoint` | Seeded mean shift, upper CUSUM, and alarm at index 64 |
| [Why a small p-value does not settle a claim](https://diogoribeiro7.github.io/science-communication/why_a_small_p_value_does_not_settle_a_claim/) | `why-a-small-p-value-does-not-settle-a-claim` | Selection probabilities, study comparisons, and two figures |

```console
poetry run python -m scripts.reproduce --list
poetry run python -m scripts.reproduce --list --domain statistics
poetry run python -m scripts.reproduce --list --search pvalue --json
poetry run python -m scripts.reproduce --check
poetry run python -m scripts.check_data
poetry run python -m scripts.reproduce --article advanced-sequential-changepoint
poetry run python -m scripts.reproduce --output-dir build/review-figures
```

Use `--domain physics` without `--list` to reproduce an entire domain. Engineering
figures read the recorded benchmark results in `data/engineering/`; expensive
benchmark harnesses are run separately and require the optional `benchmarks`
dependency group. See [benchmark provenance and commands](data/engineering/README.md).

To inspect calculations without creating figures:

```console
poetry run python scripts/figures/time_series/sequential_cusum.py --dry-run
poetry run python scripts/figures/statistics/pvalue_evidence.py --dry-run
```

Numerical results are regression-tested with explicit tolerances. Image bytes can
vary with the operating system, fonts, and rendering libraries. See the
[reproducibility guide](docs/reproducibility.md) for the guarantees and limitations.

Registered data inputs are verified against the committed `data/SHA256SUMS` before
reproduction. See [the data policy](data/README.md) for reviewing intentional updates.

## Development

```console
poetry run pre-commit install
poetry run pre-commit run actionlint --all-files
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src scripts tests
poetry run pytest --cov --cov-report=term-missing
poetry check --lock --strict
poetry run python -m scripts.catalog --check
```

CI checks Python 3.11–3.14 on Linux and Python 3.13 on Windows and macOS. It also
checks workflow definitions and the manifest, enforces 90% combined statement/branch
coverage, builds both distribution formats, and renders figures from an isolated
wheel installation.
Download the distributions, figures, and run report from the CI run's artifacts.
They also include a [snapshot ZIP](docs/reproducibility.md#save-a-reproducibility-snapshot)
with the source archive, wheel, figures, checksums, and standalone verification command.

Verify saved figures against their report without rerendering:

```console
poetry run python -m scripts.verify_reproduction build/figures/reproduction.json
```

For downloaded artifacts, pass the extracted report's path and retain its figure
directories. See [artifact verification](docs/reproducibility.md#verify-saved-figures).

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and
[adding an article](docs/adding-an-article.md) for the migration checklist.

## Repository layout

```text
articles/                  Article manifest and JSON Schema
data/                      Small inputs and their provenance
docs/                      Reproduction, architecture, and maintenance guides
notebooks/                 Curated interactive derivations
scripts/                   Reproduction and packaging commands
  figures/                 Thin article-specific entry points
src/blog_reproducibility/   Typed numerical models and figure renderers
  common/                  Shared plotting utilities and packaged house style
  engineering/             Recorded benchmark analysis and figures
  health/                  Health evidence and risk communication models
  physics/                 Quantum measurement, seasons, and interferometry
  statistics/              P-value and evidence models
  time_series/             Sequential CUSUM model
tests/                     Numerical, rendering, manifest, and command tests
build/                     Generated artifacts; ignored by Git
```

The [website repository](https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io)
owns article text, Jekyll templates, and published assets. This repository owns
the computations and their verification. Additional scientific domains are being
migrated; see [ROADMAP.md](ROADMAP.md).

## Documentation and support

- [Documentation index](docs/README.md)
- [Article catalog](docs/articles.md)
- [Changes](CHANGELOG.md) and [citation metadata](CITATION.cff)
- [Report a bug or propose an article](https://github.com/DiogoRibeiro7/blog-reproducibility/issues/new/choose)
- [Security reporting](SECURITY.md)

Released under the [MIT License](LICENSE). When using a result, cite the associated
article and the repository commit or release used for the computation.
