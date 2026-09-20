# blog-reproducibility

Reproducible computations, numerical checks, simulations, benchmarks, and figure-generation code for articles published at [DiogoRibeiro7.github.io](https://diogoribeiro7.github.io/).

The website repository is responsible for publication. This repository is responsible for the computations behind that publication.

## Scope

This repository contains:

- numerical models used to support article claims;
- simulations and statistical demonstrations;
- benchmark code and derived benchmark data;
- figure-generation pipelines;
- reproducibility tests for quoted values and mathematical results;
- small notebooks when an interactive derivation is genuinely useful.

It does **not** contain the Jekyll site, article Markdown, site layouts, navigation, rendered production assets, or website-specific tests.

## Repository layout

```text
src/blog_reproducibility/
├── common/
├── engineering/
├── health/
├── physics/
├── statistics/
└── time_series/

articles/          # mapping between published articles and reproducibility code
scripts/figures/   # figure-generation entry points
tests/             # tests organised by scientific domain
data/              # metadata and small reproducibility inputs
notebooks/         # curated notebooks only
```

Generated figures should be written to `build/figures/`. The publication-ready copies belong in the website repository.

## Development

The project uses Python 3.11+, Poetry, Ruff, mypy, pytest, and pre-commit.

```bash
poetry install
poetry run pre-commit install
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src tests
poetry run pytest
```

## Reproducibility contract

Each migrated article should eventually have an entry in `articles/manifest.yml` identifying:

1. the article in the website repository;
2. the model or computation modules used by the article;
3. figure-generation entry points;
4. tests that validate published numerical claims;
5. any data inputs required to reproduce the result.

The goal is not to turn blog examples into a generic software library. The goal is to make the computational evidence behind the articles inspectable, testable, and reproducible.

## Related repository

- Website and articles: https://github.com/DiogoRibeiro7/DiogoRibeiro7.github.io

See [ROADMAP.md](ROADMAP.md) for the migration plan.
