# Contributing

Contributions should make published computations easier to inspect and reproduce.
For larger migrations or new dependencies, open an issue describing the article,
the numerical claim, and the proposed scope before implementing the change.

## Set up

1. Fork and clone the repository, then create a branch from `main`.
2. Install Python 3.11–3.14 and Poetry 2.2.x or later in the 2.x series.
3. Run `poetry sync` and `poetry run pre-commit install` from the repository root.

Use `poetry env use 3.11` to select a particular installed interpreter, then run
`poetry sync` again. The local environment lives in `.venv/`; no global package
installation is needed for the scientific dependencies.

## Make a change

- Keep numerical models independent of plotting, filesystem access, and network calls.
- Use explicit random seeds and local random generators for simulations.
- Preserve published values with independent numerical checks and stated tolerances.
- Keep figure entry points thin, with `--output-dir` and `--dry-run` options.
- Register article modules, scripts, tests, inputs, and outputs in the manifest.
- Keep generated files under `build/`; publish copies through the website repository.
- Add a concise entry under `Unreleased` in `CHANGELOG.md` for user-visible changes.

Follow [the article guide](docs/adding-an-article.md) when adding a computation.
Use type annotations for Python APIs and docstrings that explain assumptions,
units, and return values when they matter.

## Verify

```console
poetry check --lock --strict
poetry run pre-commit run actionlint --all-files
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src scripts tests
poetry run pytest --cov --cov-report=term-missing
poetry run python -m scripts.reproduce --check
poetry run python -m scripts.catalog --check
poetry run python -m scripts.reproduce
```

Apply formatting with `poetry run ruff format .`. Commit hooks check formatting,
lint, types, metadata, the manifest, and GitHub Actions workflows. The pre-push hook
runs tests with coverage.
CI runs the same checks, exercises supported interpreters and operating systems,
and verifies the built package. Archive-content regression tests run in the unit
suite; the full packaging check also installs a built wheel in a clean environment.
That packaging script is exercised separately by CI and excluded from the coverage
calculation.

Workflow checks use the pinned actionlint hook in `.pre-commit-config.yaml`.
Changing workflow YAML, local action YAML, or the hook configuration checks every
workflow, including references to local action inputs. The first hook run downloads
and builds actionlint; pre-commit installs Go automatically if it is missing and
reuses the resulting environment on later runs. This setup needs network access.
See [pre-commit's Go support](https://pre-commit.com/#golang).

The check covers workflow syntax, expressions, matrix references, job dependencies,
and action inputs. Optional ShellCheck and Pyflakes integrations are disabled so
local and CI checks use the same rules regardless of installed system tools.

Manual benchmark harnesses under `scripts/benchmarks/` use optional dependencies
and retain their original implementation. They receive lint checks; strict typing
and coverage apply to the importable models, figure entry points, tests, and
repository tooling. Reproduction tests measure figure entry points in subprocesses.
After adding or changing an article, regenerate its documentation with
`poetry run python -m scripts.catalog`.

When changing dependencies, use `poetry add` or edit `pyproject.toml` and run
`poetry lock`, then `poetry sync`. Commit both `pyproject.toml` and `poetry.lock`.
Avoid unrelated dependency updates in a scientific-model change.

## Open a pull request

Explain the problem, the resulting behavior, and how you verified it. Link the
article and issue when applicable. If a quoted value changes, include the old and
new values and explain the reason. Changes to published claims require a matching
website update; record that follow-up in the manifest's migration status.

Keep discussion respectful and focused on evidence. Contributions are provided
under the repository's [MIT License](LICENSE).
