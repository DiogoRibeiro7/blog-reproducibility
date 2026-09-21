# Maintenance

## Dependencies and CI

Review dependency update pull requests for numerical or rendering changes. Run the
tests and regenerate every figure after updating Python, Matplotlib, or numerical
dependencies. Commit the regenerated lock file. Monthly Dependabot configuration
covers Python dependencies and GitHub Actions; it does not automatically merge updates.

CI uses read-only repository permissions, full commit pins for external actions,
timeouts, and cancellation of superseded runs. See GitHub's
[secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
for the rationale for immutable action references. The dependency workflow follows
[Poetry's lock and sync commands](https://python-poetry.org/docs/cli/).

Poetry's version is pinned in `.github/actions/setup/action.yml`. Update the pin
and documented setup version together after validating the new version. Code
quality runs under the oldest supported interpreter; mypy uses the active Python
version so dependency stubs match their installed environment.

Coverage includes figure scripts launched by reproduction tests through
[coverage.py's subprocess support](https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html).
Keep the declared coverage version compatible with `patch = ["subprocess"]` when
updating test dependencies. Manual measurement harnesses are outside the typing
and coverage gates; the recorded-result models and renderers remain covered.

For a hosted repository, maintainers can configure branch protection to require
review and successful CI before merging. These settings live in GitHub and are
not enabled by the files in this repository.

## Prepare a reproducibility snapshot

1. Resolve scientific regressions and check that the article manifest is accurate.
2. Update `project.version` in `pyproject.toml` and `poetry sync`; the public
   `__version__` is read from installed package metadata.
3. Move relevant changelog entries into a dated release section.
4. Run the checks in `CONTRIBUTING.md` and reproduce all registered articles.
   Verify the saved output with
   `poetry run python -m scripts.verify_reproduction build/figures/reproduction.json`.
5. Build into an empty `dist/` directory with `poetry build`.
6. Run `poetry run python scripts/check_distribution.py` to compare archive files
   with the checkout and render from an isolated wheel installation. The wheel
   must preserve every package file; the source archive must also preserve scripts,
   tests, article metadata, data, and the dependency lock. Missing or changed bytes
   fail before installation. Rebuild after changing any of these inputs.
7. After review and a successful CI run, tag the approved commit and create a
   GitHub release with the distributions and reproduction artifacts.

Tags should identify coherent computational snapshots. Retain the commit, lock
file, interpreter version, and run report when citing a release. Release creation
and package publication are deliberate maintainer actions; CI produces reviewable
artifacts without publishing them to a package index.
