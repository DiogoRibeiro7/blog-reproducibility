# Maintenance

## Dependencies and CI

Review dependency update pull requests for numerical or rendering changes. Run the
tests and regenerate every figure after updating Python, Matplotlib, or numerical
dependencies. Commit the regenerated lock file. Monthly Dependabot configuration
covers Python dependencies and GitHub Actions; it does not automatically merge updates.

CI checks use read-only repository permissions, full commit pins for external
actions, timeouts, and cancellation of superseded runs. Only the draft release job,
which runs after successful checks on version-tag pushes, has repository write
permission. See GitHub's
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
3. Move relevant changelog entries into a section headed
   `## VERSION - YYYY-MM-DD` (for example, `## 0.1.0 - 2026-09-21`). Keep
   `## Unreleased` above it for future work. Each version must appear exactly once,
   with a valid calendar date and nonempty notes beneath its heading.
4. Run the checks in `CONTRIBUTING.md` and reproduce all registered articles.
   Verify the saved output with
   `poetry run python -m scripts.verify_reproduction build/figures/reproduction.json`.
5. Build into an empty `dist/` directory with `poetry build`.
6. Run `poetry run python scripts/check_distribution.py` to compare archive files
   with the checkout and render from an isolated wheel installation. The wheel
   must preserve every package file; the source archive must also preserve scripts,
   tests, article metadata, data, and the dependency lock. Missing or changed bytes
   fail before installation. Rebuild after changing any of these inputs.
7. Run `poetry run python -m scripts.package_snapshot` to create
   `build/snapshots/reproducibility-snapshot.zip`. It includes the distributions,
   recorded figures, report, license, standalone verifier, and checksum inventory.
   The command rejects stale distributions or report inputs and preserves existing
   snapshots. Use `--output PATH.zip` to retain multiple snapshots.
8. Validate the proposed tag against `project.version` and its changelog section:
   `poetry run python -m scripts.check_release --tag v0.1.0`. Replace `v0.1.0`
   with `v` followed by the exact version in `pyproject.toml`.

## Create a reviewed GitHub release

After the release preparation changes are reviewed, merged, and passing CI,
a maintainer can create an annotated version tag on the approved commit and push
that tag. For example, with the approved `0.1.0` commit checked out:

```bash
git tag -a v0.1.0 -m "Reproducibility snapshot 0.1.0"
git push origin v0.1.0
```

Pushing a `v*` tag runs the full CI workflow, including release metadata validation.
The tag must exactly match the package version and have a dated changelog entry;
the current `Unreleased` section alone is insufficient. Once quality checks, every
test matrix entry, and distribution/reproduction checks pass, CI downloads the
verified snapshot and extracted release notes from that same run and creates a
**draft** GitHub release. Ordinary branch pushes, pull requests, and manual workflow
runs do not create releases.

The workflow uses GitHub CLI's
[`--verify-tag` and `--draft` options](https://cli.github.com/manual/gh_release_create)
to require an existing tag and leave publication to a maintainer. Review the notes,
download and verify the attached snapshot, then publish the draft in GitHub when
ready. CI does not overwrite an existing draft or published release for that tag.
If an upload fails after creating a draft, inspect the draft and complete its
missing asset upload using the successful run's artifact before publishing.

Tags should identify coherent computational snapshots. Retain the commit, lock
file, interpreter version, and run report when citing a release. Tag creation and
release publication are deliberate maintainer actions; CI produces reviewable
drafts without publishing packages to an index.
