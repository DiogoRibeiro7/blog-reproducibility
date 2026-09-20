# Adding an article

1. Identify the published article, its numerical claims, and the original scripts.
2. Add a typed numerical model under `src/blog_reproducibility/<domain>/`.
3. Add regression tests for quoted values and independent checks of the method.
4. Add a renderer using `common.plotting` and a thin script under
   `scripts/figures/<domain>/` with `--output-dir` and `--dry-run`.
5. Record required inputs and their provenance under `data/`.
6. Add a manifest entry, run the validation and reproduction commands, and review
   the resulting figures before copying publication assets to the website.

## Manifest contract

The schema lives in [`articles/manifest.schema.json`](../articles/manifest.schema.json).
All paths use forward slashes and are relative to their owning repository.

Published entries belong under `articles` and require a `_posts/` path and a
permalink. Unpublished entries belong under `drafts`, use a `_drafts/` path, and
omit the permalink. Identifiers and output paths must be unique across both
sections. Validation, reproduction, and the generated catalog cover both.

| Field | Meaning |
| --- | --- |
| Article key | Stable lowercase identifier with words separated by hyphens |
| `path` | Markdown path under `_posts/` in the website repository |
| `permalink` | Published article URL path |
| `modules` | Import paths corresponding to local `.py` model and renderer files |
| `figure_scripts` | Existing Python entry points under `scripts/figures/` |
| `generated_figures` | Unique PNG output paths under `build/figures/` |
| `tests` | Existing Python test files under `tests/` |
| `data_inputs` | Existing files under `data/`; omit or use `[]` when no files are required |
| `benchmark_harness` | Optional manual measurement script under `scripts/benchmarks/`; the reproduction runner does not execute it |
| `original_website_sources` | Historical paths in the website repository |
| `migration` | Model, tests, renderer, and website cleanup status |

Outputs for one article must share a directory, since each script receives the
same `--output-dir`. Multiple scripts may contribute outputs within that directory.
The runner preserves the part of the path below `build/figures/` when a custom
output root is selected. Output names must be unique across articles.

Use `pending` or `migrated` for the model, test, and renderer migration fields, and
`pending` or `complete` for website cleanup. An executable manifest entry must have
local files even if some publication cleanup remains pending. Track articles that
have no runnable implementation yet in the roadmap.

```console
poetry run python -m scripts.check_data --update
poetry run python -m scripts.reproduce --check
poetry run python -m scripts.catalog
poetry run pytest
poetry run python -m scripts.reproduce --article YOUR-ARTICLE-ID
```

The existing entries are complete working examples. When a scientific result
changes, document why and coordinate the associated article update. Do not update
a regression expectation solely to make a failing test pass.

When adding or changing `data_inputs`, review the checksum diff in `data/SHA256SUMS`
with the input files and provenance. See [the data policy](../data/README.md).

Regenerate `docs/articles.md` after modifying the manifest. CI and the commit hook
check it with `poetry run python -m scripts.catalog --check`. The manifest remains
the source of truth; edit its entries rather than the generated catalog.
