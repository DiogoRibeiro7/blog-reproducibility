"""Validate article metadata and references before executing repository scripts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, cast
from urllib.parse import quote

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "articles" / "manifest.schema.json"


class ArticleEntry(TypedDict):
    """Fields consumed after JSON Schema validation."""

    path: str
    permalink: NotRequired[str]
    modules: list[str]
    figure_scripts: list[str]
    generated_figures: list[str]
    tests: list[str]
    data_inputs: NotRequired[list[str]]
    benchmark_harness: NotRequired[str]


@dataclass(frozen=True)
class Article:
    """Validated paths, relative to the repository root."""

    identifier: str
    scripts: tuple[Path, ...]
    figures: tuple[Path, ...]
    modules: tuple[str, ...] = ()
    tests: tuple[Path, ...] = ()
    inputs: tuple[Path, ...] = ()
    source_url: str = ""
    permalink: str = ""
    benchmark_harness: Path | None = None
    publication_status: Literal["published", "draft"] = "published"

    @property
    def domains(self) -> tuple[str, ...]:
        """Return the scientific domains referenced by this article."""
        return tuple(sorted({module.split(".")[1] for module in self.modules}))


def _check_file(root: Path, path: Path, directory: str, suffix: str | None = None) -> None:
    """Require an existing file inside its designated repository directory."""
    resolved = (root / path).resolve()
    if not resolved.is_relative_to((root / directory).resolve()):
        raise ValueError(f"{path.as_posix()} must be inside {directory}/")
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"{path.as_posix()} must have the {suffix} extension")
    if not resolved.is_file():
        raise ValueError(f"Missing file: {path.as_posix()}")


def load_manifest(root: Path = ROOT) -> dict[str, Article]:
    """Reject malformed metadata, missing source files, and colliding outputs."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    document = yaml.safe_load((root / "articles" / "manifest.yml").read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        details = "; ".join(
            f"{'/'.join(map(str, error.absolute_path)) or 'manifest'}: {error.message}"
            for error in errors
        )
        raise ValueError(details)

    published = cast(dict[str, ArticleEntry], document["articles"])
    drafts = cast(dict[str, ArticleEntry], document.get("drafts", {}))
    if duplicates := published.keys() & drafts.keys():
        raise ValueError(
            f"Identifiers must be unique across articles and drafts: {sorted(duplicates)}"
        )
    entries = {**published, **drafts}
    articles: dict[str, Article] = {}
    outputs: set[Path] = set()
    for identifier, entry in entries.items():
        harness = Path(entry["benchmark_harness"]) if "benchmark_harness" in entry else None
        if harness is not None:
            _check_file(root, harness, "scripts/benchmarks", ".py")
        for module in entry["modules"]:
            _check_file(root, Path("src", *module.split(".")).with_suffix(".py"), "src", ".py")
        for values, directory, suffix in (
            (entry["figure_scripts"], "scripts/figures", ".py"),
            (entry["tests"], "tests", ".py"),
            (entry.get("data_inputs", []), "data", None),
        ):
            for value in values:
                _check_file(root, Path(value), directory, suffix)

        figures = tuple(Path(value) for value in entry["generated_figures"])
        for figure in figures:
            if not (root / figure).resolve().is_relative_to((root / "build/figures").resolve()):
                raise ValueError(f"{identifier}: figures must be inside build/figures/")
            if figure.suffix != ".png":
                raise ValueError(f"{identifier}: expected a .png figure: {figure}")
            if figure in outputs:
                raise ValueError(f"Duplicate generated figure: {figure.as_posix()}")
            outputs.add(figure)
        if len({figure.parent for figure in figures}) != 1:
            raise ValueError(f"{identifier}: figures must share one output directory")

        articles[identifier] = Article(
            identifier=identifier,
            scripts=tuple(Path(value) for value in entry["figure_scripts"]),
            figures=figures,
            modules=tuple(entry["modules"]),
            tests=tuple(Path(value) for value in entry["tests"]),
            inputs=tuple(Path(value) for value in entry.get("data_inputs", [])),
            source_url=(
                f"https://github.com/{document['website']['repository']}/blob/"
                f"{quote(document['website']['branch'], safe='')}/{quote(entry['path'], safe='/')}"
            ),
            permalink=entry.get("permalink", ""),
            benchmark_harness=harness,
            publication_status="draft" if identifier in drafts else "published",
        )
    return articles
