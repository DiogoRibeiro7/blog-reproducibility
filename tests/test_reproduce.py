"""Exercise reproduction commands, artifact provenance, and failure reporting."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.manifest import ROOT, Article, load_manifest
from scripts.reproduce import git_state, main, reproduce


@pytest.fixture
def reproduction_root(tmp_path: Path) -> Path:
    """Create the inputs required by a minimal reproduction checkout."""
    for name in ("pyproject.toml", "poetry.lock"):
        (tmp_path / name).write_text("# test fixture\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    return tmp_path


def test_reproduce_all_articles_and_record_provenance(tmp_path: Path) -> None:
    articles = list(load_manifest().values())
    report_path = reproduce(articles, tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["articles"] == [article.identifier for article in articles]
    assert report["packages"]["blog-reproducibility"]
    assert report["backend"] == "Agg"
    assert (
        report["inputs"]["poetry.lock"]
        == hashlib.sha256((ROOT / "poetry.lock").read_bytes()).hexdigest()
    )
    expected = {
        figure.relative_to("build/figures").as_posix()
        for article in articles
        for figure in article.figures
    }
    assert {figure["path"] for figure in report["figures"]} == expected
    for figure in report["figures"]:
        content = (tmp_path / figure["path"]).read_bytes()
        assert content.startswith(b"\x89PNG\r\n\x1a\n")
        assert figure["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("mode", ["--check", "--list"])
def test_read_only_commands(mode: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([mode, "--output-dir", str(tmp_path / "unused")]) == 0
    assert capsys.readouterr().out
    assert not (tmp_path / "unused").exists()


def test_unknown_article_has_a_helpful_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--article", "does-not-exist"])
    assert error.value.code == 2
    assert "use --list" in capsys.readouterr().err


def test_selected_article_command(tmp_path: Path) -> None:
    identifier = next(iter(load_manifest()))
    assert main(["--article", identifier, "--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "reproduction.json").read_text(encoding="utf-8"))
    assert report["articles"] == [identifier]


@pytest.mark.parametrize("source", ["raise RuntimeError('render failed')", "pass"])
def test_failed_render_cannot_reuse_stale_figures_or_success_report(
    reproduction_root: Path, source: str
) -> None:
    tmp_path = reproduction_root
    script = tmp_path / "renderer.py"
    script.write_text(source, encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (output / "reproduction.json").write_text("{}", encoding="utf-8")
    article = Article("broken", (Path("renderer.py"),), (Path("build/figures/stale.png"),))
    with pytest.raises(RuntimeError):
        reproduce([article], output, root=tmp_path)
    assert not (output / "reproduction.json").exists()


@pytest.fixture
def forbid_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an invalid output path fail the test if it starts any external command."""

    def unexpected_command(*args: object, **kwargs: object) -> object:
        pytest.fail("Invalid output paths must be rejected before starting commands")

    monkeypatch.setattr(subprocess, "run", unexpected_command)


@pytest.mark.parametrize("nested", [False, True])
def test_command_reports_invalid_output_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], forbid_commands: None, nested: bool
) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("occupied", encoding="utf-8")
    output = blocked / "new" if nested else blocked
    assert main(["--output-dir", str(output)]) == 1
    error = capsys.readouterr().err
    assert "Output directory" in error
    assert str(blocked) in error
    assert blocked.read_text(encoding="utf-8") == "occupied"


def test_blocked_figure_directory_preserves_previous_outputs(
    reproduction_root: Path, forbid_commands: None
) -> None:
    output = reproduction_root / "output"
    output.mkdir()
    (output / "domain").write_text("occupied", encoding="utf-8")
    report = output / "reproduction.json"
    report.write_text("previous report\n", encoding="utf-8")
    article = Article("example", (), (Path("build/figures/domain/figure.png"),))
    with pytest.raises(NotADirectoryError, match="domain"):
        reproduce([article], output, root=reproduction_root)
    assert report.read_text(encoding="utf-8") == "previous report\n"
    assert (output / "domain").read_text(encoding="utf-8") == "occupied"


@pytest.mark.parametrize("target", ["domain/figure.png", "reproduction.json"])
def test_output_files_cannot_replace_directories(
    reproduction_root: Path, forbid_commands: None, target: str
) -> None:
    output = reproduction_root / "output"
    blocked = output / target
    blocked.mkdir(parents=True)
    article = Article("example", (), (Path("build/figures/domain/figure.png"),))
    with pytest.raises(IsADirectoryError, match="Output file"):
        reproduce([article], output, root=reproduction_root)
    assert blocked.is_dir()


def test_output_plan_cannot_require_a_path_to_be_both_file_and_directory(
    reproduction_root: Path, forbid_commands: None
) -> None:
    articles = [
        Article("outer", (), (Path("build/figures/outer.png"),)),
        Article("inner", (), (Path("build/figures/outer.png/inner.png"),)),
    ]
    output = reproduction_root / "output"
    with pytest.raises(ValueError, match="both a file and a directory"):
        reproduce(articles, output, root=reproduction_root)
    assert not output.exists()


def test_source_archive_has_no_git_revision(tmp_path: Path) -> None:
    assert git_state(tmp_path) == {"revision": None, "dirty": None}


def test_nested_directory_does_not_inherit_parent_checkout_revision() -> None:
    assert git_state(ROOT / "docs") == {"revision": None, "dirty": None}


@pytest.mark.parametrize(
    "arguments",
    [
        ["--json"],
        ["--check", "--domain", "statistics"],
        ["--list", "--domain", "unknown"],
        ["--list", "--search", "no-such-article"],
    ],
)
def test_invalid_filters_fail_clearly(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_filtered_json_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list", "--domain", "statistics", "--search", "PVALUE", "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)
    assert len(entries) == 1
    assert entries[0]["identifier"] == "why-a-small-p-value-does-not-settle-a-claim"
    assert entries[0]["domains"] == ["statistics"]
    assert len(entries[0]["figures"]) == 2


def _renderer(root: Path, name: str, body: str) -> Path:
    """Write a fixture renderer that produces real PNGs before a custom action."""
    script = Path("scripts", name + ".py")
    (root / script).write_text(
        "import sys\nfrom pathlib import Path\nimport matplotlib.pyplot as plt\n"
        "output = Path(sys.argv[2])\noutput.mkdir(parents=True, exist_ok=True)\n" + body,
        encoding="utf-8",
    )
    return script


def test_articles_cannot_overwrite_each_others_staged_outputs(reproduction_root: Path) -> None:
    root = reproduction_root
    articles = []
    for name, color in (("first", "red"), ("second", "blue")):
        script = _renderer(
            root,
            name,
            f"for name in ('first', 'second'):\n"
            f"    figure = plt.figure(facecolor='{color}')\n"
            "    figure.savefig(output / (name + '.png'))\n    plt.close(figure)\n",
        )
        articles.append(Article(name, (script,), (Path(f"build/figures/{name}.png"),)))
    report_path = reproduce(articles, root / "output", root=root)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hashes = []
    for figure in report["figures"]:
        digest = hashlib.sha256((root / "output" / figure["path"]).read_bytes()).hexdigest()
        assert figure["sha256"] == digest
        hashes.append(digest)
    assert hashes[0] != hashes[1]


def test_another_articles_extra_output_cannot_hide_a_missing_figure(
    reproduction_root: Path,
) -> None:
    root = reproduction_root
    first = _renderer(
        root,
        "first",
        "figure = plt.figure()\nfigure.savefig(output / 'first.png')\n"
        "figure.savefig(output / 'second.png')\nplt.close(figure)\n",
    )
    second = _renderer(root, "second", "pass\n")
    articles = [
        Article("first", (first,), (Path("build/figures/first.png"),)),
        Article("second", (second,), (Path("build/figures/second.png"),)),
    ]
    with pytest.raises(RuntimeError, match="second: missing or invalid PNG"):
        reproduce(articles, root / "output", root=root)
    assert not (root / "output").exists()


def test_changing_inputs_invalidates_the_run(reproduction_root: Path) -> None:
    root = reproduction_root
    script = _renderer(
        root,
        "mutation",
        "figure = plt.figure()\nfigure.savefig(output / 'figure.png')\nplt.close(figure)\n"
        "Path('pyproject.toml').write_text('# changed during run\\n', encoding='utf-8')\n",
    )
    article = Article("mutation", (script,), (Path("build/figures/figure.png"),))
    with pytest.raises(RuntimeError, match="files changed during reproduction"):
        reproduce([article], root / "output", root=root)
    assert not (root / "output").exists()


@pytest.mark.parametrize(
    "script",
    [
        "scripts/figures/time_series/sequential_cusum.py",
        "scripts/figures/statistics/pvalue_evidence.py",
    ],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_figure_entry_points_emit_json(
    script: str,
    dry_run: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "figures"
    args = [script, "--output-dir", str(output)]
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_path(str(ROOT / script), run_name="__main__")
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    assert output.exists() is not dry_run
    if "alarm_index" in payload:
        assert payload["alarm_index"] == 64
    else:
        assert payload["z_2_p"] == pytest.approx(0.04550026389635844)
