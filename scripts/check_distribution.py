"""Check source archive contents and render from an isolated wheel installation."""

import os
import subprocess
import tarfile
import venv
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def _check_files(label: str, actual: dict[str, bytes], expected: dict[str, bytes]) -> None:
    """Name missing or changed files before attempting an installation."""
    missing = sorted(expected.keys() - actual.keys())
    changed = sorted(
        name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
    )
    if missing or changed:
        raise RuntimeError(
            f"{label} differs from the checkout: missing={missing}; changed={changed}"
        )


def check_contents(wheel: Path, source: Path, *, root: Path = ROOT) -> None:
    """Verify package files and reproduction inputs against their checkout bytes."""
    paths = {root / name for name in ("LICENSE", "README.md", "pyproject.toml", "poetry.lock")}
    for directory in ("src/blog_reproducibility", "scripts", "tests", "articles", "data"):
        paths.update(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    expected = {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(paths)}
    with ZipFile(wheel) as wheel_archive:
        wheel_files = {
            info.filename: wheel_archive.read(info)
            for info in wheel_archive.infolist()
            if not info.is_dir()
        }
    _check_files(
        "Wheel",
        wheel_files,
        {
            name.removeprefix("src/"): content
            for name, content in expected.items()
            if name.startswith("src/")
        },
    )

    source_files = {}
    with tarfile.open(source) as source_archive:
        for member in source_archive.getmembers():
            if member.isfile():
                stream = source_archive.extractfile(member)
                assert stream is not None
                with stream:
                    source_files[member.name.split("/", 1)[-1]] = stream.read()
    _check_files("Source archive", source_files, expected)


def main() -> None:
    """Fail if packaging loses resources or relies on the editable checkout."""
    wheels = list((ROOT / "dist").glob("*.whl"))
    archives = list((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise RuntimeError(
            "Expected one wheel and one source archive; use an empty dist/ directory."
        )
    check_contents(wheels[0], archives[0], root=ROOT)

    with TemporaryDirectory(prefix="blog-wheel-") as temporary:
        directory = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(directory / "venv")
        python = directory / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheels[0])],
            check=True,
        )
        smoke = """
from importlib.resources import files
from pathlib import Path
from blog_reproducibility import __version__
from blog_reproducibility.time_series.sequential_cusum_figure import render_sequential_cusum_figure
from blog_reproducibility.statistics.pvalue_evidence_figure import render_pvalue_evidence_figures
assert files('blog_reproducibility').joinpath('py.typed').is_file()
artifacts = [render_sequential_cusum_figure(output_dir=Path('figures'))]
artifacts.extend(render_pvalue_evidence_figures(output_dir=Path('figures')))
for artifact in artifacts:
    assert artifact.path.read_bytes().startswith(b'\\x89PNG\\r\\n\\x1a\\n')
print(f'Installed version {__version__}: all three figures rendered successfully.')
"""
        subprocess.run(
            [str(python), "-I", "-c", smoke],
            cwd=directory,
            env={**os.environ, "MPLBACKEND": "Agg"},
            check=True,
        )
    print("Wheel and source distribution checks passed.")


if __name__ == "__main__":
    main()
