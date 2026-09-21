"""Check both distributions, rebuild from source, and render from isolated installs."""

import os
import subprocess
import tarfile
import tomllib
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


def find_distributions(directory: Path) -> tuple[Path, Path]:
    """Select an unambiguous wheel and source archive from a build directory."""
    wheels = list(directory.glob("*.whl"))
    archives = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise RuntimeError(
            "Expected one wheel and one source archive; use an empty dist/ directory."
        )
    return wheels[0], archives[0]


def _smoke_install(wheel: Path, directory: Path, version: str) -> Path:
    """Install one wheel in a fresh environment and render into a fresh directory."""
    venv.EnvBuilder(with_pip=True).create(directory / "venv")
    python = directory / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
        cwd=directory,
        check=True,
    )
    smoke = """
import sys
from importlib.resources import files
from pathlib import Path
from blog_reproducibility import __version__
from blog_reproducibility.time_series.sequential_cusum_figure import render_sequential_cusum_figure
from blog_reproducibility.statistics.pvalue_evidence_figure import render_pvalue_evidence_figures
assert __version__ == sys.argv[1], (__version__, sys.argv[1])
assert files('blog_reproducibility').joinpath('py.typed').is_file()
artifacts = [render_sequential_cusum_figure(output_dir=Path('figures'))]
artifacts.extend(render_pvalue_evidence_figures(output_dir=Path('figures')))
for artifact in artifacts:
    assert artifact.path.read_bytes().startswith(b'\\x89PNG\\r\\n\\x1a\\n')
print(f'Installed version {__version__}: all three figures rendered successfully.')
"""
    subprocess.run(
        [str(python), "-I", "-c", smoke, version],
        cwd=directory,
        env={**os.environ, "MPLBACKEND": "Agg"},
        check=True,
    )
    return python


def main() -> None:
    """Fail if either distribution cannot supply a working, complete installation."""
    wheel, source = find_distributions(ROOT / "dist")
    check_contents(wheel, source, root=ROOT)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]

    with TemporaryDirectory(prefix="blog-distributions-") as temporary:
        directory = Path(temporary)
        print("Checking the directly built wheel...", flush=True)
        python = _smoke_install(wheel, directory / "wheel-install", version)

        # Pass the source archive itself to pip from outside the checkout. Build
        # isolation supplies its declared backend; disable cache reuse so this
        # run must exercise the archive's build configuration and inputs.
        print("Rebuilding a wheel from the source archive...", flush=True)
        rebuilt_directory = directory / "rebuilt"
        subprocess.run(
            [
                str(python),
                "-I",
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-cache-dir",
                "--wheel-dir",
                str(rebuilt_directory),
                str(source),
            ],
            cwd=directory,
            check=True,
        )
        rebuilt_wheels = list(rebuilt_directory.glob("*.whl"))
        if len(rebuilt_wheels) != 1:
            raise RuntimeError("Expected exactly one wheel rebuilt from the source archive")
        rebuilt_wheel = rebuilt_wheels[0]
        check_contents(rebuilt_wheel, source, root=ROOT)
        print("Checking the source-rebuilt wheel in a separate environment...", flush=True)
        _smoke_install(rebuilt_wheel, directory / "source-install", version)
    print("Wheel, source rebuild, and both isolated installation checks passed.")


if __name__ == "__main__":
    main()
