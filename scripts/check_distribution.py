"""Check source archive contents and render from an isolated wheel installation."""

import os
import subprocess
import tarfile
import venv
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Fail if packaging loses resources or relies on the editable checkout."""
    wheels = list((ROOT / "dist").glob("*.whl"))
    archives = list((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise RuntimeError(
            "Expected one wheel and one source archive; use an empty dist/ directory."
        )
    with tarfile.open(archives[0]) as archive:
        members = {Path(name).as_posix().split("/", 1)[-1] for name in archive.getnames()}
    required = {
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "poetry.lock",
        "articles/manifest.yml",
        "articles/manifest.schema.json",
        "data/SHA256SUMS",
        "scripts/check_data.py",
        "scripts/reproduce.py",
        "scripts/manifest.py",
        "tests/test_package.py",
        "src/blog_reproducibility/common/house.mplstyle",
        "src/blog_reproducibility/py.typed",
    }
    if missing := required - members:
        raise RuntimeError(f"Source archive is missing: {sorted(missing)}")

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
