"""Protect recorded inputs from silent changes and incomplete checksum inventories."""

from pathlib import Path

import pytest

from scripts import check_data, reproduce
from scripts.manifest import Article, load_manifest

ABC_DIGEST = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.fixture
def registered_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Article]:
    """Register a tiny, independently known input without copying scientific models."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data/example.csv").write_bytes(b"abc")
    articles = {"example": Article("example", (), (), inputs=(Path("data/example.csv"),))}
    monkeypatch.setattr(check_data, "ROOT", tmp_path)
    monkeypatch.setattr(check_data, "load_manifest", lambda root: articles)
    return articles


def test_committed_inventory_matches_all_registered_inputs() -> None:
    check_data.verify_checksums(load_manifest().values())


def test_shared_inputs_are_recorded_once_in_standard_format(
    registered_input: dict[str, Article], tmp_path: Path
) -> None:
    article = registered_input["example"]
    (tmp_path / "data/Z.csv").write_bytes(b"abc")
    other = Article("other", (), (), inputs=(Path("data/Z.csv"),))
    assert check_data.render_checksums([article, other, article], root=tmp_path) == (
        f"{ABC_DIGEST}  data/Z.csv\n{ABC_DIGEST}  data/example.csv\n"
    )


@pytest.mark.parametrize("change", ["bytes", "missing", "extra", "duplicate", "malformed"])
def test_changed_data_or_inventory_fails_without_rewriting(
    registered_input: dict[str, Article], tmp_path: Path, change: str
) -> None:
    inventory = tmp_path / check_data.INVENTORY
    original = f"{ABC_DIGEST}  data/example.csv\n"
    inventory.write_text(original, encoding="utf-8", newline="\n")
    if change == "bytes":
        (tmp_path / "data/example.csv").write_bytes(b"changed")
    elif change == "missing":
        inventory.write_text("", encoding="utf-8")
    elif change == "extra":
        inventory.write_text(original + f"{ABC_DIGEST}  data/old.csv\n", encoding="utf-8")
    elif change == "duplicate":
        inventory.write_text(original * 2, encoding="utf-8")
    else:
        inventory.write_text("invalid digest  data/example.csv\n", encoding="utf-8")
    before = inventory.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        check_data.verify_checksums(registered_input.values(), root=tmp_path)
    assert inventory.read_bytes() == before


def test_inventory_must_follow_new_and_removed_manifest_inputs(
    registered_input: dict[str, Article], tmp_path: Path
) -> None:
    assert check_data.main(["--update"]) == 0
    (tmp_path / "data/new.csv").write_bytes(b"new input")
    registered_input["new"] = Article("new", (), (), inputs=(Path("data/new.csv"),))
    assert check_data.main([]) == 1
    assert check_data.main(["--update"]) == 0
    assert check_data.main([]) == 0
    del registered_input["example"]
    assert check_data.main([]) == 1
    assert check_data.main(["--update"]) == 0
    assert "data/example.csv" not in (tmp_path / check_data.INVENTORY).read_text(encoding="utf-8")


def test_only_explicit_update_creates_or_replaces_the_inventory(
    registered_input: dict[str, Article], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inventory = tmp_path / check_data.INVENTORY
    assert check_data.main([]) == 1
    assert "Data verification failed:" in capsys.readouterr().err
    assert not inventory.exists()
    assert check_data.main(["--update"]) == 0
    assert inventory.read_bytes() == f"{ABC_DIGEST}  data/example.csv\n".encode()
    assert check_data.main([]) == 0
    (tmp_path / "data/example.csv").unlink()
    before = inventory.read_bytes()
    assert check_data.main(["--update"]) == 1
    assert inventory.read_bytes() == before


def test_reproduction_stops_before_rendering_unverified_data(
    registered_input: dict[str, Article], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert check_data.main(["--update"]) == 0
    (tmp_path / "data/example.csv").write_bytes(b"changed")
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    monkeypatch.setattr(reproduce, "load_manifest", lambda: registered_input)

    def unexpected_renderer(articles: list[Article], output_dir: Path) -> Path:
        pytest.fail("Rendering must not start after a checksum mismatch")

    monkeypatch.setattr(reproduce, "reproduce", unexpected_renderer)
    output = tmp_path / "output"
    assert reproduce.main(["--output-dir", str(output)]) == 1
    assert not output.exists()


def test_empty_inputs_have_an_empty_inventory(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / check_data.INVENTORY).write_text("", encoding="utf-8")
    check_data.verify_checksums([], root=tmp_path)
