"""Tests for the corrections-ledger place-name loader."""

from gomoveshift.pipeline.validation.place_corrections import (
    PlaceCorrections,
    apply_corrections,
    default_corrections_path,
    find_denylisted,
    load_place_corrections,
)


def test_missing_empty_malformed_ledger_yields_empty_corrections(tmp_path):
    assert load_place_corrections(tmp_path / "absent.yaml") == PlaceCorrections()
    for contents in ("", "not: [valid, yaml:"):
        ledger = tmp_path / "ledger.yaml"
        ledger.write_text(contents, encoding="utf-8")
        assert load_place_corrections(ledger) == PlaceCorrections()


def test_loads_mapping_and_denylist(tmp_path):
    ledger = tmp_path / "corrections.yaml"
    ledger.write_text(
        "corrections:\n  - wrong: Ridgeway\n    correct: Ridgeville\n"
        "denylist:\n  - Frogmore Depot\n",
        encoding="utf-8",
    )
    corrections = load_place_corrections(ledger)
    assert corrections.mapping == {"Ridgeway": "Ridgeville"}
    assert corrections.denylist == ("Frogmore Depot",)


def test_apply_corrections_substitutes_case_insensitively():
    corrections = PlaceCorrections(mapping={"Ridgeway": "Ridgeville"})
    assert (
        apply_corrections("Charleston to ridgeway via US-17, SC", corrections)
        == "Charleston to Ridgeville via US-17, SC"
    )
    assert apply_corrections("Charleston, SC", PlaceCorrections()) == "Charleston, SC"


def test_find_denylisted_matches_word_boundary_case_insensitive():
    corrections = PlaceCorrections(denylist=("Frogmore Depot",))
    assert find_denylisted("Beaufort to frogmore depot, SC", corrections) == ["Frogmore Depot"]
    assert find_denylisted("Beaufort, SC", corrections) == []


def test_default_corrections_path_is_under_data_home(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_CORRECTIONS_PATH", raising=False)
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "home"))
    assert default_corrections_path() == tmp_path / "home" / "place_corrections.yaml"


def test_default_corrections_path_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "custom.yaml"))
    assert default_corrections_path() == tmp_path / "custom.yaml"
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", "   ")
    assert default_corrections_path() == tmp_path / "home" / "place_corrections.yaml"


def test_load_place_corrections_resolves_default_at_call_time(tmp_path, monkeypatch):
    import gomoveshift.pipeline.validation.place_corrections as pc

    monkeypatch.delenv("GMS_CORRECTIONS_PATH", raising=False)
    first, second = tmp_path / "a", tmp_path / "b"
    for home, wrong in ((first, "Aaa"), (second, "Bbb")):
        home.mkdir()
        (home / "place_corrections.yaml").write_text(
            f"corrections:\n  - wrong: {wrong}\n    correct: Ccc\n", encoding="utf-8"
        )
    monkeypatch.setenv("GMS_DATA_HOME", str(first))
    assert pc.load_place_corrections().mapping == {"Aaa": "Ccc"}
    monkeypatch.setenv("GMS_DATA_HOME", str(second))
    assert pc.load_place_corrections().mapping == {"Bbb": "Ccc"}
    assert pc.default_corrections_path() == second / "place_corrections.yaml"
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(first / "place_corrections.yaml"))
    assert pc.load_place_corrections().mapping == {"Aaa": "Ccc"}
