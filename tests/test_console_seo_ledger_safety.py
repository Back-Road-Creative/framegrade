"""A corrections edit never destroys a ledger it could not understand.

One malformed byte used to turn every read into `{}`, so the next "add
correction" dumped a two-key file over hand-curated data, silently. These pin
the properties that make that impossible: corrupt bytes survive, the edit is
refused legibly, and a save that loses a race never overwrites the winner.
"""

import pytest
import yaml
from fastapi.testclient import TestClient

from gomoveshift.console import seo
from gomoveshift.console.app import create_app

MALFORMED = "corrections: [{wrong: 'Sligeo', correct: 'Sligo'}\ndenylist: [Nowhereville\n"


def _client() -> TestClient:
    return TestClient(create_app())


def _boom(*_args, **_kwargs):
    raise OSError("disk full")


def _ledger(tmp_path, monkeypatch, text: str):
    path = tmp_path / "place_corrections.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(path))
    return path


@pytest.mark.parametrize("text", [MALFORMED, "- just\n- a list\n"])
def test_an_unreadable_ledger_is_reported_not_swallowed(tmp_path, monkeypatch, text):
    path = _ledger(tmp_path, monkeypatch, text)

    with pytest.raises(seo.LedgerCorruptError):
        seo.read(path)


def test_editing_a_malformed_ledger_preserves_every_byte(tmp_path, monkeypatch):
    path = _ledger(tmp_path, monkeypatch, MALFORMED)
    before = path.read_bytes()

    with pytest.raises(seo.LedgerCorruptError):
        seo.add_correction("Sligeo", "Sligo", path)
    response = _client().post(
        "/seo/corrections", data={"wrong": "Sligeo", "correct": "Sligo"}, follow_redirects=False
    )

    assert response.status_code == 409
    assert path.read_bytes() == before


def test_get_shows_the_damage_instead_of_an_empty_state(tmp_path, monkeypatch):
    _ledger(tmp_path, monkeypatch, MALFORMED)

    response = _client().get("/seo")

    assert response.status_code == 200
    assert "could not be read" in response.text
    assert "No corrections yet" not in response.text


def test_a_stale_revision_loses_the_race_instead_of_overwriting(tmp_path, monkeypatch):
    path = _ledger(tmp_path, monkeypatch, yaml.safe_dump({"corrections": [], "denylist": []}))
    stale = seo.read(path).revision

    seo.add_denylist_entry("Nowhereville", path)

    with pytest.raises(seo.LedgerConflictError):
        seo.add_correction("Sligeo", "Sligo", path, expected_revision=stale)

    assert seo.read(path).denylist == ["Nowhereville"]
    assert seo.read(path).corrections == []


def test_the_form_carries_its_revision_and_a_stale_one_is_a_conflict(tmp_path, monkeypatch):
    path = _ledger(tmp_path, monkeypatch, yaml.safe_dump({"corrections": [], "denylist": []}))

    assert f'value="{seo.read(path).revision}"' in _client().get("/seo").text
    response = _client().post(
        "/seo/corrections",
        data={"denylist_name": "Nowhereville", "revision": "written-by-someone-else"},
        follow_redirects=False,
    )

    assert response.status_code == 409
    assert seo.read(path).denylist == []


def test_a_write_that_fails_mid_dump_leaves_the_ledger_intact(tmp_path, monkeypatch):
    path = _ledger(
        tmp_path,
        monkeypatch,
        yaml.safe_dump({"corrections": [{"wrong": "Sligeo", "correct": "Sligo"}], "denylist": []}),
    )
    before = path.read_bytes()

    monkeypatch.setattr(seo.yaml, "safe_dump", _boom)

    with pytest.raises(OSError):
        seo.add_denylist_entry("Nowhereville", path)

    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp*"))


def test_a_missing_ledger_reads_empty_with_no_revision(tmp_path):
    assert seo.read(tmp_path / "nope.yaml") == seo.Ledger([], [], "")
