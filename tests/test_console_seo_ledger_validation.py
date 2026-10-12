"""A corrections edit that cannot take effect is refused, not recorded.

The ledger is not a notepad -- `load_place_corrections` feeds
`title_builder` and `content_gates`, so a row here changes burned OST and
YouTube titles. But the loader is deliberately forgiving: it silently drops
a blank rule, silently keeps only the LAST of two rules for the same wrong
name, and happily substitutes a correction whose result the metadata gate
then hard-fails on. Every one of those is invisible on the page, which
shows the row the operator wrote and no hint that it does nothing, does
something else, or guarantees a failed run.

The writer cannot borrow the loader's tolerance -- the same reasoning that
made it refuse corrupt bytes. It refuses at the gate instead, upstream
of the irreversible write.
"""

import pytest
import yaml

from gomoveshift.console import seo
from gomoveshift.pipeline.validation.place_corrections import load_place_corrections


def _ledger(tmp_path, monkeypatch, data: dict):
    path = tmp_path / "place_corrections.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(path))
    return path


def _corrections(path):
    return seo.read(path).corrections


# --- rules that would never apply -------------------------------------------


@pytest.mark.parametrize("wrong,correct", [("", "Sligo"), ("Sligeo", ""), ("  ", "Sligo")])
def test_a_blank_side_is_refused(tmp_path, monkeypatch, wrong, correct):
    """`load_place_corrections` drops any rule with a blank side, so the page
    would show a correction that has never once been applied."""
    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": []})

    with pytest.raises(seo.LedgerInvalidError):
        seo.add_correction(wrong, correct, path)

    assert _corrections(path) == []


def test_a_correction_to_itself_is_refused(tmp_path, monkeypatch):
    """Substituting a name with itself is a rule that can never change any
    text. Recorded, it reads on the page exactly like one that does."""
    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": []})

    with pytest.raises(seo.LedgerInvalidError):
        seo.add_correction("Sligo", "sligo ", path)

    assert _corrections(path) == []


def test_a_blank_denylist_name_is_refused(tmp_path, monkeypatch):
    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": []})

    with pytest.raises(seo.LedgerInvalidError):
        seo.add_denylist_entry("   ", path)

    assert seo.read(path).denylist == []


# --- rules that would silently override, or duplicate, an existing one ------


def test_a_second_rule_for_the_same_name_is_refused(tmp_path, monkeypatch):
    """`mapping[wrong] = correct` means the LAST rule wins and the earlier one
    vanishes with no signal anywhere. The refusal names what is already there
    so the operator can decide which they meant."""
    path = _ledger(
        tmp_path,
        monkeypatch,
        {"corrections": [{"wrong": "Sligeo", "correct": "Sligo"}], "denylist": []},
    )

    with pytest.raises(seo.LedgerInvalidError) as exc:
        seo.add_correction("sligeo", "County Sligo", path)

    assert "Sligo" in str(exc.value)
    assert _corrections(path) == [{"wrong": "Sligeo", "correct": "Sligo"}]


def test_an_identical_rule_is_refused_rather_than_duplicated(tmp_path, monkeypatch):
    path = _ledger(
        tmp_path,
        monkeypatch,
        {"corrections": [{"wrong": "Sligeo", "correct": "Sligo"}], "denylist": []},
    )

    with pytest.raises(seo.LedgerInvalidError):
        seo.add_correction("Sligeo", "Sligo", path)

    assert len(_corrections(path)) == 1


def test_a_repeated_denylist_name_is_refused(tmp_path, monkeypatch):
    """A repeat makes `find_denylisted` report the same name twice in the
    gate's failure message, which reads like two separate problems."""
    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": ["Nowhereville"]})

    with pytest.raises(seo.LedgerInvalidError):
        seo.add_denylist_entry("nowhereville", path)

    assert seo.read(path).denylist == ["Nowhereville"]


# --- rules that contradict the other half of the ledger ---------------------


def test_a_correction_onto_a_denylisted_name_is_refused(tmp_path, monkeypatch):
    """`title_builder` substitutes, then `check_metadata_quality` hard-fails
    on the denylisted result -- so saving this rule guarantees a failed run,
    authored by the console itself."""
    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": ["Nowhereville"]})

    with pytest.raises(seo.LedgerInvalidError) as exc:
        seo.add_correction("Nowherevile", "Nowhereville", path)

    assert "denylist" in str(exc.value).lower()
    assert _corrections(path) == []


def test_denylisting_the_result_of_an_existing_correction_is_refused(tmp_path, monkeypatch):
    """The same contradiction reached from the other side."""
    path = _ledger(
        tmp_path,
        monkeypatch,
        {"corrections": [{"wrong": "Sligeo", "correct": "Sligo"}], "denylist": []},
    )

    with pytest.raises(seo.LedgerInvalidError) as exc:
        seo.add_denylist_entry("Sligo", path)

    assert "Sligeo" in str(exc.value)
    assert seo.read(path).denylist == []


# --- accept-cases: the editor still works -----------------------------------


def test_an_ordinary_correction_still_saves_and_loads(tmp_path, monkeypatch):
    """The guard must not break the feature: a genuinely new rule saves, and
    the pipeline loader picks it up -- the round-trip this module exists for."""
    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": []})

    seo.add_correction("Sligeo", "Sligo", path)

    assert _corrections(path) == [{"wrong": "Sligeo", "correct": "Sligo"}]
    assert load_place_corrections(path).mapping == {"Sligeo": "Sligo"}


def test_a_second_rule_for_a_different_name_still_saves(tmp_path, monkeypatch):
    """Only the SAME wrong name collides. Two unrelated rules coexist."""
    path = _ledger(
        tmp_path,
        monkeypatch,
        {"corrections": [{"wrong": "Sligeo", "correct": "Sligo"}], "denylist": []},
    )

    seo.add_correction("Kery", "Kerry", path)

    assert len(_corrections(path)) == 2


def test_an_ordinary_denylist_entry_still_saves(tmp_path, monkeypatch):
    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": []})

    seo.add_denylist_entry("Nowhereville", path)

    assert load_place_corrections(path).denylist == ("Nowhereville",)


def test_a_denylist_name_that_only_matches_a_wrong_side_still_saves(tmp_path, monkeypatch):
    """Denylisting the name a correction REPLACES is coherent -- belt and
    braces, not a contradiction. Only the correction's result conflicts."""
    path = _ledger(
        tmp_path,
        monkeypatch,
        {"corrections": [{"wrong": "Sligeo", "correct": "Sligo"}], "denylist": []},
    )

    seo.add_denylist_entry("Sligeo", path)

    assert seo.read(path).denylist == ["Sligeo"]


# --- the page says why ------------------------------------------------------


def test_the_page_refuses_a_colliding_rule_and_says_which_one(tmp_path, monkeypatch):
    """A 500, or a silent redirect back to an unchanged page, both leave the
    operator believing the rule they typed is now in force."""
    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    path = _ledger(
        tmp_path,
        monkeypatch,
        {"corrections": [{"wrong": "Sligeo", "correct": "Sligo"}], "denylist": []},
    )
    client = TestClient(create_app(), raise_server_exceptions=False)

    resp = client.post(
        "/seo/corrections",
        data={"wrong": "Sligeo", "correct": "County Sligo", "revision": seo.read(path).revision},
        follow_redirects=False,
    )

    assert resp.status_code == 400
    assert "Sligo" in resp.text
    assert _corrections(path) == [{"wrong": "Sligeo", "correct": "Sligo"}]


def test_the_page_still_accepts_a_good_correction(tmp_path, monkeypatch):
    """Accept-case for the route: a real edit still redirects back to /seo."""
    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    path = _ledger(tmp_path, monkeypatch, {"corrections": [], "denylist": []})
    client = TestClient(create_app())

    resp = client.post(
        "/seo/corrections",
        data={"wrong": "Sligeo", "correct": "Sligo", "revision": seo.read(path).revision},
        follow_redirects=False,
    )

    assert resp.status_code == 303
    assert _corrections(path) == [{"wrong": "Sligeo", "correct": "Sligo"}]
