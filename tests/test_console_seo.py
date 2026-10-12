"""Console SEO corrections editor + comments inbox (minimal slice)."""

import yaml
from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app
from gomoveshift.pipeline.validation.place_corrections import load_place_corrections


def _client() -> TestClient:
    return TestClient(create_app())


def _write_ledger(path, corrections=None, denylist=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"corrections": corrections or [], "denylist": denylist or []}))


def test_seo_get_renders_existing_corrections_and_denylist(tmp_path, monkeypatch):
    ledger = tmp_path / "place_corrections.yaml"
    _write_ledger(
        ledger, corrections=[{"wrong": "Sligeo", "correct": "Sligo"}], denylist=["Nowhereville"]
    )
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(ledger))

    response = _client().get("/seo")

    assert response.status_code == 200
    assert "Sligeo" in response.text
    assert "Sligo" in response.text
    assert "Nowhereville" in response.text


def test_seo_get_missing_ledger_is_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "does-not-exist.yaml"))

    response = _client().get("/seo")

    assert response.status_code == 200
    assert "SEO" in response.text


def test_post_correction_round_trips_through_the_real_loader(tmp_path, monkeypatch):
    ledger = tmp_path / "place_corrections.yaml"
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(ledger))

    response = _client().post(
        "/seo/corrections",
        data={"wrong": "Sligeo", "correct": "Sligo"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    loaded = load_place_corrections(ledger)
    assert loaded.mapping == {"Sligeo": "Sligo"}

    second_get = _client().get("/seo")
    assert "Sligeo" in second_get.text
    assert "Sligo" in second_get.text


def test_post_denylist_entry_round_trips_through_the_real_loader(tmp_path, monkeypatch):
    ledger = tmp_path / "place_corrections.yaml"
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(ledger))

    response = _client().post(
        "/seo/corrections",
        data={"denylist_name": "Nowhereville"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    loaded = load_place_corrections(ledger)
    assert loaded.denylist == ("Nowhereville",)

    second_get = _client().get("/seo")
    assert "Nowhereville" in second_get.text


def test_post_corrections_requires_a_pair_or_a_denylist_name(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "place_corrections.yaml"))

    response = _client().post("/seo/corrections", data={})

    assert response.status_code == 400


def test_comments_inbox_renders_fixture_rows(tmp_path, monkeypatch):
    comments_path = tmp_path / "comments.jsonl"
    comments_path.write_text(
        '{"platform": "youtube", "video_id": "abc123", "author": "Jo", '
        '"text": "Actually this is in Sligo", "comment_id": "c1"}\n'
    )
    monkeypatch.setenv("GMS_COMMENTS_PATH", str(comments_path))

    response = _client().get("/comments")

    assert response.status_code == 200
    assert "youtube" in response.text
    assert "abc123" in response.text
    assert "Sligo" in response.text


def test_comments_inbox_missing_store_is_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_COMMENTS_PATH", str(tmp_path / "does-not-exist.jsonl"))

    response = _client().get("/comments")

    assert response.status_code == 200
    assert "No comments" in response.text
