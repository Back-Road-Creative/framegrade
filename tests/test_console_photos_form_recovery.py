"""A rejected photos form keeps what you typed.

`POST /photos/run` is the same defect the drives form already fixed, one page over.
It backs TWO forms -- ingest (five inputs) and develop (sixteen, eleven of
them numbers a person tunes by eye and re-enters from nothing if they are
lost) -- and answered any rejected field with a 400 error page. Worse than
drives: the eleven numeric fields were typed `float | None`, so a typo never
reached the route at all. FastAPI answered it with a 422 validation page whose
body is a JSON list of pydantic error dicts -- not a sentence, not a form, and
naming the field by its wire name rather than its label.

These pin the same rule: a field failure is an inline correction with the
original form preserved, not a separate page. The form comes back, with the
values in it, with a sentence attached, and still refusing to launch.

Two things here that the drives page did not have to get right:

* **Only the submitted form comes back filled.** The page renders both;
  restoring typed values into the one the operator was not using would put
  text in a form they never touched.
* **Ingest's "Dry run" defaults ON.** Re-ticking one the operator deliberately
  cleared would turn a corrected form into a live run over their card.
"""

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    return TestClient(create_app())


def _ingest(client, **over):
    form = {
        "action": "ingest",
        "card_dir": "/media/card",
        "journey": "west-coast",
        "day": "2026-09-01",
        "location": "County Clare",
        "dry_run": "on",
    }
    form.update(over)
    return client.post("/photos/run", data=form, follow_redirects=False)


def _develop(client, **over):
    form = {
        "action": "develop",
        "paths": "/photos/a.raf\n/photos/b.raf",
        "out": "/photos/out",
        "styles": "vivid,mono",
        "look": "",
        "vignette": "0.3",
        "clarity": "0.2",
        "grain": "0.1",
        "grain_size": "1.5",
        "halation": "0.05",
        "white_balance_g": "0.02",
        "white_balance_r": "0.01",
        "white_balance_b": "-0.01",
        "white_balance_r_high": "0.0",
        "white_balance_b_high": "0.0",
        "white_balance_g_high": "0.0",
    }
    form.update(over)
    return client.post("/photos/run", data=form, follow_redirects=False)


def test_an_unusable_number_comes_back_as_a_sentence_naming_the_field(client):
    """`float | None = Form(None)` sent a typo to FastAPI's 422 handler, whose
    body is a list of pydantic dicts. An operator cannot read that, and it
    names `vignette`, not "Vignette"."""
    response = _develop(client, vignette="quite a lot")

    assert response.status_code == 400
    assert "Vignette" in response.text
    assert "is not a number" in response.text
    assert "value_error" not in response.text


def test_a_rejected_develop_form_still_holds_the_other_fifteen_fields(client):
    response = _develop(client, vignette="quite a lot")

    for kept in ("/photos/out", "vivid,mono", "1.5", "0.05", "/photos/a.raf"):
        assert kept in response.text, kept


def test_a_rejected_develop_form_keeps_the_number_that_was_wrong(client):
    """Showing the field empty asks the operator to remember what they typed;
    showing it back is what lets them see the typo."""
    response = _develop(client, clarity="0..2")

    assert 'name="clarity" value="0..2"' in response.text


def test_a_rejected_ingest_form_holds_its_own_four_fields(client):
    response = _ingest(client, day="")

    assert response.status_code == 400
    for kept in ("/media/card", "west-coast", "County Clare"):
        assert kept in response.text, kept


def test_a_rejected_ingest_form_does_not_fill_the_develop_form(client):
    """Both forms are on the page. Putting the card path into the develop
    form's output box would be text in a form nobody touched."""
    response = _ingest(client, day="")

    assert 'name="out" value=""' in response.text or 'name="out">' in response.text


def test_a_cleared_dry_run_stays_cleared_through_a_refusal(client):
    """Ingest's Dry run defaults ON. Re-ticking one the operator deliberately
    cleared would turn a corrected form into a live run over their card."""
    response = _ingest(client, day="", dry_run="")

    body = response.text
    checkbox = body[body.index('name="dry_run"') - 40 : body.index('name="dry_run"') + 40]
    assert "checked" not in checkbox


def test_a_kept_dry_run_comes_back_ticked(client):
    response = _ingest(client, day="")

    body = response.text
    checkbox = body[body.index('name="dry_run"') - 40 : body.index('name="dry_run"') + 40]
    assert "checked" in checkbox


def test_a_fresh_photos_page_still_offers_dry_run_ticked(client):
    """The accept case for the trap above: no submitted form at all means the
    safe default, not the empty-checkbox reading of a refusal."""
    body = client.get("/photos").text

    checkbox = body[body.index('name="dry_run"') - 40 : body.index('name="dry_run"') + 40]
    assert "checked" in checkbox


def test_an_unknown_action_is_answered_on_the_page_too(client):
    response = _develop(client, action="reel")

    assert response.status_code == 400
    assert "reel" in response.text


def test_a_refusal_does_not_launch_anything(client):
    """The whole point is a correction, not a run. A 400 that still queued the
    job would be worse than the error page it replaced."""
    _develop(client, vignette="quite a lot")

    assert client.get("/jobs").text.count("<tr") <= 1


def test_a_good_develop_form_still_launches(client, tmp_path):
    """The accept case. Nothing here may stop the page doing its job."""
    photo = tmp_path / "a.raf"
    photo.write_bytes(b"x")
    response = _develop(client, paths=str(photo), out=str(tmp_path / "out"))

    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs/")


def test_a_good_ingest_form_still_launches(client):
    response = _ingest(client)

    assert response.status_code == 303
    assert response.headers["location"].startswith("/jobs/")


def test_a_blank_required_ingest_field_is_named(client):
    """The page marks card dir / journey / day `required`; a browser enforces
    that and a direct POST does not. Blank, they used to build `--day ''` and
    launch a job that could only fail inside the log -- by which point the
    form is gone."""
    response = _ingest(client, journey="")

    assert response.status_code == 400
    assert "journey" in response.text
