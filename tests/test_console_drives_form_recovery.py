"""A rejected drive form keeps what you typed.

`POST /drives/run` answered every rejected field with a 400 error page. The
form has thirteen inputs -- two directory paths, a region, a publish day, a
long-form day and mode, a target duration, two shorts numbers and three
checkboxes -- and an operator who mistyped one of them got a page with the
error on it and no way back to the other twelve but the browser's history,
which does not reliably restore a POSTed form and never restores one after a
redirect.

The rule is that a field failure is an inline correction with
the original form preserved, not a separate page. These pin that: the form
comes back, with the values in it, with the message attached, and still
refusing to launch.
"""

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    return TestClient(create_app())


def _submit(client, **over):
    form = {
        "directory": "/media/card",
        "output_dir": "/media/out",
        "region": "ireland",
        "publish": "wed",
        "lf_day": "mon",
        "lf_mode": "",
        "lf_target_duration": "",
        "shorts_target": "",
        "shorts_gap": "",
        "dry_run": "on",
    }
    form.update(over)
    return client.post("/drives/run", data=form, follow_redirects=False)


def test_a_bad_number_is_refused_without_launching(client):
    response = _submit(client, shorts_target="seven")

    assert response.status_code == 400
    assert "/jobs/" not in response.headers.get("location", "")


def test_the_form_comes_back_instead_of_an_error_page(client):
    response = _submit(client, shorts_target="seven")

    assert '<form method="post" action="/drives/run"' in response.text


def test_the_paths_the_operator_typed_are_still_there(client):
    """The two most expensive fields to retype, and the ones least likely to
    be the thing that was wrong."""
    response = _submit(client, directory="/media/card-04", shorts_target="seven")

    assert 'value="/media/card-04"' in response.text
    assert 'value="/media/out"' in response.text


def test_the_selected_day_is_still_selected(client):
    response = _submit(client, publish="thu", shorts_target="seven")

    assert '<option value="thu" selected' in response.text or (
        'value="thu"' in response.text and "selected" in response.text
    )


def test_a_ticked_checkbox_is_still_ticked(client):
    response = _submit(client, no_review="on", shorts_target="seven")

    assert response.text.count("checked") >= 2


def test_an_unticked_checkbox_is_still_unticked(client):
    """Dry run defaults ON. An operator who deliberately cleared it must not
    get it back ticked -- that turns a refused form into a live run."""
    form = {
        "directory": "/media/card",
        "output_dir": "/media/out",
        "shorts_target": "seven",
    }

    response = client.post("/drives/run", data=form, follow_redirects=False)

    assert 'name="dry_run" checked' not in response.text


def test_the_message_names_the_field_that_was_wrong(client):
    """`int("seven")` says "invalid literal for int() with base 10", which
    tells an operator nothing about which of thirteen inputs to fix."""
    response = _submit(client, shorts_target="seven")

    assert "Shorts per long-form" in response.text
    assert "is not a number" in response.text


def test_a_bad_duration_names_its_own_field(client):
    response = _submit(client, lf_target_duration="twelve")

    assert response.status_code == 400
    assert "Long-form target duration" in response.text
    assert "is not a number" in response.text


def test_an_unknown_brand_is_a_form_error_not_a_page(client):
    """The brand is a field on this form like any other."""
    response = _submit(client, brand="not-a-brand")

    assert response.status_code == 400
    assert '<form method="post" action="/drives/run"' in response.text


def test_a_good_form_still_launches(client, monkeypatch):
    """The accept case. Nothing here may make a valid submission render a
    page instead of starting the run."""
    monkeypatch.setattr("gomoveshift.console.app.jobs.launch", lambda *a, **k: 42)

    response = _submit(client)

    assert response.status_code == 303
    assert response.headers["location"] == "/jobs/42"
