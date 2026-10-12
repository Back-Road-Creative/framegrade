"""The last two console forms that threw away what you typed.

`/drives/run` and `/photos/run` already keep what you typed. Two routes were left answering
a rejected field with a separate error page:

* `POST /jobs/schedules` -- name, subcommand and interval. `interval_minutes`
  was typed `int = Form(...)`, so "every hour" never reached the route at all:
  FastAPI answered with a 422 whose body is a list of pydantic error dicts.
* `POST /seo/corrections` -- the place-name correction and denylist forms. This
  one matters most on the **409**: a revision conflict means another editor
  wrote while this page was open, which is exactly when an operator's typed
  correction is worth keeping and exactly when it used to be discarded.

The other `HTTPException`s in these files are deliberately left alone: an
unknown `output_dir`, `state_path` or `sidecar_dir` is a 404 for a path that
does not exist, not a field an operator can correct on the page.
"""

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import seo
from gomoveshift.console.app import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "corrections.yaml"))
    return TestClient(create_app())


def _schedule(client, **over):
    form = {"name": "nightly", "subcommand": "publishing-status", "interval_minutes": "60"}
    form.update(over)
    return client.post("/jobs/schedules", data=form, follow_redirects=False)


def _correction(client, **over):
    form = {"wrong": "Kilkeny", "correct": "Kilkenny", "revision": ""}
    form.update(over)
    return client.post("/seo/corrections", data=form, follow_redirects=False)


def test_an_interval_that_is_not_a_number_is_answered_on_the_page(client):
    """`int = Form(...)` sent "every hour" to FastAPI's 422 handler, whose body
    is a list of pydantic dicts naming `interval_minutes`, not the label."""
    response = _schedule(client, interval_minutes="every hour")

    assert response.status_code == 400
    assert "Interval" in response.text
    assert "is not a number" in response.text
    assert "value_error" not in response.text


def test_a_rejected_schedule_keeps_its_name_and_interval(client):
    response = _schedule(client, interval_minutes="every hour")

    assert 'value="nightly"' in response.text
    assert 'value="every hour"' in response.text


def test_a_rejected_schedule_keeps_the_subcommand_that_was_chosen(client):
    response = _schedule(client, subcommand="publishing-status", interval_minutes="0")

    assert response.status_code == 400
    # Anchored on the schedule select's id: the page also carries the one-off
    # launch form, whose `subcommand` select comes first in the markup.
    chosen = response.text[response.text.index('id="schedule-subcommand"') :]
    assert 'value="publishing-status" selected' in chosen[: chosen.index("</select>")]


def test_a_rejected_schedule_still_lists_the_jobs_page_as_normal(client):
    """It re-renders the whole tab, not a stripped-down form: an operator who
    lands back here still has the schedule list and the job table."""
    response = _schedule(client, interval_minutes="every hour")

    assert "Add schedule" in response.text


def test_a_good_schedule_still_saves(client):
    """The accept case. Nothing here may stop the page doing its job."""
    response = _schedule(client)

    assert response.status_code == 303
    assert response.headers["location"] == "/jobs"


def test_a_half_filled_correction_comes_back_with_the_half_that_was_typed(client):
    response = _correction(client, correct="")

    assert response.status_code == 400
    assert 'value="Kilkeny"' in response.text


def test_an_empty_submission_says_what_it_wanted_without_a_separate_page(client):
    response = _correction(client, wrong="", correct="")

    assert response.status_code == 400
    assert "denylist" in response.text
    assert "Add correction" in response.text


def test_a_stale_revision_gives_the_correction_back_with_the_current_ledger(client):
    """The 409 case, and the reason this behaviour exists. Another editor wrote while
    this page was open. Discarding the operator's typed correction at exactly
    that moment is the worst time to do it -- and the page they get back now
    shows the other editor's entry, so their next submit is an informed one
    rather than a blind overwrite."""
    seo.add_correction("Sligoo", "Sligo")

    response = _correction(client, revision="a-revision-from-before-that-write")

    assert response.status_code == 409
    assert 'value="Kilkeny"' in response.text
    assert "Sligoo" in response.text


def test_a_stale_revision_hands_back_a_usable_revision_to_resubmit_with(client):
    """Handing back the stale one would make the retry fail the same way
    forever; the page now shows the current state, so the current revision is
    the honest thing to carry."""
    seo.add_correction("Sligoo", "Sligo")
    fresh = seo.read().revision

    response = _correction(client, revision="stale")

    assert f'name="revision" value="{fresh}"' in response.text


def test_a_rejected_denylist_entry_does_not_fill_the_correction_form(client):
    """Two forms on this page. Putting the denylist name into the correction
    boxes would be text in a form nobody touched."""
    response = client.post(
        "/seo/corrections",
        data={"wrong": "", "correct": "", "denylist_name": "   ", "revision": ""},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert 'name="wrong" value=""' in response.text


def test_a_good_correction_still_saves(client):
    response = _correction(client)

    assert response.status_code == 303
    assert response.headers["location"] == "/seo"
    assert any(entry["wrong"] == "Kilkeny" for entry in seo.read().corrections)


def test_a_good_denylist_entry_still_saves(client):
    response = client.post(
        "/seo/corrections",
        data={"wrong": "", "correct": "", "denylist_name": "Ballygobackwards", "revision": ""},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "Ballygobackwards" in seo.read().denylist
