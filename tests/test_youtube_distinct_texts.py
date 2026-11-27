# ruff: noqa: F811  (fixtures are imported from the adapter test module)
"""No two YouTube posts on one channel carry near-identical titles or descriptions.

2026-08: the channel was likely suspended under YouTube's repetitive-content
rules; runs of Shorts shared one title ("Virginia to Pennsylvania"), and every
Short of a drive that named one place sent the same reviewed description,
"Drive through <place>". Generation now makes both distinct from the row's own
facts, enqueue refuses a near-identical text, publish re-checks and holds, and
each drain pass holds the later of two colliding queued rows. Synthetic data.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest
from test_cli_enqueue_plan import _media_for, _record, _sidecar
from test_publishing_youtube_adapter import (  # noqa: F401 -- fixtures
    _base_record,
    _measured_route_record,
    _publish,
    _setup,
    _titled,
    mock_service,
    queue_conn,
)

from gomoveshift.health.store import load_all
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline import social_generator as sg
from gomoveshift.pipeline.drive_phases import youtube_uploader as youtube_uploader_module
from gomoveshift.pipeline.social_generator import SocialGenerator, TitleFacts
from gomoveshift.publishing import dispatch
from gomoveshift.publishing.enqueue_from_plan import enqueue_from_plan_sidecar
from gomoveshift.publishing.models import PlatformPayloads, YouTubePayload
from gomoveshift.publishing.queue import find_delivery, open_queue_db, transition
from gomoveshift.publishing.youtube_adapter import build_short_final_text_review
from gomoveshift.publishing.youtube_distinct import (
    HOLD_ERROR_CLASS,
    hold_near_duplicate_rows,
    near_identical,
)

PLACE = "County Kerry"
DATE = "2026-08-20"  # the adapter fixture's recording date


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())
    monkeypatch.setattr(SocialGenerator, "_meta_ranked", lambda self, platform, order: order)
    monkeypatch.setattr(sg, "shipped_title_snapshot", lambda: None)
    monkeypatch.setattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", True, raising=False)
    monkeypatch.setenv("GMS_HEALTH_INCIDENTS_DIR", str(tmp_path / "incidents"))


# --- the rule ---------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        ("Virginia to Pennsylvania", "virginia to pennsylvania"),
        ("Virginia to Pennsylvania", "Virginia to Pennsylvania 🚗 #Shorts #POVDriving"),
        ("Tryon", "Tryon drive"),
        ("Tryon", "TRYON — POV"),
        ("Dún Laoghaire", "Dun Laoghaire!"),
        ("Drive through Tryon", "Drive through Tryon\nFull drive: https://youtu.be/abcdefghijk"),
        (
            "A slow climb past the old mill, the church and the long stone bridge over the river",
            "A slow climb past the old mill, the chapel and the long stone bridge over the river",
        ),
    ],
)
def test_texts_that_say_the_same_thing_are_near_identical(a, b):
    assert near_identical(a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("Mar 5, 2nd stretch: Virginia to Pennsylvania", "Mar 5, 3rd stretch: Virginia to Pennsylvania"),
        ("Saluda", "Salida"),
        ("Drive through Tryon\nFull drive: https://youtu.be/A", "Drive through Saluda\nFull drive: https://youtu.be/A"),
        ("POV Pt.2: driving through X", "POV Pt.5: driving through X"),
        ("", ""),
    ],
)  # fmt: skip
def test_a_fact_from_the_row_tells_two_texts_apart(a, b):
    assert not near_identical(a, b)


# --- generation -------------------------------------------------------------


def _generated(n):
    gen = SocialGenerator(
        location=PLACE, project_name="drive", series_part=1, spooled_texts={}, experiments=()
    )
    return gen.generate_all(
        num_shorts=n,
        per_short_locations=[PLACE] * n,
        per_short_regions=[[PLACE]] * n,
        per_short_title_facts=[TitleFacts(DATE, k, 38, True) for k in range(1, n + 1)],
    )


def _reviewable(title):
    record = _titled(_base_record(routes=("youtube",)), title)
    return dataclasses.replace(record, route_record=_measured_route_record())


def _sent_description(title):
    from seo_manager.analytics.final_text_surfaces import compose_description

    review = build_short_final_text_review(_reviewable(title))
    assert review["verdict"] == "agree", review["reasons"]
    return compose_description(review["surfaces"])


def test_twenty_one_shorts_of_one_place_get_distinct_titles_and_descriptions():
    titles = [p.youtube_title for p in _generated(21)]
    descriptions = [_sent_description(t) for t in titles]
    for field in (titles, descriptions):
        clashes = [(a, b) for a, b in itertools.combinations(field, 2) if near_identical(a, b)]
        assert clashes == [], clashes[:3]


def test_a_place_published_under_a_near_identical_title_is_not_reused(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: {"County Kerry drive 🚗"})
    (post,) = _generated(1)
    assert not near_identical(post.youtube_title, "County Kerry drive 🚗"), post.youtube_title


# --- enqueue ----------------------------------------------------------------


def _short(index, title, description):
    payloads = PlatformPayloads(youtube=YouTubePayload(title, description))
    return dataclasses.replace(
        _record(short_index=index, media=_media_for(index)), payloads=payloads
    )


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    yield c
    c.close()


def _enqueue(conn, tmp_path, name, *records):
    out = tmp_path / name
    out.mkdir()
    return enqueue_from_plan_sidecar(conn, _sidecar(out, *records), review=False)


@pytest.mark.parametrize(
    "second",
    [
        ("Virginia to Pennsylvania 🚗", "POV Pt.2: driving through Virginia to Pennsylvania"),
        ("Mar 5, 2nd stretch: Virginia to Pennsylvania", "Front-seat view of the leg, Pt.1"),
    ],
    ids=["title", "description"],
)
def test_enqueue_refuses_a_near_identical_youtube_text(conn, tmp_path, second):
    first = _short(1, "Virginia to Pennsylvania", "Front-seat view of the leg, Pt.1")
    _enqueue(conn, tmp_path, "a", first)
    result = _enqueue(conn, tmp_path, "b", _short(2, *second))
    assert result.enqueued == ()
    ((_, index, reason),) = result.refused
    assert index == 2 and reason.startswith("near_duplicate:youtube")
    rows, _ = load_all(tmp_path / "incidents")
    assert [r["kind"] for r in rows] == ["near_duplicate_text_refused"]


def test_two_shorts_of_one_plan_are_compared_too(conn, tmp_path):
    shorts = (
        _short(1, "Tryon", "Pt.1"),
        _short(2, "Tryon drive", "x"),
        _short(3, "Saluda", "Pt.3"),
    )
    result = _enqueue(conn, tmp_path, "a", *shorts)
    assert [r[1] for r in result.refused] == [2]
    assert [e[1] for e in result.enqueued] == [1, 3]


# --- publish re-check and the drain sweep -------------------------------------


def _published_twin(queue_conn, tmp_path, title):
    """A row already on the channel, queued from its own sidecar."""
    twin = dataclasses.replace(_reviewable(title), project_name="other-drive", short_index=1)
    out = tmp_path / "twin"
    out.mkdir()
    path = _sidecar(out, twin)
    from gomoveshift.publishing.queue import enqueue

    row = enqueue(queue_conn, "other-drive", 1, "youtube", "h", sidecar_path=str(path))
    transition(
        queue_conn,
        row,
        "queued",
        "published",
        platform_upload_id="OLDVID",
        published_at_source="platform_confirmed",
    )
    return row


def test_publish_holds_a_short_whose_title_is_already_on_the_channel(
    mock_service, queue_conn, tmp_path
):
    adapter, record, did, vp = _setup(mock_service, queue_conn, tmp_path)
    _published_twin(queue_conn, tmp_path, "county kerry DRIVE 🚗")
    result = _publish(adapter, record, queue_conn, did, vp)
    assert result.held and not result.video_id
    assert mock_service.videos.return_value.insert.call_count == 0
    row = find_delivery(queue_conn, record.project_name, record.short_index, "youtube")
    assert row["status"] == "needs_operator" and row["error_class"] == HOLD_ERROR_CLASS
    assert "title is near-identical" in row["error_detail"]


def test_publish_proceeds_when_the_channel_holds_nothing_alike(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _setup(mock_service, queue_conn, tmp_path)
    _published_twin(queue_conn, tmp_path, "Aug 20, 2nd stretch: County Kerry")
    assert _publish(adapter, record, queue_conn, did, vp).video_id == "SHORTVID123"


def test_the_drain_holds_the_later_of_two_queued_collisions_and_raises_an_incident(conn, tmp_path):
    _enqueue(conn, tmp_path, "a", _short(1, "Tryon", "Pt.1"))
    _enqueue(conn, tmp_path, "b", _short(2, "Saluda", "Pt.2"))
    # Queued before this guard existed: written straight through the queue.
    path = tmp_path / "c"
    path.mkdir()
    late = _sidecar(path, dataclasses.replace(_short(3, "Tryon 🚗", "Pt.3"), project_name="p2"))
    from gomoveshift.publishing.queue import enqueue

    late_id = enqueue(conn, "p2", 3, "youtube", "h", sidecar_path=str(late))
    sent = []
    result = dispatch.drain_queue(
        conn, {"youtube": lambda row: sent.append(row["id"])}, incidents_dir=tmp_path / "incidents"
    )
    assert result.held_near_duplicate == 1
    assert sent and late_id not in sent  # one post per platform per pass (#639)
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (late_id,)).fetchone()
    assert row["status"] == "needs_operator" and row["error_class"] == HOLD_ERROR_CLASS
    assert hold_near_duplicate_rows(conn) == []  # a held row blocks nothing and is held once
    rows, _ = load_all(tmp_path / "incidents")
    (incident,) = [r for r in rows if r["kind"] == "delivery_needs_operator"]
    assert "held before upload" in incident["detail"] and "Tryon" in incident["detail"]


def test_a_hold_and_an_auth_pause_never_overwrite_each_other(conn, tmp_path):
    from gomoveshift.publishing.queue import (
        AUTH_PAUSED_ERROR_CLASS,
        enqueue,
        pause_for_auth,
        resume_platform,
    )

    _enqueue(conn, tmp_path, "a", _short(1, "Tryon", "Pt.1"))
    rows = []
    for index, project in ((2, "p2"), (3, "p3")):
        out = tmp_path / project
        out.mkdir()
        record = dataclasses.replace(_short(index, "Tryon 🚗", f"Pt.{index}"), project_name=project)
        rows.append(
            enqueue(conn, project, index, "youtube", "h", sidecar_path=str(_sidecar(out, record)))
        )
    paused, held = rows
    # The auth pause held the first collision back with its own reason.
    transition(conn, paused, "queued", "queued", error_class=AUTH_PAUSED_ERROR_CLASS)
    assert hold_near_duplicate_rows(conn) == [held]

    class RefusedError(Exception):
        queue_state = "auth_blocked"

    pause_for_auth(conn, "youtube", RefusedError("token refused"))
    resume_platform(conn, "youtube")
    got = dict(conn.execute("SELECT id, status || ':' || error_class FROM deliveries").fetchall())
    assert got[paused] == f"queued:{AUTH_PAUSED_ERROR_CLASS}"
    assert got[held] == f"needs_operator:{HOLD_ERROR_CLASS}"
