"""A Short the final-text review would refuse at dispatch is refused at enqueue.

``short_review_refusal`` ran only at dispatch, so a refused Short sat queued, burnt its
attempts and lost its slot. The plan builder now repairs the one mechanical fault (an
unreviewable tag) and enqueue refuses what still fails, raising one health incident per
Short. All data is synthetic."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest import mock

import pytest
from test_cli_enqueue_plan import _approved_state, _args, _config, _media_for
from test_publishing_plan_builder import _build, _post
from test_publishing_youtube_adapter import _measured_route_record
from test_publishing_models import _record as _base_record

from gomoveshift.cli.handlers.enqueue_plan import cmd_enqueue_plan
from gomoveshift.health.store import load_all
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.publishing.enqueue_from_plan import enqueue_from_plan_sidecar
from gomoveshift.publishing.models import PublishPlan
from gomoveshift.publishing.queue import open_queue_db
from gomoveshift.publishing.youtube_adapter import short_review_refusal

TITLE = "County Kerry drive"
DESC = "A quiet drive through County Kerry."
BAD_TAG = "Ireland road trip"


def _short(short_index, *, tags=(), title=TITLE):
    record = _base_record(routes=("youtube",))
    yt = dataclasses.replace(record.payloads.youtube, title=title, description=DESC, tags=tags)
    return dataclasses.replace(
        record,
        short_index=short_index,
        payloads=dataclasses.replace(record.payloads, youtube=yt),
        route_record=_measured_route_record(),
        media=_media_for(short_index),
    )


def _sidecar(tmp_path, *records):
    path = tmp_path / "p_publish_plan.json"
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=tuple(records))), encoding="utf-8")
    return path


@pytest.fixture
def incidents(tmp_path, monkeypatch):
    out = tmp_path / "incidents"
    monkeypatch.setenv("GMS_HEALTH_INCIDENTS_DIR", str(out))
    return out


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    yield c
    c.close()


def _indexes(conn):
    return sorted(r[0] for r in conn.execute("SELECT short_index FROM deliveries"))


def test_fixture_good_short_passes_and_bad_tag_short_is_refused():
    assert short_review_refusal(_short(1)) is None
    assert "tag_unreviewable" in (short_review_refusal(_short(2, tags=(BAD_TAG,))) or "")


def test_plan_builder_drops_an_unreviewable_tag_and_the_short_passes():
    post = _post(youtube_tags=f"County Kerry,{BAD_TAG}")
    record = _build(posts=[post]).shorts[0]
    assert BAD_TAG not in record.payloads.youtube.tags
    assert "tag_unreviewable" not in (short_review_refusal(record) or "")


def test_refused_short_is_not_enqueued_and_raises_an_incident(tmp_path, conn, incidents):
    # Its own title and description: enqueue refuses a second Short with either
    # already queued, and a stamped title's description carries its stamp.
    third = _short(3, title="Aug 20, 3rd stretch: County Kerry")
    sidecar = _sidecar(tmp_path, _short(1), _short(2, tags=(BAD_TAG,)), third)
    result = enqueue_from_plan_sidecar(conn, sidecar)
    assert _indexes(conn) == [1, 3]
    assert [(r[0], r[1]) for r in result.refused] == [("example-2026-08", 2)]
    rows, bad = load_all(incidents)
    assert bad == 0 and len(rows) == 1
    assert rows[0]["kind"] == "short_review_refused"
    assert rows[0]["short_index"] == 2
    assert f"tag_unreviewable:{BAD_TAG}" in rows[0]["reason"]
    assert rows[0]["resolved_at"] is None


def test_repeated_enqueue_keeps_one_incident(tmp_path, conn, incidents):
    sidecar = _sidecar(tmp_path, _short(1), _short(2, tags=(BAD_TAG,)))
    enqueue_from_plan_sidecar(conn, sidecar)
    first = load_all(incidents)[0]
    enqueue_from_plan_sidecar(conn, sidecar)
    rows, _ = load_all(incidents)
    assert rows == first
    assert _indexes(conn) == [1]


def _run_cli(tmp_path, out, **over):
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")
    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=_approved_state(out)
    ):
        return cmd_enqueue_plan(_args(state_path, **over), _config(tmp_path))


def test_cli_dry_run_reports_refusals_and_writes_nothing(tmp_path, incidents, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _short(1), _short(2, tags=(BAD_TAG,)))
    assert _run_cli(tmp_path, out) == 0
    text = capsys.readouterr().out
    assert "DRY RUN" in text
    assert "short_index=2" in text and "tag_unreviewable" in text
    assert "short_index=1" not in text
    assert not incidents.exists()
    assert not (tmp_path / "queue.db").exists()


def test_cli_apply_enqueues_the_rest_and_reports_the_refusal(tmp_path, incidents, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _short(1), _short(2, tags=(BAD_TAG,), title="A County Kerry afternoon"))
    assert _run_cli(tmp_path, out, apply=True) == 0
    text = capsys.readouterr().out
    assert "enqueued=1" in text and "refused=1" in text
    assert len(load_all(incidents)[0]) == 1


def test_health_check_pushes_the_refusal_detail_and_returns(tmp_path, conn, incidents):
    from gomoveshift.health import run_check

    sidecar = _sidecar(tmp_path, _short(2, tags=(BAD_TAG,)))
    enqueue_from_plan_sidecar(conn, sidecar)
    pushed = []
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    result = run_check(
        SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "seo"))),
        incidents_dir=incidents,
        jobs_dir=jobs,
        code_root=tmp_path,
        notify_fn=lambda event, payload, text: pushed.append(text) or True,
    )
    key = "short_review_refused:example-2026-08:2"
    assert key in result["pushed"]
    assert any(
        "Short 2 of example-2026-08" in t and f"tag_unreviewable:{BAD_TAG}" in t for t in pushed
    )


def test_refused_short_is_not_counted_by_the_ceiling_or_anti_spam(tmp_path, incidents, capsys):
    out = tmp_path / "out"
    out.mkdir()
    # Same title prefix would trip the anti-spam gate; two Shorts would trip a ceiling of 1.
    _sidecar(out, _short(1), _short(2, tags=(BAD_TAG,)))
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")
    with mock.patch(
        "gomoveshift.cli.handlers.enqueue_plan._load_state", return_value=_approved_state(out)
    ):
        rc = cmd_enqueue_plan(_args(state_path, apply=True), _config(tmp_path, ceiling=1))
    text = capsys.readouterr().out
    assert rc == 0, text
    assert "enqueued=1" in text and "refused=1" in text
    assert "shared-prefix" not in text and "ceiling exceeded" not in text
