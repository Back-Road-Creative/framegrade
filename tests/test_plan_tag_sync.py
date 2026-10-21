"""A queued Short whose sidecar predates the Short tag rule is brought into line.

The Short tag rule limits a Short's YouTube tags to Waypost tag templates and the
clip's own counties and roads. A plan written earlier with free tags (``driving``,
``travel``) would have the final-text review refuse its queued deliveries
(``tag_unreviewable:driving``), and would refuse every queued row of that plan. ``gomoveshift queue sync-plan-tags`` rewrites each non-terminal YouTube row's
sidecar tags through the same ``reviewable_short_tags`` the generator now uses: local file
write only, dry run unless ``--apply``. All data is synthetic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_cli_enqueue_plan import _record, _sidecar

from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.publishing import youtube_adapter as ya
from gomoveshift.publishing.models import PlatformPayloads, YouTubePayload
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.plan_title_sync import sync_plan_tags
from gomoveshift.publishing.queue import enqueue, open_queue_db

LEGACY = ("driving", "road trip", "travel", "a")


def _legacy_record(short_index=3):
    return _record(
        short_index=short_index,
        payloads=PlatformPayloads(youtube=YouTubePayload("t", "d", tags=LEGACY)),
    )


@pytest.fixture
def queue(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sidecar = _sidecar(out, _legacy_record(3), _legacy_record(4))
    conn = open_queue_db(tmp_path)
    queued = enqueue(conn, "proj", 3, "youtube", "HASH1", sidecar_path=str(sidecar))
    published = enqueue(conn, "proj", 4, "youtube", "HASH1", sidecar_path=str(sidecar))
    conn.execute("UPDATE deliveries SET status='published' WHERE id=?", (published,))
    conn.commit()
    yield SimpleNamespace(conn=conn, sidecar=sidecar, queued=queued, dir=tmp_path)
    conn.close()


def _tags(sidecar, short_index):
    plan = load_publish_plan(sidecar)
    return next(s for s in plan.shorts if s.short_index == short_index).payloads.youtube.tags


def test_dry_run_reports_and_writes_nothing(queue):
    changes = sync_plan_tags(queue.conn, apply=False)
    assert [(c.delivery_id, c.new) for c in changes] == [(queue.queued, ("road trip", "a"))]
    assert _tags(queue.sidecar, 3) == LEGACY


def test_apply_rewrites_only_the_non_terminal_row(queue):
    sync_plan_tags(queue.conn, apply=True)
    assert _tags(queue.sidecar, 3) == ("road trip", "a")
    assert _tags(queue.sidecar, 4) == LEGACY  # published: the live video keeps its record
    logged = queue.conn.execute(
        "SELECT count(*) FROM response_log WHERE action = 'sync_plan_tags'"
    ).fetchone()[0]
    assert logged == 1
    assert sync_plan_tags(queue.conn, apply=True) == []


def test_synced_tags_pass_the_short_review(queue):
    sync_plan_tags(queue.conn, apply=True)
    record = next(s for s in load_publish_plan(queue.sidecar).shorts if s.short_index == 3)
    seg = "proj-short003"
    places = ya._clip_places(record, seg)
    ya._short_text_entries(record, places, ya._clip_roads(record, seg), places[0], seg)


def test_cli_dry_run(queue, capsys):
    args = SimpleNamespace(queue_action="sync-plan-tags", apply=False)
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(queue.dir)))
    assert cmd_queue(args, config) == 0
    out = capsys.readouterr().out
    assert "would sync" in out and "'driving'" in out
