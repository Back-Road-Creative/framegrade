"""Plan title sync applies each video independently; retitle_apply refuses an
orphan title hashtag upstream. All data is synthetic."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from test_cli_enqueue_plan import _record, _sidecar

from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.publishing.models import PlatformPayloads, YouTubePayload
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import enqueue, open_queue_db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_retitle_apply import FakeService, _row, _write_worklist  # noqa: E402
from tools import retitle_apply as ra  # noqa: E402

BAD, GOOD = "SYNTHaaa", "SYNTHbbb"
ORPHAN_TITLE = "Somewhere #Nowhere"
GOOD_TITLE = "Another Place"


def _published(conn, tmp_path, name, vid, plan_hash):
    out = tmp_path / name
    out.mkdir()
    rec = replace(
        _record(payloads=PlatformPayloads(youtube=YouTubePayload("t", "plain description"))),
        project_name=name,
    )
    sidecar = _sidecar(out, rec)
    did = enqueue(conn, name, 3, "youtube", plan_hash, sidecar_path=str(sidecar))
    conn.execute(
        "UPDATE deliveries SET status='published', platform_upload_id=? WHERE id=?", (vid, did)
    )
    conn.commit()
    return sidecar


def _setup(tmp_path):
    conn = open_queue_db(tmp_path)
    bad = _published(conn, tmp_path, "pa", BAD, "H1")
    good = _published(conn, tmp_path, "pb", GOOD, "H2")
    conn.close()
    audit = tmp_path / "a.audit.jsonl"
    rows = [
        {"video_id": v, "old": "t", "new": n, "result": "APPLIED", "timestamp": "2026-10-06"}
        for v, n in ((BAD, ORPHAN_TITLE), (GOOD, GOOD_TITLE))
    ]
    audit.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    return bad, good, audit, config


def _run(audit, config, apply):
    args = SimpleNamespace(queue_action="sync-plan-titles", audit=[str(audit)], apply=apply)
    return cmd_queue(args, config)


def _title(path):
    return load_publish_plan(path).shorts[0].payloads.youtube.title


def test_apply_continues_past_a_refused_video(tmp_path, capsys):
    bad, good, audit, config = _setup(tmp_path)

    rc = _run(audit, config, apply=True)

    cap = capsys.readouterr()
    assert rc == 1
    assert _title(bad) == "t"
    assert _title(good) == GOOD_TITLE
    assert f"REFUSED {BAD}:" in cap.err
    assert "absent from its description" in cap.err
    assert "synced delivery" in cap.out and GOOD in cap.out
    assert "1 synced, 1 refused" in cap.out


def test_dry_run_predicts_the_refusal(tmp_path, capsys):
    bad, good, audit, config = _setup(tmp_path)
    before = bad.read_bytes(), good.read_bytes()

    rc = _run(audit, config, apply=False)

    cap = capsys.readouterr()
    assert rc == 1
    assert f"REFUSED {BAD}:" in cap.err
    assert GOOD in cap.out and "1 would sync, 1 refused" in cap.out
    assert (bad.read_bytes(), good.read_bytes()) == before
    assert not list(tmp_path.glob("p*/*.sync-tmp"))


def test_all_clean_returns_zero(tmp_path):
    bad, good, audit, config = _setup(tmp_path)
    audit.write_text(
        json.dumps(
            {"video_id": GOOD, "old": "t", "new": GOOD_TITLE, "result": "APPLIED", "timestamp": "x"}
        )
        + "\n",
        encoding="utf-8",
    )
    assert _run(audit, config, apply=True) == 0
    assert _title(good) == GOOD_TITLE


# --- retitle_apply upstream gate ---------------------------------------------


def test_row_with_orphan_title_hashtag_is_refused_by_current_description():
    row = _row(suggested_title="Clonmel to Clonakilty #Orphan", current_description="live text")
    assert any("#Orphan" in i for i in ra.hashtag_issues(row))


def test_row_hashtag_present_in_suggested_description_is_clean():
    row = _row(
        suggested_title="Clonmel to Clonakilty #Cork",
        suggested_description="a drive #cork",
        current_description="old",
    )
    assert ra.hashtag_issues(row) == []


def test_dry_run_without_any_description_defers_to_apply():
    assert ra.hashtag_issues(_row(suggested_title="X #Orphan")) == []


def test_apply_refuses_orphan_hashtag_against_live_description(tmp_path):
    row = _row(suggested_title="Clonmel to Clonakilty, County Cork #Orphan")
    path = _write_worklist(tmp_path, [row])
    service = FakeService(
        {"vid1": {"title": "To Clonakilty in 4K", "description": "plain", "categoryId": "19"}}
    )
    ra.main(["--worklist", str(path), "--apply", "--yes-live"], service=service)
    records = [json.loads(x) for x in path.with_suffix(".audit.jsonl").read_text().splitlines()]
    assert service.videos().update_calls == []
    assert [r["result"] for r in records if r["result"] != "SNAPSHOT"] == [ra.REFUSED]


def test_retry_reports_already_synced_and_writes_nothing(tmp_path, capsys):
    bad, good, audit, config = _setup(tmp_path)
    assert _run(audit, config, apply=True) == 1
    capsys.readouterr()
    after = good.read_bytes()

    rc = _run(audit, config, apply=True)

    cap = capsys.readouterr()
    assert rc == 1  # the refused video is still refused
    assert f"already synced {GOOD}" in cap.out
    assert "synced delivery" not in cap.out
    assert f"REFUSED {BAD}:" in cap.err
    assert good.read_bytes() == after


# --- retitle_apply: plan sync only follows a verified write -------------------


class _Boom:
    def __init__(self):
        self.calls = []

    def __call__(self, vid, title):
        self.calls.append((vid, title))
        raise ValueError("plan gate refused")


def _apply(tmp_path, snippet, plan_sync, read_back_lies=False):
    path = _write_worklist(tmp_path, [_row()])
    service = FakeService({"vid1": snippet})
    if read_back_lies:
        real = service.videos().update

        def update(part, body):  # the write is accepted but the title never lands
            body = {"id": body["id"], "snippet": snippet}
            return real(part, body)

        service.videos().update = update
    rows = ra.dry_run(ra.read_worklist(path), out=open(tmp_path / "o", "w"))
    err = open(tmp_path / "e", "w")
    ra.apply_updates(service, rows, err=err, sleep=lambda _s: None, plan_sync=plan_sync)
    err.close()
    return (tmp_path / "e").read_text()


def test_refused_plan_sync_is_reported_per_video_after_a_verified_write(tmp_path):
    boom = _Boom()
    err = _apply(tmp_path, {"title": "To Clonakilty in 4K", "categoryId": "19"}, boom)
    assert boom.calls == [("vid1", "Clonmel to Clonakilty, County Cork")]
    assert "vid1: WARNING plan title NOT synced (plan gate refused)" in err


def test_no_plan_sync_when_read_back_fails(tmp_path):
    boom = _Boom()
    snippet = {"title": "To Clonakilty in 4K", "categoryId": "19"}
    err = _apply(tmp_path, snippet, boom, read_back_lies=True)
    assert boom.calls == []
    assert "VERIFY FAILED" in err
