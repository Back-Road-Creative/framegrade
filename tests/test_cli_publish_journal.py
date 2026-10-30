"""End-to-end test of ``gomoveshift publish-journal`` against a tmp site dir.

Pure-merge logic covered in test_journal_publisher.py; this file pins the
CLI surface — argparse wiring, dispatch, exit codes, dry-run vs --apply,
the failure-mode exit codes.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.cli.handlers.publish_journal import (
    cmd_publish_journal,
    run_publish_journal,
    run_replay_pending,
)
from gomoveshift.pipeline.orchestrator_lock import _acquire_run_lock, _release_run_lock
from gomoveshift.pipeline.pending_journal import (
    PENDING_FILENAME,
    load_pending,
)

from _state_helpers import make_state

FIXTURE = Path(__file__).parent / "fixtures" / "journals" / "sample_trip.md"


def _fake_ffprobe(_path):
    return 600.0


def _make_site(tmp_path: Path, journey_slug: str) -> Path:
    site = tmp_path / "site"
    (site / "content" / "journeys").mkdir(parents=True)
    shutil.copy(FIXTURE, site / "content" / "journeys" / f"{journey_slug}.md")
    return site


def _write_state(tmp_path: Path, state) -> Path:
    path = tmp_path / "pipeline_state.json"
    payload = state.to_dict() if hasattr(state, "to_dict") else state
    path.write_text(json.dumps(payload))
    return path


@contextmanager
def _another_writer_holds(journey: Path):
    """Hold this journey's lock the way another writer does (a photo ingest
    mid-register, say), through the shared machinery directly — the test must
    not prove the lock with the very call it is exercising."""
    fd, lock_path = _acquire_run_lock(
        journey.parent, journey.parent, lock_name=journey.name + ".lock"
    )
    try:
        yield
    finally:
        _release_run_lock(fd, lock_path)


def test_merge_skips_loudly_while_another_writer_holds_the_journey(tmp_path, capsys):
    """Any doc this merge loads is already stale, so writing it back is a
    last-writer-wins clobber. Refuse recoverably — exit 1, loud, no write and
    NO exception: the drive hook calls this after the (irreversible) uploads."""
    site = _make_site(tmp_path, "sample-trip")
    state = make_state(
        youtube_uploads={"1": {"video_id": "LOCKED1", "title": "Clobber me"}},
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    state_path = _write_state(tmp_path, state)
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()

    with _another_writer_holds(journey):
        rc = run_publish_journal(
            state_path=state_path,
            site_path=site,
            journey_slug="sample-trip",
            apply=True,
            allow_day_create=False,
            ffprobe_duration=_fake_ffprobe,
        )

    assert rc == 1
    assert journey.read_text() == before, "a held journey must not be clobbered"
    assert "lock" in capsys.readouterr().err


def test_replay_keeps_entries_pending_while_another_writer_holds_the_journey(tmp_path):
    """Same window on the cron path: skip this journey and keep its entries
    pending (never consume what was never written) for the next sweep."""
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()

    with _another_writer_holds(journey):
        rc = run_replay_pending(pending_files=[pending], apply=True)

    assert rc == 0  # no auto_commit → a skipped journey is informational
    assert journey.read_text() == before, "a held journey must not be clobbered"
    assert [e.youtube_id for e in load_pending(pending)] == ["REPLAY_VID"]


def test_dry_run_prints_plan_and_does_not_write(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    state = make_state(
        youtube_uploads={"1": {"video_id": "DRY1", "title": "Dry-run insert"}},
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    state_path = _write_state(tmp_path, state)
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=False,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "DRY1" in captured.out
    assert "dry-run" in captured.out
    assert journey.read_text() == before, "dry-run must not write"


def test_apply_writes_journey_file(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    state = make_state(
        youtube_uploads={"1": {"video_id": "WROTE1", "title": "Real Insert"}},
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    state_path = _write_state(tmp_path, state)
    journey = site / "content" / "journeys" / "sample-trip.md"

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 0
    assert "WROTE1" in journey.read_text()
    assert "## About This Trip" in journey.read_text(), "body preserved"


def test_missing_state_file_returns_2(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    rc = run_publish_journal(
        state_path=tmp_path / "no-such-state.json",
        site_path=site,
        journey_slug="sample-trip",
        apply=False,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 2


def test_invalid_json_state_returns_2(tmp_path):
    site = _make_site(tmp_path, "sample-trip")
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json")
    rc = run_publish_journal(
        state_path=bad,
        site_path=site,
        journey_slug="sample-trip",
        apply=False,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 2


def test_empty_uploads_returns_1(tmp_path):
    site = _make_site(tmp_path, "sample-trip")
    state_path = _write_state(tmp_path, {"youtube_uploads": {}})
    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=False,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 1


def test_missing_journey_file_returns_1(tmp_path):
    site = _make_site(tmp_path, "other-trip")  # not the slug we'll ask for
    state_path = _write_state(
        tmp_path,
        {
            "youtube_uploads": {"1": {"video_id": "X"}},
            "per_lf_recording_date": {"1": "2025-09-03"},
            "regular_lf_paths": {"1": "/tmp/x.mp4"},
        },
    )
    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=False,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 1


def test_unmatched_date_returns_1_and_does_not_write(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    state_path = _write_state(
        tmp_path,
        {
            "youtube_uploads": {"1": {"video_id": "ORPHAN", "title": "x"}},
            "per_lf_recording_date": {"1": "2099-01-01"},
            "regular_lf_paths": {"1": "/tmp/x.mp4"},
        },
    )
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()
    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,  # even with --apply, errors block the write
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 1
    assert journey.read_text() == before


def test_subparser_is_registered():
    """Confirm ``gomoveshift publish-journal`` is dispatched by the top-level
    parser, not a typo."""
    from gomoveshift.cli import _COMMAND_MAP, _build_parser

    assert "publish-journal" in _COMMAND_MAP
    parser = _build_parser()
    args = parser.parse_args(
        [
            "publish-journal",
            "/tmp/state.json",
            "--journey",
            "example-2025-09",
            "--site-path",
            "/tmp/site",
        ]
    )
    assert args.command == "publish-journal"
    assert args.journey == "example-2025-09"
    assert args.apply is False
    assert args.allow_day_create is False


# ---------------------------------------------------------------------------
# Date-gate end-to-end — future publish_at lands in the pending sidecar
# ---------------------------------------------------------------------------


def test_apply_with_future_publish_at_writes_pending_sidecar_not_journey(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    state = make_state(
        youtube_uploads={
            "1": {
                "video_id": "FUTURE_VID",
                "title": "Future Insert",
                "publish_at": "2099-01-01T00:00:00Z",
            }
        },
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    state_path = _write_state(tmp_path, state)
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )

    assert rc == 0
    assert journey.read_text() == before, "future publish_at must not touch journey"
    pending_path = state_path.parent / PENDING_FILENAME
    assert pending_path.is_file()
    entries = load_pending(pending_path)
    assert len(entries) == 1
    assert entries[0].youtube_id == "FUTURE_VID"
    assert entries[0].publish_at == "2099-01-01T00:00:00Z"
    assert "Deferred 1 entry" in capsys.readouterr().out


def test_pending_dir_routes_sidecar_to_drive_output_not_state_dir(tmp_path):
    """The drive auto-hook passes pending_dir=<output_base>/<project> so the
    deferral lands where the replay cron's --scan-root looks, not in the
    ephemeral staging dir that holds pipeline_state.json."""
    site = _make_site(tmp_path, "sample-trip")
    state = make_state(
        youtube_uploads={
            "1": {"video_id": "FUT", "title": "Fut", "publish_at": "2099-01-01T00:00:00Z"}
        },
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    state_path = _write_state(tmp_path, state)
    out_dir = tmp_path / "drive_output"
    out_dir.mkdir()

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
        pending_dir=out_dir,
    )
    assert rc == 0
    assert (out_dir / PENDING_FILENAME).is_file(), "sidecar must land in pending_dir"
    assert not (state_path.parent / PENDING_FILENAME).is_file(), "not next to state file"
    assert [e.youtube_id for e in load_pending(out_dir / PENDING_FILENAME)] == ["FUT"]


def test_apply_with_mixed_publish_at_splits_inserts_and_deferrals(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    state = make_state(
        youtube_uploads={
            "1": {
                "video_id": "READY_VID",
                "title": "Ready",
                "publish_at": "2025-01-01T00:00:00Z",
            },
            "2": {
                "video_id": "WAITING_VID",
                "title": "Waiting",
                "publish_at": "2099-01-01T00:00:00Z",
            },
        },
        per_lf_recording_date={"1": "2025-09-03", "2": "2025-09-04"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4", "2": "/tmp/lf_02.mp4"},
    )
    state_path = _write_state(tmp_path, state)
    journey = site / "content" / "journeys" / "sample-trip.md"

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 0
    assert "READY_VID" in journey.read_text()
    assert "WAITING_VID" not in journey.read_text()
    entries = load_pending(state_path.parent / PENDING_FILENAME)
    assert [e.youtube_id for e in entries] == ["WAITING_VID"]


# ---------------------------------------------------------------------------
# Replay-pending mode
# ---------------------------------------------------------------------------


def _seed_pending_for_replay(
    tmp_path: Path, site: Path, publish_at: str, youtube_id: str = "REPLAY_VID"
) -> Path:
    """Write a pending_journal.json next to a synthetic state dir, returning the path."""
    pending_dir = tmp_path / "out"
    pending_dir.mkdir()
    journey = site / "content" / "journeys" / "sample-trip.md"
    pending_path = pending_dir / PENDING_FILENAME
    pending_path.write_text(
        json.dumps(
            [
                {
                    "youtube_id": youtube_id,
                    "publish_at": publish_at,
                    "title": "Replay Test",
                    "duration": "10:00",
                    "day_date": "2025-09-03",
                    "journey_path": str(journey),
                    "lf_index": 1,
                }
            ]
        )
    )
    return pending_path


def test_replay_pending_dry_run_does_not_write(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()

    rc = run_replay_pending(pending_files=[pending], apply=False)
    assert rc == 0
    assert journey.read_text() == before
    assert "dry-run" in capsys.readouterr().out


def test_replay_pending_apply_flushes_ready_entry(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    journey = site / "content" / "journeys" / "sample-trip.md"

    rc = run_replay_pending(pending_files=[pending], apply=True)
    assert rc == 0
    assert "REPLAY_VID" in journey.read_text()
    # Sidecar should be cleaned up after flushing.
    assert not pending.is_file()


def test_replay_pending_apply_keeps_future_entry(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2099-01-01T00:00:00Z")
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()

    rc = run_replay_pending(pending_files=[pending], apply=True)
    assert rc == 0
    assert journey.read_text() == before, "future entry must not flush"
    assert pending.is_file(), "sidecar must persist while entries are still future"
    remaining = load_pending(pending)
    assert [e.youtube_id for e in remaining] == ["REPLAY_VID"]


def test_replay_pending_takes_title_and_duration_from_youtube(tmp_path):
    """A replay writes what YouTube shows now, not what the drive stored: the
    video may have been retitled live since (Ace Basin LF 1) or stored a zero
    duration while still processing (Ace Basin LF 2)."""
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    journey = site / "content" / "journeys" / "sample-trip.md"
    asked = []

    def live(ids):
        asked.append(list(ids))
        return {"REPLAY_VID": {"title": "Retitled Live", "seconds": 5080.0}}

    rc = run_replay_pending(pending_files=[pending], apply=True, live_lookup=live)
    assert rc == 0
    assert asked == [["REPLAY_VID"]]
    text = journey.read_text()
    assert "Retitled Live" in text
    assert "1:24:40" in text
    assert "Replay Test" not in text


def test_replay_pending_keeps_stored_values_youtube_cannot_supply(tmp_path):
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    journey = site / "content" / "journeys" / "sample-trip.md"

    rc = run_replay_pending(
        pending_files=[pending],
        apply=True,
        live_lookup=lambda ids: {"REPLAY_VID": {"title": "", "seconds": 0.0}},
    )
    assert rc == 0
    text = journey.read_text()
    assert "Replay Test" in text
    assert "10:00" in text


def test_cmd_publish_journal_replay_looks_up_youtube(monkeypatch, tmp_path):
    import gomoveshift.cli.handlers.publish_journal as pj

    seen = {}
    monkeypatch.setattr(pj, "run_replay_pending", lambda **k: seen.update(k) or 0)
    assert cmd_publish_journal(_replay_args(tmp_path, str(tmp_path)), config=None) == 0
    assert seen["live_lookup"] is pj._default_youtube_live_lookup


def test_replay_pending_no_files_is_success(capsys):
    """An empty scan is the normal steady state, not a cron failure."""
    rc = run_replay_pending(pending_files=[], apply=True)
    assert rc == 0
    assert "no pending sidecars" in capsys.readouterr().out.lower()


def _replay_args(tmp_path, scan_root):
    args = MagicMock()
    args.replay_pending = True
    args.pending_file = None
    args.scan_root = scan_root
    args.apply = True
    args.auto_commit = False
    args.auto_push = False
    args.log_file = None
    return args


def test_cmd_publish_journal_replay_empty_scan_root_exits_0(tmp_path):
    """A real, empty --scan-root dir is a no-op success, not a cron alarm."""
    rc = cmd_publish_journal(_replay_args(tmp_path, str(tmp_path)), config=None)
    assert rc == 0


def test_cmd_publish_journal_replay_missing_scan_root_exits_nonzero(tmp_path):
    """A --scan-root that does not exist is an operator error, not a no-op."""
    missing = tmp_path / "does-not-exist"
    rc = cmd_publish_journal(_replay_args(tmp_path, str(missing)), config=None)
    assert rc != 0


def test_replay_pending_idempotent(tmp_path, capsys):
    """Second --apply run on an empty sidecar is a no-op exit 0."""
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    assert run_replay_pending(pending_files=[pending], apply=True) == 0
    # First run cleaned up the sidecar; second run should find no files.
    assert run_replay_pending(pending_files=[pending], apply=True) == 0


# ---------------------------------------------------------------------------
# Auto-commit / auto-push (cron-friendly invocation)
# ---------------------------------------------------------------------------


import subprocess  # noqa: E402


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _init_site_repo(site: Path) -> None:
    subprocess.run(["git", "init", "-q", str(site)], check=True)
    _git(site, "config", "user.email", "test@example.com")
    _git(site, "config", "user.name", "Test")
    subprocess.run(["git", "-C", str(site), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(site), "commit", "-qm", "seed"], check=True)


def test_replay_pending_auto_commit_creates_commit(tmp_path):
    site = _make_site(tmp_path, "sample-trip")
    _init_site_repo(site)
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")

    rc = run_replay_pending(pending_files=[pending], apply=True, auto_commit=True)
    assert rc == 0
    log = _git(site, "log", "-1", "--pretty=%B")
    assert "publish-journal flush" in log
    assert "content/journeys/sample-trip.md" in log


def test_replay_pending_auto_commit_refuses_dirty_wt(tmp_path, caplog):
    import logging as _logging

    site = _make_site(tmp_path, "sample-trip")
    _init_site_repo(site)
    # Dirty the repo BEFORE the sweep runs.
    journey = site / "content" / "journeys" / "sample-trip.md"
    journey.write_text(journey.read_text() + "\n# dirty\n")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")

    with caplog.at_level(_logging.WARNING, logger="gomoveshift.cli.handlers.publish_journal"):
        rc = run_replay_pending(pending_files=[pending], apply=True, auto_commit=True)
    assert rc == 3
    # The failure message flows through
    # ``logger.warning`` instead of ``print(file=err)``; assert via caplog.
    assert any("tracked working tree is dirty" in rec.getMessage() for rec in caplog.records)
    # Sidecar must survive so the entry is flushable on the next clean run.
    remaining = load_pending(pending)
    assert [e.youtube_id for e in remaining] == ["REPLAY_VID"]


def test_replay_pending_auto_push_without_auto_commit_exits_2(tmp_path, capsys):
    site = _make_site(tmp_path, "sample-trip")
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    rc = run_replay_pending(pending_files=[pending], apply=True, auto_push=True)
    assert rc == 2
    assert "--auto-push requires --auto-commit" in capsys.readouterr().err


def test_replay_pending_auto_push_against_bare_remote_succeeds(tmp_path):
    """End-to-end happy path: pending → commit → push to local bare remote."""
    site = _make_site(tmp_path, "sample-trip")
    _init_site_repo(site)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _git(site, "remote", "add", "origin", str(bare))
    _git(site, "push", "-u", "origin", "master")

    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")
    rc = run_replay_pending(pending_files=[pending], apply=True, auto_commit=True, auto_push=True)
    assert rc == 0
    remote_log = subprocess.run(
        ["git", "-C", str(bare), "log", "-1", "--pretty=%s"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert "publish-journal flush" in remote_log


def test_replay_pending_auto_commit_not_in_git_repo_logs_warning(tmp_path, caplog):
    """Journey file outside any git repo with --auto-commit logs and skips."""
    import logging as _logging

    site = _make_site(tmp_path, "sample-trip")
    # Intentionally NOT initializing a git repo for `site`.
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")

    with caplog.at_level(_logging.WARNING, logger="gomoveshift.cli.handlers.publish_journal"):
        rc = run_replay_pending(pending_files=[pending], apply=True, auto_commit=True)
    assert rc == 3
    # The failure message flows through
    # ``logger.warning`` instead of ``print(file=err)``; assert via caplog.
    assert any("not in a git working tree" in rec.getMessage() for rec in caplog.records)
    assert [e.youtube_id for e in load_pending(pending)] == ["REPLAY_VID"]


# ---------------------------------------------------------------------------
# Data-loss regressions: replay must never consume a
# sidecar entry that was not durably written, and one bad entry must not
# crash the sweep or sink the others.
# ---------------------------------------------------------------------------


def _seed_pending(tmp_path: Path, items: list[dict]) -> Path:
    pending_dir = tmp_path / "out"
    pending_dir.mkdir(exist_ok=True)
    pending_path = pending_dir / PENDING_FILENAME
    pending_path.write_text(json.dumps(items))
    return pending_path


def test_replay_pending_plan_error_keeps_entry(tmp_path):
    """A ready entry whose day stub is missing (plan error) must stay in the
    sidecar — not be silently consumed before the write that never happened."""
    site = _make_site(tmp_path, "sample-trip")
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()
    pending = _seed_pending(
        tmp_path,
        [
            {
                "youtube_id": "ORPHAN_VID",
                "publish_at": "2025-01-01T00:00:00Z",
                "title": "Orphan",
                "duration": "10:00",
                "day_date": "2099-12-31",  # no such day in the fixture journey
                "journey_path": str(journey),
                "lf_index": 1,
            }
        ],
    )
    rc = run_replay_pending(pending_files=[pending], apply=True)
    assert rc == 0  # no auto_commit → failures are informational
    assert journey.read_text() == before, "errored entry must not touch journey"
    assert [e.youtube_id for e in load_pending(pending)] == ["ORPHAN_VID"]


def test_replay_pending_missing_journey_keeps_entry(tmp_path):
    """If the target journey file is gone, the entry stays pending (recoverable
    once restored) rather than being dropped."""
    site = _make_site(tmp_path, "sample-trip")
    missing = site / "content" / "journeys" / "does-not-exist.md"
    pending = _seed_pending(
        tmp_path,
        [
            {
                "youtube_id": "GHOST_VID",
                "publish_at": "2025-01-01T00:00:00Z",
                "title": "Ghost",
                "duration": "10:00",
                "day_date": "2025-09-03",
                "journey_path": str(missing),
                "lf_index": 1,
            }
        ],
    )
    rc = run_replay_pending(pending_files=[pending], apply=True)
    assert rc == 0
    assert [e.youtube_id for e in load_pending(pending)] == ["GHOST_VID"]


def test_replay_pending_malformed_duration_does_not_sink_other_entries(tmp_path):
    """A malformed duration must not crash the whole sweep; the bad entry stays
    pending and the journey is untouched."""
    site = _make_site(tmp_path, "sample-trip")
    journey = site / "content" / "journeys" / "sample-trip.md"
    pending = _seed_pending(
        tmp_path,
        [
            {
                "youtube_id": "BAD_DUR",
                "publish_at": "2025-01-01T00:00:00Z",
                "title": "Bad duration",
                "duration": "not-a-duration",
                "day_date": "2025-09-03",
                "journey_path": str(journey),
                "lf_index": 1,
            }
        ],
    )
    rc = run_replay_pending(pending_files=[pending], apply=True)
    assert rc == 0
    assert "BAD_DUR" not in journey.read_text()
    assert [e.youtube_id for e in load_pending(pending)] == ["BAD_DUR"]


def test_publish_journal_persists_deferrals_despite_errors(tmp_path):
    """One LF erroring (unmatched date) must not sink a valid future-dated LF's
    deferral — it lands in the sidecar so replay still flushes it later."""
    site = _make_site(tmp_path, "sample-trip")
    journey = site / "content" / "journeys" / "sample-trip.md"
    before = journey.read_text()
    state = make_state(
        youtube_uploads={
            "1": {
                "video_id": "FUT_OK",
                "title": "Future ok",
                "publish_at": "2099-01-01T00:00:00Z",
            },
            "2": {
                "video_id": "BAD_DATE",
                "title": "Bad date",
                "publish_at": "2025-01-01T00:00:00Z",
            },
        },
        per_lf_recording_date={"1": "2025-09-03", "2": "2099-12-31"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4", "2": "/tmp/lf_02.mp4"},
    )
    state_path = _write_state(tmp_path, state)
    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 1  # the unmatched-date LF is a real error
    assert journey.read_text() == before, "journey YAML untouched on error"
    entries = load_pending(state_path.parent / PENDING_FILENAME)
    assert [e.youtube_id for e in entries] == ["FUT_OK"]


# ---------------------------------------------------------------------------
# Atomic journey writes — a torn write must never truncate a published page
# ---------------------------------------------------------------------------

import io  # noqa: E402

_COMMENT_LINE = "# hand-authored note the round-trip must keep"


def _comment_the_journey(site: Path) -> Path:
    """Add an operator comment to the fixture so we can prove the ruamel
    round-trip is untouched by the write path change."""
    journey = site / "content" / "journeys" / "sample-trip.md"
    journey.write_text(journey.read_text().replace("days:", f"{_COMMENT_LINE}\ndays:", 1))
    return journey


def _crash_on_file_writes(monkeypatch):
    """Kill every write-mode file handle on its first ``write`` call.

    Faithful to a real interruption: a bare ``Path.write_text`` truncates the
    target the moment it opens it, so dying before the bytes land leaves an
    empty page on the live site. An atomic write stages the bytes elsewhere
    and only swaps them in at the end, so the published page is untouched.
    """
    real_open = io.open

    def guarded_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if "w" in mode:

            def boom(*_a, **_k):
                raise RuntimeError("simulated crash mid-write")

            handle.write = boom
        return handle

    monkeypatch.setattr(io, "open", guarded_open)


def test_apply_journey_write_survives_crash_mid_write(tmp_path, monkeypatch):
    """A crash during the merge write must leave the published page intact.

    A missing merge is replayable; a truncated page is a corrupt artifact that
    blocks the site deploy (which hard-fails on a ``video_count`` mismatch).
    """
    site = _make_site(tmp_path, "sample-trip")
    journey = _comment_the_journey(site)
    before = journey.read_text()
    state = make_state(
        youtube_uploads={"1": {"video_id": "CRASH1", "title": "Crash insert"}},
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    state_path = _write_state(tmp_path, state)

    _crash_on_file_writes(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash mid-write"):
        run_publish_journal(
            state_path=state_path,
            site_path=site,
            journey_slug="sample-trip",
            apply=True,
            allow_day_create=False,
            ffprobe_duration=_fake_ffprobe,
        )
    monkeypatch.undo()

    assert journey.read_text() == before, "not truncated, not emptied"
    assert [p.name for p in (site / "content" / "journeys").iterdir()] == ["sample-trip.md"]


def test_apply_journey_write_lands_full_content_and_keeps_comments(tmp_path):
    """The successful merge path still writes everything and round-trips comments."""
    site = _make_site(tmp_path, "sample-trip")
    journey = _comment_the_journey(site)
    state = make_state(
        youtube_uploads={"1": {"video_id": "FULL1", "title": "Full insert"}},
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    rc = run_publish_journal(
        state_path=_write_state(tmp_path, state),
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )
    assert rc == 0
    text = journey.read_text()
    assert _COMMENT_LINE in text
    assert "FULL1" in text and "EXISTING_VID_2" in text
    assert text.startswith("---\n")
    assert [p.name for p in (site / "content" / "journeys").iterdir()] == ["sample-trip.md"]


def test_replay_journey_write_survives_crash_mid_write(tmp_path, monkeypatch):
    """Same guarantee on the cron replay path, which writes the same page."""
    site = _make_site(tmp_path, "sample-trip")
    journey = _comment_the_journey(site)
    before = journey.read_text()
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")

    _crash_on_file_writes(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash mid-write"):
        run_replay_pending(pending_files=[pending], apply=True)
    monkeypatch.undo()

    assert journey.read_text() == before, "not truncated, not emptied"
    assert [p.name for p in (site / "content" / "journeys").iterdir()] == ["sample-trip.md"]
    assert load_pending(pending), "unflushed entry stays pending"


def test_replay_journey_write_lands_full_content_and_keeps_comments(tmp_path):
    """The successful replay path still writes everything and keeps comments."""
    site = _make_site(tmp_path, "sample-trip")
    journey = _comment_the_journey(site)
    pending = _seed_pending_for_replay(tmp_path, site, "2025-01-01T00:00:00Z")

    assert run_replay_pending(pending_files=[pending], apply=True) == 0
    text = journey.read_text()
    assert _COMMENT_LINE in text
    assert "REPLAY_VID" in text and "EXISTING_VID_2" in text
    assert text.startswith("---\n")
    assert [p.name for p in (site / "content" / "journeys").iterdir()] == ["sample-trip.md"]
