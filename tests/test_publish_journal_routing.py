"""An embed never outlives the route that reproduces it.

``sync-site --journeys`` re-derives every journey page from
``data/video_routing.yaml`` (``journey_reconcile.load_routing``). A video
embedded into ``content/journeys/<slug>.md`` with no route is therefore
invisible to that backstop: every later regeneration drops it, and the
journal cron's superset guard refuses the push forever, because nothing can
re-derive an embed whose route was never written. (One unrouted
``--replay-pending`` embed was enough to deadlock the hourly sync.)

So both writers in ``publish_journal`` must record routing for every insert
they actually write — and only for those: not for skips, not for deferrals,
and not on a dry run.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import yaml

from gomoveshift.cli.handlers.publish_journal import (
    run_publish_journal,
    run_replay_pending,
)
from gomoveshift.pipeline.pending_journal import PENDING_FILENAME
from gomoveshift.pipeline.site_sync import journey_reconcile

from _state_helpers import make_state

FIXTURE = Path(__file__).parent / "fixtures" / "journals" / "sample_trip.md"


def _fake_ffprobe(_path):
    return 600.0


def _make_site(tmp_path: Path, journey_slug: str = "sample-trip") -> Path:
    site = tmp_path / "site"
    (site / "content" / "journeys").mkdir(parents=True)
    shutil.copy(FIXTURE, site / "content" / "journeys" / f"{journey_slug}.md")
    return site


def _routing_path(site: Path) -> Path:
    return site / "data" / journey_reconcile.ROUTING_FILENAME


def _routes(site: Path) -> dict:
    return (yaml.safe_load(_routing_path(site).read_text()) or {}).get("routes") or {}


def _write_state(tmp_path: Path, video_id: str) -> Path:
    state = make_state(
        youtube_uploads={"1": {"video_id": video_id, "title": "Routed insert"}},
        per_lf_recording_date={"1": "2025-09-03"},
        regular_lf_paths={"1": "/tmp/lf_01.mp4"},
    )
    path = tmp_path / "pipeline_state.json"
    path.write_text(json.dumps(state.to_dict() if hasattr(state, "to_dict") else state))
    return path


def _seed_pending(tmp_path: Path, site: Path, youtube_id: str = "REPLAY_VID") -> Path:
    pending_dir = tmp_path / "out"
    pending_dir.mkdir()
    pending_path = pending_dir / PENDING_FILENAME
    pending_path.write_text(
        json.dumps(
            [
                {
                    "youtube_id": youtube_id,
                    "publish_at": "2025-01-01T00:00:00Z",
                    "title": "Replay Test",
                    "duration": "10:00",
                    "day_date": "2025-09-03",
                    "journey_path": str(site / "content" / "journeys" / "sample-trip.md"),
                    "lf_index": 1,
                }
            ]
        )
    )
    return pending_path


def test_replay_pending_records_a_route_for_every_flushed_entry(tmp_path):
    """The replay path: ``--replay-pending --apply`` embedded an id
    and recorded no route, so the reconciler could never reproduce the embed."""
    site = _make_site(tmp_path)
    pending = _seed_pending(tmp_path, site)

    rc = run_replay_pending(pending_files=[pending], apply=True)

    assert rc == 0
    assert "REPLAY_VID" in (site / "content" / "journeys" / "sample-trip.md").read_text()
    assert _routes(site)["REPLAY_VID"] == {"journey": "sample-trip", "date": "2025-09-03"}


def test_publish_journal_records_a_route_for_every_insert(tmp_path):
    site = _make_site(tmp_path)
    state_path = _write_state(tmp_path, "PUBLISHED1")

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )

    assert rc == 0
    assert _routes(site)["PUBLISHED1"] == {"journey": "sample-trip", "date": "2025-09-03"}


def test_dry_runs_write_no_routing_file_at_all(tmp_path):
    """``apply=False`` writes nothing anywhere — a route for an embed that
    was never made would make the reconciler embed it on its own."""
    site = _make_site(tmp_path)
    state_path = _write_state(tmp_path, "DRY1")
    pending = _seed_pending(tmp_path, site, youtube_id="DRY2")

    assert (
        run_publish_journal(
            state_path=state_path,
            site_path=site,
            journey_slug="sample-trip",
            apply=False,
            allow_day_create=False,
            ffprobe_duration=_fake_ffprobe,
        )
        == 0
    )
    assert run_replay_pending(pending_files=[pending], apply=False) == 0

    assert not _routing_path(site).exists()


def test_recording_a_route_preserves_unrelated_existing_routes(tmp_path):
    """``merge_routing`` upserts; the back-catalog routes hand-authored in the
    site repo must survive a publish that adds one more."""
    site = _make_site(tmp_path)
    (site / "data").mkdir(parents=True, exist_ok=True)
    _routing_path(site).write_text(
        yaml.safe_dump({"routes": {"OLD_BACKCAT": {"journey": "other-trip", "date": "2024-01-02"}}})
    )
    state_path = _write_state(tmp_path, "PUBLISHED1")

    rc = run_publish_journal(
        state_path=state_path,
        site_path=site,
        journey_slug="sample-trip",
        apply=True,
        allow_day_create=False,
        ffprobe_duration=_fake_ffprobe,
    )

    assert rc == 0
    routes = _routes(site)
    assert routes["OLD_BACKCAT"] == {"journey": "other-trip", "date": "2024-01-02"}
    assert routes["PUBLISHED1"] == {"journey": "sample-trip", "date": "2025-09-03"}


def test_a_failing_routing_write_never_fails_a_journey_that_was_written(tmp_path, monkeypatch):
    """Routing is a side-quest (the ``drive.py`` precedent): a read-only site
    checkout must not turn a successful, already-durable journey write into a
    reported failure."""
    site = _make_site(tmp_path)
    pending = _seed_pending(tmp_path, site)

    def _boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(journey_reconcile, "merge_routing", _boom)

    rc = run_replay_pending(pending_files=[pending], apply=True)

    assert rc == 0
    assert "REPLAY_VID" in (site / "content" / "journeys" / "sample-trip.md").read_text()
    assert not _routing_path(site).exists()


def _init_site_repo(site: Path) -> None:
    subprocess.run(["git", "init", "-q", str(site)], check=True)
    for k, v in (("user.email", "test@example.com"), ("user.name", "Test")):
        subprocess.run(["git", "-C", str(site), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(site), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(site), "commit", "-qm", "seed"], check=True)


def test_auto_commit_commits_the_route_with_the_embed_it_reproduces(tmp_path):
    """``video_routing.yaml`` is tracked in the site repo. Committing the
    journey but not the route it needs would leave the tree dirty, and the
    NEXT sweep's ``is_tracked_dirty`` pre-check skips every journey in a dirty
    repo — a fresh flavour of the same stall."""
    site = _make_site(tmp_path)
    _init_site_repo(site)
    pending = _seed_pending(tmp_path, site)

    rc = run_replay_pending(pending_files=[pending], apply=True, auto_commit=True)

    assert rc == 0
    assert _routes(site)["REPLAY_VID"] == {"journey": "sample-trip", "date": "2025-09-03"}
    assert (
        "data/video_routing.yaml"
        in subprocess.run(
            ["git", "-C", str(site), "log", "-1", "--name-only", "--pretty="],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    assert (
        subprocess.run(["git", "-C", str(site), "diff", "--quiet"], check=False).returncode == 0
    ), "a tracked write left uncommitted stalls the next sweep"
