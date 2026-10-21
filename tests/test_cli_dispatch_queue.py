"""Tests for ``gomoveshift dispatch-queue`` (cli/handlers/dispatch_queue.py).

This command calls live publishing APIs, so the tests that matter most are
the ones proving it does NOT: dry-run is the default, and an unrouted
platform never has its adapter constructed (which would decrypt credentials
through SOPS).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.publishing.dispatch import DrainResult
from unittest import mock

import pytest

from gomoveshift.cli.handlers.dispatch_queue import (
    SidecarRecordLoader,
    _media_path,
    _read_instagram_media,
    _read_media,
    _read_tiktok_media,
    cmd_dispatch_queue,
)
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.publishing.models import (
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    YouTubePayload,
)
from gomoveshift.publishing.dispatch import TransientMediaError
from gomoveshift.publishing.plan_io import PublishPlanFormatError
from gomoveshift.publishing.tiktok_pack import PackReconcileResult, PackSweepResult
from gomoveshift.publishing.queue import cancel_delivery, enqueue, open_queue_db, transition

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _own_console_data_dir(tmp_path, monkeypatch):
    # Keeps ``dispatch-queue --apply``'s notify seen file out of the live console dir.
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "_console"))


def _record(short_index=3, selected=("youtube",)):
    return ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=short_index,
        media=MediaFingerprint("/v/x.mp4", 1, 2.0, 1080, 1920, "h264", "a" * 64),
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("a",),
        parent_lf_video_id="vid",
        parent_lf_publish_at=SCHEDULED,
        payloads=PlatformPayloads(youtube=YouTubePayload("t", "d")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=tuple(selected)),
    )


def _sidecar(tmp_path, *records, project="proj"):
    path = tmp_path / f"{project}_publish_plan.json"
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=tuple(records))), encoding="utf-8")
    return path


def _config(tmp_path, **routes):
    drive = SimpleNamespace(
        publishing_route_youtube=routes.get("youtube", "metricool"),
        publishing_route_instagram=routes.get("instagram", "metricool"),
        publishing_route_facebook=routes.get("facebook", "metricool"),
        publishing_route_tiktok=routes.get("tiktok", "metricool"),
        publishing_tiktok_audit_approved=False,
    )
    return SimpleNamespace(drive=drive, seo=SimpleNamespace(storage_path=str(tmp_path)))


def _args(**over):
    return SimpleNamespace(**{"apply": False, "sidecar_dir": None, **over})


@pytest.fixture(autouse=True)
def _pack_dir_stays_in_tmp(tmp_path, monkeypatch):
    """The TikTok pack defaults to a folder under the data home.

    No test in this file may ever reach it.
    """
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))


# --- the safety properties --------------------------------------------------


def test_dry_run_is_the_default_and_drains_nothing(tmp_path, capsys):
    """Without --apply the command must not drain, so no adapter is built
    and no platform API is reachable."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())

    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers") as build,
    ):
        rc = cmd_dispatch_queue(_args(), _config(tmp_path, youtube="first_party"))

    assert rc == 0
    drain.assert_not_called()
    build.assert_not_called()
    assert "DRY RUN" in capsys.readouterr().out


def test_apply_with_no_first_party_route_builds_no_adapter(tmp_path, capsys):
    """Every route still on metricool means the CSV path is live and there
    is legitimately nothing to drain. Constructing adapters anyway would
    decrypt credentials through SOPS for no reason."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()

    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers") as build,
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path))

    assert rc == 0
    drain.assert_not_called()
    build.assert_not_called()
    assert "nothing to drain" in capsys.readouterr().out


def test_apply_drains_only_routed_platforms(tmp_path):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers",
            return_value={},
        ) as build,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
    ):
        drain.return_value = DrainResult(
            attempted=0,
            succeeded=0,
            retried=0,
            failed_permanent=0,
            skipped_leased=0,
            skipped_no_handler=0,
            throttled=0,
            errors=[],
        )
        rc = cmd_dispatch_queue(
            _args(apply=True), _config(tmp_path, youtube="first_party", tiktok="metricool")
        )

    assert rc == 0
    routed = build.call_args[0][3]
    assert routed == ["youtube"], "an unrouted platform must not get an adapter"


def test_summary_counts_deliveries_waiting_on_a_human(tmp_path, capsys):
    """The scheduled drain logs the LAST line of this command's output, so the
    count of deliveries parked in ``awaiting_manual_publish`` has to ride the
    summary line itself — a separate line printed after it would displace the
    summary from the log, and one printed before it would never be logged at
    all. TikTok's inbox route parks a row that no drain can ever advance; a
    pass that reports only what it attempted says nothing about the shorts
    sitting unpublished."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    parked = enqueue(conn, "proj", 3, "tiktok", "HASH1")
    transition(conn, parked, "queued", "awaiting_manual_publish")
    conn.close()
    _sidecar(tmp_path, _record())

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers",
            return_value={},
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
    ):
        drain.return_value = DrainResult(
            attempted=0,
            succeeded=0,
            retried=0,
            failed_permanent=0,
            skipped_leased=0,
            skipped_no_handler=0,
            throttled=0,
            errors=[],
        )
        rc = cmd_dispatch_queue(
            _args(apply=True), _config(tmp_path, youtube="first_party", tiktok="first_party")
        )

    assert rc == 0
    summary = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("attempted=")]
    assert len(summary) == 1
    assert "awaiting_manual_publish=1" in summary[0]


def test_apply_resolves_deliveries_via_row_sidecar_path_not_sidecar_dir(tmp_path):
    """A real run writes its sidecar into the run's own output dir, not a
    flat directory keyed off ``seo.storage_path``. The drain must resolve
    each row's record from the path recorded at enqueue time, not from
    ``--sidecar-dir`` (here left unset / pointed at an unrelated dir)."""
    real_dir = tmp_path / "runs" / "proj_output"
    real_dir.mkdir(parents=True)
    real_sidecar = _sidecar(real_dir, _record(short_index=3))

    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1", sidecar_path=str(real_sidecar))
    conn.close()

    captured = {}

    def fake_build(conn, loader, config, routed, *, worker_id="adapter"):
        captured["record"] = loader("proj", 3)
        return {}

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers",
            side_effect=fake_build,
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
    ):
        drain.return_value = DrainResult(
            attempted=1,
            succeeded=1,
            retried=0,
            failed_permanent=0,
            skipped_leased=0,
            skipped_no_handler=0,
            throttled=0,
            errors=[],
        )
        rc = cmd_dispatch_queue(
            _args(apply=True, sidecar_dir=str(tmp_path / "unrelated")),
            _config(tmp_path, youtube="first_party"),
        )

    assert rc == 0
    assert captured["record"].short_index == 3
    assert captured["record"].reviewed_plan_hash == "HASH1"


def test_malformed_sidecar_exits_nonzero_naming_the_file(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()
    (tmp_path / "proj_publish_plan.json").write_text("{not json", encoding="utf-8")

    def boom(*a, **k):
        raise PublishPlanFormatError("publish-plan sidecar bad: not valid JSON")

    with mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers", boom):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, youtube="first_party"))
    assert rc == 1


# --- SidecarRecordLoader ----------------------------------------------------


def test_loader_returns_the_record_for_a_queue_row(tmp_path):
    _sidecar(tmp_path, _record(short_index=3))
    loader = SidecarRecordLoader(tmp_path)
    record = loader("proj", 3)
    assert record.short_index == 3
    assert record.reviewed_plan_hash == "HASH1"


def test_loader_prefers_the_row_sidecar_path_over_the_configured_dir(tmp_path):
    """A delivery enqueued from run A's output dir must resolve from THAT
    file even when the loader was constructed against an unrelated
    directory (a stale/default --sidecar-dir, or none at all)."""
    real_dir = tmp_path / "real_output"
    real_dir.mkdir()
    real_sidecar = _sidecar(real_dir, _record(short_index=3))

    unrelated_dir = tmp_path / "unrelated"
    unrelated_dir.mkdir()

    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1", sidecar_path=str(real_sidecar))

    loader = SidecarRecordLoader(unrelated_dir, conn=conn)
    record = loader("proj", 3)
    assert record.short_index == 3
    assert record.reviewed_plan_hash == "HASH1"
    conn.close()


def test_loader_falls_back_to_the_directory_convention_when_sidecar_path_is_null(tmp_path):
    """A row enqueued before this column existed carries no sidecar_path --
    the loader must still resolve it from the configured directory."""
    _sidecar(tmp_path, _record(short_index=3))
    conn = open_queue_db(tmp_path / "seo")
    enqueue(conn, "proj", 3, "youtube", "HASH1")  # no sidecar_path

    loader = SidecarRecordLoader(tmp_path, conn=conn)
    record = loader("proj", 3)
    assert record.short_index == 3
    conn.close()


def test_loader_caches_the_sidecar_per_project(tmp_path):
    """One drain pass covers many shorts of the same project; re-reading and
    re-parsing the file per delivery would be pure waste."""
    path = _sidecar(tmp_path, _record(short_index=1), _record(short_index=2))
    loader = SidecarRecordLoader(tmp_path)
    assert loader("proj", 1).short_index == 1
    path.unlink()  # gone from disk — a cached loader must still answer
    assert loader("proj", 2).short_index == 2


def test_loader_raises_when_queue_and_sidecar_disagree(tmp_path):
    _sidecar(tmp_path, _record(short_index=3))
    loader = SidecarRecordLoader(tmp_path)
    with pytest.raises(LookupError) as exc:
        loader("proj", 99)
    assert "99" in str(exc.value)
    assert "disagree" in str(exc.value)


def test_loader_surfaces_a_missing_sidecar_as_transient_not_a_format_error(tmp_path):
    """Contract split, deliberate: an ABSENT sidecar is a filesystem
    condition (the run's drive is offline / the dir moved) and must be
    retried; a MALFORMED one is a real defect needing a human and stays a
    ``PublishPlanFormatError`` — see the malformed case above."""
    loader = SidecarRecordLoader(tmp_path)
    with pytest.raises(TransientMediaError):
        loader("nosuchproject", 1)


def test_loader_still_surfaces_a_MALFORMED_sidecar_as_a_format_error(tmp_path):
    """The other half of the split, exercised through the real loader rather
    than a mock: a sidecar that EXISTS but is unparseable is a defect, not a
    filesystem blip, and must not be retried forever."""
    (tmp_path / "proj_publish_plan.json").write_text("{not json", encoding="utf-8")
    loader = SidecarRecordLoader(tmp_path)
    with pytest.raises(PublishPlanFormatError):
        loader("proj", 3)


def test_loader_reads_the_real_serializer_output(tmp_path):
    """Round-trip through the actual producer, not a hand-built dict."""
    path = _sidecar(tmp_path, _record(short_index=7, selected=("youtube", "tiktok")))
    raw = json.loads(path.read_text())
    assert raw["shorts"][0]["short_index"] == 7
    loader = SidecarRecordLoader(tmp_path)
    assert loader("proj", 7).routes.selected == ("youtube", "tiktok")


def test_dry_run_reports_shadow_rows_as_held_back(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "youtube", "h")
    enqueue(conn, "proj", 2, "youtube", "h", shadow=True)
    conn.close()

    assert cmd_dispatch_queue(_args(), _config(tmp_path)) == 0
    out = capsys.readouterr().out
    # Assert the REPORT line, not a bare "shadow" substring: pytest's
    # tmp_path embeds the test name, so the printed queue path contains the
    # word "shadow" and a loose match passes even with no report at all.
    assert "shadow (never published): 1" in out
    assert "non-terminal deliveries: youtube=1" in out


# --- drain-time media verification -------------------------------------


def _real_media(tmp_path, data=b"video-bytes"):
    """A real file on disk plus a fingerprint that matches it."""
    path = tmp_path / "short.mp4"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), len(data)


def _record_with_media(tmp_path, *, data=b"video-bytes", sha=None, size=None, codec="h264"):
    path, real_sha, real_size = _real_media(tmp_path, data)
    record = _record()
    return dataclasses.replace(
        record,
        media=MediaFingerprint(
            str(path),
            size if size is not None else real_size,
            2.0,
            1080,
            1920,
            codec,
            sha if sha is not None else real_sha,
        ),
    )


def test_tiktok_uploads_an_h264_rendition_of_an_hevc_master(tmp_path, monkeypatch):
    """The HEVC master is what the app cannot open (see
    publishing/tiktok_rendition.py). Verification still runs against the
    MASTER — the rendition is derived and carries no fingerprint of its own —
    so a swapped or missing short is still caught before any transcode."""
    record = _record_with_media(tmp_path, data=b"hevc-master", codec="hevc")
    seen = {}

    def fake_rendition(source, *, codec):
        seen["source"] = Path(source)
        seen["codec"] = codec
        out = Path(source).with_suffix(".h264.mp4")
        out.write_bytes(b"h264-rendition")
        return out

    monkeypatch.setattr(
        "gomoveshift.cli.handlers.dispatch_queue.ensure_h264_rendition", fake_rendition
    )

    assert _read_tiktok_media(record) == b"h264-rendition"
    assert seen["source"] == Path(record.media.path)
    assert seen["codec"] == "hevc"


def test_tiktok_media_verification_runs_before_any_transcode(tmp_path, monkeypatch):
    record = _record_with_media(tmp_path, codec="hevc", sha="b" * 64)
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.dispatch_queue.ensure_h264_rendition",
        lambda *a, **k: pytest.fail("transcoded a short that failed its fingerprint"),
    )

    with pytest.raises(TransientMediaError):
        _read_tiktok_media(record)


def test_read_media_raises_transient_when_the_file_is_gone(tmp_path):
    """A delivery outliving its media must be RETRYABLE, never terminal: an
    unmounted drive is operator-fixable, and burning the row would need a
    manual re-enqueue."""
    record = _record_with_media(tmp_path)
    Path(record.media.path).unlink()

    with pytest.raises(TransientMediaError):
        _read_media(record)


def test_media_path_raises_transient_on_fingerprint_mismatch(tmp_path):
    """A swapped or truncated file is detectable from the record's own
    fingerprint — catch it before it is uploaded, not after."""
    record = _record_with_media(tmp_path, sha="b" * 64)

    with pytest.raises(TransientMediaError):
        _media_path(record)


def test_read_media_returns_bytes_when_the_file_still_matches(tmp_path):
    record = _record_with_media(tmp_path, data=b"still-here")
    assert _read_media(record) == b"still-here"


def test_media_path_returns_the_path_when_the_file_still_matches(tmp_path):
    record = _record_with_media(tmp_path)
    assert _media_path(record) == Path(record.media.path)


def test_missing_sidecar_is_transient_not_a_permanent_failure(tmp_path):
    """After the sidecar_path change the sidecar is ALSO on the run's drive,
    so it has the same offline exposure as the media."""
    conn = open_queue_db(tmp_path)
    loader = SidecarRecordLoader(tmp_path / "gone", conn=conn)
    with pytest.raises(TransientMediaError):
        loader("proj", 3)
    conn.close()


# --- the queue must be the SAME database enqueue wrote to -------------------
#
# ``seo.storage_path`` defaults to the cwd-relative ``./data/seo`` and the
# project ``config.yaml`` that overrides it with an absolute path is found
# only as ``Path("config.yaml")`` -- also cwd-relative (``Config.load``).
# Run from anywhere but the service directory, the drain therefore CREATED an
# empty SQLite queue beside its cwd and reported "non-terminal deliveries:
# none": the cron (cwd ``$HOME``) would have published nothing, forever,
# with a zero exit and a cheerful log line. A relative storage_path is now a
# loud refusal instead of a silent wrong database.


def test_relative_storage_path_is_refused_before_any_db_is_opened(tmp_path, capsys):
    config = _config(tmp_path)
    config.seo.storage_path = "./data/seo"

    with mock.patch("gomoveshift.cli.handlers.dispatch_queue.open_queue_db") as open_db:
        rc = cmd_dispatch_queue(_args(apply=True), config)

    assert rc == 1
    open_db.assert_not_called()
    out = capsys.readouterr().out
    assert "./data/seo" in out
    assert "absolute" in out


def test_absolute_storage_path_still_runs(tmp_path, capsys):
    """The guard must not fire on the configured deployment."""
    conn = open_queue_db(tmp_path)
    conn.close()
    rc = cmd_dispatch_queue(_args(), _config(tmp_path))
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_drain_and_adapters_share_one_worker_identity(tmp_path):
    """``drain_queue`` takes the lease; every adapter re-takes it before its
    irreversible call. Both must use the SAME owner.

    If drain_queue minted ``host:pid:uuid`` while every adapter defaulted to
    the literal ``"adapter"``, the adapter's acquire could never succeed
    against its own pass's live lease and each delivery would raise
    RuntimeError("... leased by another worker"), failing every due delivery
    even with no second worker running.
    """
    sidecar_dir = tmp_path / "runs" / "proj_output"
    sidecar_dir.mkdir(parents=True)
    sidecar = _sidecar(sidecar_dir, _record(short_index=3))
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1", sidecar_path=str(sidecar))
    conn.close()

    seen = {}

    def fake_build(conn, loader, config, routed, *, worker_id="adapter"):
        seen["build"] = worker_id
        return {}

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers",
            side_effect=fake_build,
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
    ):
        drain.return_value = DrainResult(
            attempted=0,
            succeeded=0,
            retried=0,
            failed_permanent=0,
            skipped_leased=0,
            skipped_no_handler=0,
            throttled=0,
            errors=[],
        )
        rc = cmd_dispatch_queue(
            _args(apply=True, sidecar_dir=str(sidecar_dir)),
            _config(tmp_path, youtube="first_party"),
        )

    assert rc == 0
    assert seen["build"] != "adapter", "the adapters must not keep the default identity"
    assert drain.call_args.kwargs.get("owner") == seen["build"], (
        "drain_queue and the adapters must share one owner"
    )


# --- Reconciling drafts published from the app ------------------------------


def test_dry_run_never_reconciles(tmp_path, capsys):
    """Reconcile polls a live TikTok endpoint and writes terminal statuses.
    Without --apply the command must not reach it, same as the drain."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())

    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.reconcile_awaiting_manual") as rec,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.build_tiktok_reconciler") as build,
    ):
        rc = cmd_dispatch_queue(_args(), _config(tmp_path, tiktok="first_party"))

    assert rc == 0
    rec.assert_not_called()
    build.assert_not_called()


def test_reconcile_runs_before_the_drain(tmp_path):
    """Order is the point: a row the human published is closed BEFORE the
    drain reads the queue, so the pass reports it as published today rather
    than as an outstanding chore."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())
    calls = []

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.build_tiktok_reconciler",
            return_value=(lambda _p: {}, lambda p: f"https://x/{p}"),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.reconcile_awaiting_manual",
            side_effect=lambda *a, **k: (
                calls.append("reconcile")
                or SimpleNamespace(polled=0, published=0, failed=0, errors=[])
            ),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.drain_queue",
            side_effect=lambda *a, **k: calls.append("drain") or DrainResult(),
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, tiktok="first_party"))

    assert rc == 0
    assert calls == ["reconcile", "drain"]


def test_a_reconciler_that_cannot_be_built_does_not_stop_the_drain(tmp_path, capsys):
    """Credentials that will not decrypt must cost the parked rows, not the
    shorts that are due."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.build_tiktok_reconciler",
            side_effect=FileNotFoundError("no creds"),
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, tiktok="first_party"))

    assert rc == 0
    drain.assert_called_once()
    assert "reconcile: skipped" in capsys.readouterr().out


def test_the_reconciler_polls_once_and_builds_a_canonical_permalink():
    """The shipped poll default is 10 attempts with backoff and it RAISES on
    a draft still in the inbox -- the normal case. And the permalink must not
    be video/list's share_url, which carries the client key in utm_source."""
    from gomoveshift.cli.handlers.dispatch_queue import build_tiktok_reconciler

    seen = {}

    def fake_poll(token, publish_id, *, max_attempts=10, **kw):
        seen["max_attempts"] = max_attempts
        return {"status": "SEND_TO_USER_INBOX"}

    config = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        photos=SimpleNamespace(site_root="/site-a", resolve_site_root=lambda: Path("/site-a")),
    )

    with (
        mock.patch(
            "gomoveshift.publishing.tiktok_auth.get_valid_tiktok_credentials",
            return_value={"access_token": "tok"},
        ),
        mock.patch(
            "gomoveshift.publishing.tiktok_adapter.query_creator_info",
            return_value={"creator_username": "example_account"},
        ),
        mock.patch("gomoveshift.publishing.tiktok_adapter.poll_publish_status", fake_poll),
    ):
        poll, permalink_for = build_tiktok_reconciler(config, ["tiktok"])
        poll("v_inbox_file~v2.1")

    assert seen["max_attempts"] == 1
    url = permalink_for("7000000000001")
    assert url == "https://www.tiktok.com/@example_account/video/7000000000001"
    assert "utm_" not in url


def test_an_unrouted_tiktok_builds_no_reconciler():
    from gomoveshift.cli.handlers.dispatch_queue import build_tiktok_reconciler

    assert build_tiktok_reconciler(object(), ["youtube"]) is None


def test_reconciler_resolves_tiktok_credentials_from_the_brands_own_dir():
    """Two brands dispatching concurrently must never cross-resolve tokens
    (brand isolation) -- the reconciler has to pass ITS config's
    credentials_dir through, not fall back to whatever process-wide default
    ``tiktok_auth`` happened to import first."""
    from gomoveshift.cli.handlers.dispatch_queue import build_tiktok_reconciler

    config_a = SimpleNamespace(seo=SimpleNamespace(credentials_dir="/creds/brand-a"))
    config_b = SimpleNamespace(seo=SimpleNamespace(credentials_dir="/creds/brand-b"))

    with (
        mock.patch(
            "gomoveshift.publishing.tiktok_auth.get_valid_tiktok_credentials",
            return_value={"access_token": "tok"},
        ) as get_creds,
        mock.patch(
            "gomoveshift.publishing.tiktok_adapter.query_creator_info",
            return_value={"creator_username": "example_account"},
        ),
    ):
        build_tiktok_reconciler(config_a, ["tiktok"])
        build_tiktok_reconciler(config_b, ["tiktok"])

    assert get_creds.call_args_list[0].args[0] == Path("/creds/brand-a")
    assert get_creds.call_args_list[1].args[0] == Path("/creds/brand-b")


def test_routed_handlers_wire_a_photo_adapter_for_first_party_meta_platforms():
    """A routed Instagram/Facebook platform must also get a
    ``PhotoCarouselAdapter`` wired via ``instagram_photo``/``facebook_photo``
    so a ``format=photo``/``format=carousel`` delivery has a handler at all
    (previously only the video tuple was ever built)."""
    from gomoveshift.cli.handlers.dispatch_queue import _build_routed_handlers
    from gomoveshift.publishing.photo_adapter import PhotoCarouselAdapter

    config = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        photos=SimpleNamespace(site_root="/site-a", resolve_site_root=lambda: Path("/site-a")),
    )

    with (
        mock.patch(
            "gomoveshift.publishing.meta_auth.load_meta_credentials",
            return_value={"page_access_token": "tok", "page_id": "1", "ig_user_id": "2"},
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.build_handlers") as build,
    ):
        _build_routed_handlers(None, None, config, ["instagram", "facebook"], worker_id="w1")

    kwargs = build.call_args.kwargs
    assert kwargs["instagram_photo"] is not None
    assert kwargs["facebook_photo"] is not None
    instagram_adapter, instagram_loader = kwargs["instagram_photo"]
    facebook_adapter, facebook_loader = kwargs["facebook_photo"]
    assert isinstance(instagram_adapter, PhotoCarouselAdapter)
    assert isinstance(facebook_adapter, PhotoCarouselAdapter)
    assert callable(instagram_loader)
    assert callable(facebook_loader)
    # The config's own site_root reaches the client, not
    # photos.selector.default_site_root()'s import-time fallback.
    assert instagram_adapter._client._site_root == Path("/site-a")
    assert facebook_adapter._client._site_root == Path("/site-a")


def test_routed_handlers_skip_photo_adapters_when_no_site_root_is_configured():
    """With ``photos.site_root`` unset there is no site to derive public image
    URLs from: video routes still get their handlers, photo/carousel routes
    get none (those rows fall to ``skipped_no_handler``)."""
    from gomoveshift.cli.handlers.dispatch_queue import _build_routed_handlers
    from gomoveshift.config import Config

    config = Config()
    config.seo.credentials_dir = "/creds/brand-a"
    assert config.photos.site_root == ""

    with (
        mock.patch(
            "gomoveshift.publishing.meta_auth.load_meta_credentials",
            return_value={"page_access_token": "tok", "page_id": "1", "ig_user_id": "2"},
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.build_handlers") as build,
    ):
        _build_routed_handlers(None, None, config, ["instagram", "facebook"], worker_id="w1")

    kwargs = build.call_args.kwargs
    assert kwargs["instagram"] is not None
    assert kwargs["facebook"] is not None
    assert kwargs.get("instagram_photo") is None
    assert kwargs.get("facebook_photo") is None


def test_photo_image_url_loader_verifies_fingerprint_then_derives_url(tmp_path):
    """The loader keeps _read_photo_media's fingerprint-vs-disk
    discipline (a mismatch is TransientMediaError, retried without burning
    max_attempts) and, once verified, derives the image_url a real photo
    row would publish -- never raw bytes."""
    from gomoveshift.cli.handlers.dispatch_queue import _photo_image_url_loader
    from gomoveshift.publishing.models import MediaFingerprint

    site_root = tmp_path
    config_dir = site_root / "config" / "_default"
    config_dir.mkdir(parents=True)
    (config_dir / "hugo.toml").write_text('baseURL = "https://gomoveshift.com"\n')

    media_path = site_root / "static" / "images" / "journeys" / "example-2026-08" / "photo-1.jpg"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(b"x")
    fingerprint = MediaFingerprint(
        str(media_path), media_path.stat().st_size, 0.0, 0, 0, "jpeg", "a" * 64
    )

    loader = _photo_image_url_loader(site_root)
    with mock.patch(
        "gomoveshift.cli.handlers.dispatch_queue.verify_media",
        return_value=None,
    ):
        url = loader(fingerprint)

    assert url == "https://gomoveshift.com/images/journeys/example-2026-08/photo-1.jpg"


def test_photo_image_url_loader_raises_permanent_error_for_an_unservable_path(tmp_path):
    """A path that isn't under the site's served images layout (e.g. a
    resized social copy) is a distinct, permanent failure -- never a
    guessed URL -- and translates to PhotoPermanentError so the handler
    treats it as terminal rather than retrying forever."""
    from gomoveshift.cli.handlers.dispatch_queue import _photo_image_url_loader
    from gomoveshift.publishing.models import MediaFingerprint
    from gomoveshift.publishing.photo_adapter import PhotoPermanentError

    site_root = tmp_path / "site"
    site_root.mkdir()
    resized_path = tmp_path / "photo_resized" / "example-2026-08_photo-1.jpg"
    resized_path.parent.mkdir(parents=True)
    resized_path.write_bytes(b"x")
    fingerprint = MediaFingerprint(
        str(resized_path), resized_path.stat().st_size, 0.0, 0, 0, "jpeg", "a" * 64
    )

    loader = _photo_image_url_loader(site_root)
    with mock.patch(
        "gomoveshift.cli.handlers.dispatch_queue.verify_media",
        return_value=None,
    ):
        with pytest.raises(PhotoPermanentError, match=str(resized_path)):
            loader(fingerprint)


def test_routed_handlers_omit_the_photo_adapter_for_an_unrouted_meta_platform():
    """A platform still on the metricool default gets no photo adapter --
    mirrors the existing video-adapter behaviour for the same reason."""
    from gomoveshift.cli.handlers.dispatch_queue import _build_routed_handlers

    config = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        photos=SimpleNamespace(site_root="/site-a", resolve_site_root=lambda: Path("/site-a")),
    )

    with mock.patch("gomoveshift.cli.handlers.dispatch_queue.build_handlers") as build:
        _build_routed_handlers(None, None, config, ["youtube"], worker_id="w1")

    kwargs = build.call_args.kwargs
    assert kwargs["instagram_photo"] is None
    assert kwargs["facebook_photo"] is None


def test_routed_handlers_resolve_meta_credentials_from_the_brands_own_dir():
    """Same isolation property for the Meta (Instagram/Facebook) branch of
    _build_routed_handlers -- brand B's dispatch must load brand B's creds
    dir, not whatever DEFAULT_CREDENTIALS_DIR froze at import time."""
    from gomoveshift.cli.handlers.dispatch_queue import _build_routed_handlers

    config_a = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        photos=SimpleNamespace(site_root="/site-a", resolve_site_root=lambda: Path("/site-a")),
    )
    config_b = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-b"),
        photos=SimpleNamespace(resolve_site_root=lambda: Path("/site-b")),
    )

    with mock.patch(
        "gomoveshift.publishing.meta_auth.load_meta_credentials",
        return_value={"page_access_token": "tok", "page_id": "1", "ig_user_id": "2"},
    ) as load_meta:
        _build_routed_handlers(None, None, config_a, ["instagram"])
        _build_routed_handlers(None, None, config_b, ["instagram"])

    assert load_meta.call_args_list[0].args[0] == Path("/creds/brand-a")
    assert load_meta.call_args_list[1].args[0] == Path("/creds/brand-b")


def test_routed_handlers_resolve_tiktok_credentials_from_the_brands_own_dir(monkeypatch):
    """Same isolation property for the TikTok branch of _build_routed_handlers
    (inbox mode: pack mode loads no TikTok credentials)."""
    from gomoveshift.cli.handlers.dispatch_queue import _build_routed_handlers

    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")

    config_a = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        drive=SimpleNamespace(publishing_tiktok_audit_approved=False),
    )
    config_b = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-b"),
        drive=SimpleNamespace(publishing_tiktok_audit_approved=False),
    )

    with mock.patch(
        "gomoveshift.publishing.tiktok_auth.get_valid_tiktok_credentials",
        return_value={"access_token": "tok"},
    ) as get_creds:
        _build_routed_handlers(None, None, config_a, ["tiktok"])
        _build_routed_handlers(None, None, config_b, ["tiktok"])

    assert get_creds.call_args_list[0].args[0] == Path("/creds/brand-a")
    assert get_creds.call_args_list[1].args[0] == Path("/creds/brand-b")


def test_a_cancelled_delivery_is_not_counted_as_still_pending(tmp_path, capsys):
    """The queue summary follows ``TERMINAL_STATUSES``. A cancelled delivery
    is terminal, so reporting it under "non-terminal deliveries" would tell an
    operator work is outstanding that nothing will ever pick up."""
    conn = open_queue_db(tmp_path)
    live = enqueue(conn, "proj", 3, "youtube", "HASH1")
    gone = enqueue(conn, "proj", 4, "youtube", "HASH2")
    assert cancel_delivery(conn, gone) is True
    conn.close()
    _sidecar(tmp_path, _record())

    with mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue"):
        rc = cmd_dispatch_queue(_args(), _config(tmp_path, youtube="first_party"))

    assert rc == 0
    assert live
    assert "non-terminal deliveries: youtube=1" in capsys.readouterr().out


def _drain_summary(tmp_path, capsys, result):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())
    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
    ):
        drain.return_value = result
        cmd_dispatch_queue(_args(apply=True), _config(tmp_path, youtube="first_party"))
    return capsys.readouterr().out


def test_the_summary_reports_a_worker_that_died_mid_attempt(tmp_path, capsys):
    """A crash nobody hears about is a crash nobody acts on. drain_queue now
    charges an attempt to every delivery whose worker never came back, and
    the cron log is where an operator would see that it happened."""
    out = _drain_summary(tmp_path, capsys, DrainResult(reconciled_interrupted=2))
    assert "reconciled_interrupted=2" in out


def test_the_summary_stays_quiet_when_no_worker_died(tmp_path, capsys):
    """Printed only when non-zero, for the same reason `throttled` is: a
    field that reads `=0` on every healthy run is a field nobody reads on the
    run that matters."""
    out = _drain_summary(tmp_path, capsys, DrainResult(attempted=1, succeeded=1))
    assert "reconciled_interrupted" not in out


def test_the_summary_flags_a_blocked_pass_not_healthy_idle(tmp_path, capsys):
    """``attempted=0 succeeded=0 failed_permanent=0`` is what a genuinely
    idle pass ALSO reports. A pass that skipped due work for a missing
    handler must read differently on the one line the cron log's `tail -1`
    keeps."""
    out = _drain_summary(tmp_path, capsys, DrainResult(skipped_no_handler=6))
    assert "Blocked: publishing route unavailable" in out
    summary = [ln for ln in out.splitlines() if ln.startswith("attempted=")]
    assert summary and "Blocked: publishing route unavailable" in summary[0]


def test_the_summary_stays_quiet_when_nothing_was_skipped_for_no_handler(tmp_path, capsys):
    out = _drain_summary(tmp_path, capsys, DrainResult(attempted=1, succeeded=1))
    assert "Blocked" not in out


def test_the_summary_flags_a_pass_held_for_unverified_privacy(tmp_path, capsys):
    """A hold is invisible on `attempted=0 succeeded=0` exactly like a missing
    route is. The cron log keeps `tail -1`, so it goes ON the
    summary line."""
    out = _drain_summary(tmp_path, capsys, DrainResult(held_privacy_unverified=3))
    assert "held_privacy_unverified=3" in out
    assert "Held: privacy classes unverified" in out
    summary = [ln for ln in out.splitlines() if ln.startswith("attempted=")]
    assert summary and "Held: privacy classes unverified" in summary[0]


def test_the_summary_stays_quiet_when_nothing_was_held_for_privacy(tmp_path, capsys):
    out = _drain_summary(tmp_path, capsys, DrainResult(attempted=1, succeeded=1))
    assert "held_privacy" not in out
    assert "Held:" not in out


# --- missed-slot observer runs on every real drain pass ----------------
#
# ``notify.check_missed_slots()`` (console/notify.py) is otherwise called only
# from the console's optional in-process ``Ticker`` (``GMS_CONSOLE_TICKER``).
# The cron-driven drain (``dispatch-queue --apply`` -> this handler ->
# ``drain_queue``) runs it too, or a missed slot would go unnotified on a
# deployment without the console.


def test_apply_drain_runs_the_missed_slot_observer_once(tmp_path):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())
    config = _config(tmp_path, youtube="first_party")

    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.notify") as notify,
    ):
        drain.return_value = DrainResult()
        rc = cmd_dispatch_queue(_args(apply=True), config)

    assert rc == 0
    notify.check_missed_slots.assert_called_once_with(config=config)


def test_apply_drain_does_not_run_the_observer_on_a_dry_run(tmp_path):
    """Dry run drains nothing; it must not notify either."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())

    with mock.patch("gomoveshift.cli.handlers.dispatch_queue.notify") as notify:
        rc = cmd_dispatch_queue(_args(), _config(tmp_path, youtube="first_party"))

    assert rc == 0
    notify.check_missed_slots.assert_not_called()


def test_missed_slot_observer_exception_does_not_fail_the_drain(tmp_path, capsys):
    """A failed alert must not block publishing: the drain's own outcome
    (exit status and summary) is unchanged by the observer blowing up."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())

    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.notify.check_missed_slots",
            side_effect=RuntimeError("boom"),
        ),
    ):
        drain.return_value = DrainResult(attempted=1, succeeded=1)
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, youtube="first_party"))

    assert rc == 0
    out = capsys.readouterr().out
    assert "attempted=1 succeeded=1" in out


def test_instagram_uploads_the_rendition_after_verifying_the_master(tmp_path, monkeypatch):
    record = _record_with_media(tmp_path, data=b"big-master", codec="hevc")
    seen = {}

    def fake(source, **_kwargs):
        seen["source"] = Path(source)
        out = Path(source).with_suffix(".ig.mp4")
        out.write_bytes(b"ig-rendition")
        return out

    monkeypatch.setattr("gomoveshift.cli.handlers.dispatch_queue.ensure_instagram_rendition", fake)
    assert _read_instagram_media(record) == b"ig-rendition"
    assert seen["source"] == Path(record.media.path)


def test_instagram_verification_runs_before_any_transcode(tmp_path, monkeypatch):
    record = _record_with_media(tmp_path, codec="hevc", sha="b" * 64)
    monkeypatch.setattr(
        "gomoveshift.cli.handlers.dispatch_queue.ensure_instagram_rendition",
        lambda *a, **k: pytest.fail("transcoded a short that failed its fingerprint"),
    )
    with pytest.raises(TransientMediaError):
        _read_instagram_media(record)


def test_only_the_instagram_handler_gets_the_rendition_reader():
    from gomoveshift.cli.handlers.dispatch_queue import _build_routed_handlers

    config = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        photos=SimpleNamespace(site_root="/site-a", resolve_site_root=lambda: Path("/site-a")),
    )
    with (
        mock.patch(
            "gomoveshift.publishing.meta_auth.load_meta_credentials",
            return_value={"page_access_token": "tok", "page_id": "1", "ig_user_id": "2"},
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.build_handlers") as build,
    ):
        _build_routed_handlers(None, None, config, ["instagram", "facebook"], worker_id="w1")
    kwargs = build.call_args.kwargs
    assert kwargs["instagram"][1] is _read_instagram_media
    assert kwargs["facebook"][1] is _read_media


def test_instagram_passes_the_fingerprint_bitrate(tmp_path, monkeypatch):
    record = _record_with_media(tmp_path, data=b"x" * 1000, codec="hevc")
    seen = {}

    def fake(source, **kwargs):
        seen.update(kwargs)
        return Path(source)

    monkeypatch.setattr("gomoveshift.cli.handlers.dispatch_queue.ensure_instagram_rendition", fake)
    _read_instagram_media(record)
    # 1000 bytes over the fixture's 2.0 s
    assert seen["bitrate_bps"] == 4000


# --- TikTok Studio pack ------------------------------------------------

_DQ = "gomoveshift.cli.handlers.dispatch_queue"


def _pack_run(tmp_path, monkeypatch, *, mode=None, apply=True, sweep=None, routes=None):
    """One dispatch-queue pass with the sweep, drain and handler build faked.
    Returns the order of calls and the build/sweep mocks."""
    if mode is None:
        monkeypatch.delenv("GMS_TIKTOK_MODE", raising=False)
    else:
        monkeypatch.setenv("GMS_TIKTOK_MODE", mode)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())
    calls = []
    drain_kwargs = []
    sweep_mock = mock.Mock(
        side_effect=sweep
        or (lambda *a, **k: calls.append("sweep") or PackSweepResult(considered=1, packed=1))
    )
    with (
        mock.patch(f"{_DQ}.build_tiktok_reconciler", return_value=None),
        mock.patch(f"{_DQ}.build_pack_reconciler", return_value=None),
        mock.patch(f"{_DQ}.sweep_queued", sweep_mock),
        mock.patch(
            f"{_DQ}.drain_queue",
            side_effect=lambda *a, **k: (
                calls.append("drain") or drain_kwargs.append(k) or DrainResult()
            ),
        ),
        mock.patch(f"{_DQ}._build_routed_handlers") as build,
    ):
        rc = cmd_dispatch_queue(
            _args(apply=apply), _config(tmp_path, **(routes or {"tiktok": "first_party"}))
        )
    sweep_mock.drain_kwargs = drain_kwargs
    return rc, calls, sweep_mock, build


def test_the_pack_sweep_runs_before_the_drain_by_default(tmp_path, monkeypatch, capsys):
    rc, calls, sweep, build = _pack_run(tmp_path, monkeypatch)
    assert rc == 0
    assert calls == ["sweep", "drain"]
    assert "tiktok pack: packed=1" in capsys.readouterr().out


def test_the_sweep_shares_the_drains_worker_identity(tmp_path, monkeypatch):
    _, _, sweep, build = _pack_run(tmp_path, monkeypatch)
    assert sweep.call_args.kwargs["worker_id"] == build.call_args.kwargs["worker_id"]


def test_the_sweep_never_runs_on_a_dry_run(tmp_path, monkeypatch):
    rc, calls, sweep, _ = _pack_run(tmp_path, monkeypatch, apply=False)
    assert rc == 0
    sweep.assert_not_called()
    assert calls == []


def test_the_sweep_does_not_run_when_tiktok_is_not_routed_first_party(tmp_path, monkeypatch):
    _, _, sweep, _ = _pack_run(
        tmp_path, monkeypatch, routes={"tiktok": "metricool", "youtube": "first_party"}
    )
    sweep.assert_not_called()


@pytest.mark.parametrize("mode", ["inbox", "direct"])
def test_the_sweep_does_not_run_in_the_other_modes(tmp_path, monkeypatch, mode):
    _, calls, sweep, _ = _pack_run(tmp_path, monkeypatch, mode=mode)
    sweep.assert_not_called()
    assert calls == ["drain"]


def test_a_sweep_that_raises_does_not_stop_the_drain(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise OSError("drive vanished")

    rc, _, sweep, _ = _pack_run(tmp_path, monkeypatch, sweep=boom)
    out = capsys.readouterr().out
    assert rc == 0
    assert "tiktok pack: failed (OSError" in out


def test_sweep_row_errors_are_printed_and_the_drain_still_runs(tmp_path, monkeypatch, capsys):
    calls = []
    result = PackSweepResult(considered=1, errors=["delivery 1: drive offline"])
    rc, calls, _, _ = _pack_run(tmp_path, monkeypatch, sweep=lambda *a, **k: result)
    out = capsys.readouterr().out
    assert rc == 0
    assert "error: delivery 1: drive offline" in out


def test_the_tiktok_rendition_loader_verifies_the_master_then_returns_the_path(
    tmp_path, monkeypatch
):
    from gomoveshift.cli.handlers.dispatch_queue import _tiktok_rendition_path

    record = _record_with_media(tmp_path, data=b"hevc-master", codec="hevc")
    seen = {}

    def fake_rendition(source, *, codec):
        seen["source"], seen["codec"] = Path(source), codec
        return Path(source).with_suffix(".h264.mp4")

    monkeypatch.setattr(f"{_DQ}.ensure_h264_rendition", fake_rendition)
    out = _tiktok_rendition_path(record)
    assert out == Path(record.media.path).with_suffix(".h264.mp4")
    assert seen == {"source": Path(record.media.path), "codec": "hevc"}


def test_the_tiktok_rendition_loader_refuses_a_swapped_master_before_any_transcode(
    tmp_path, monkeypatch
):
    from gomoveshift.cli.handlers.dispatch_queue import _tiktok_rendition_path

    record = _record_with_media(tmp_path, codec="hevc", sha="b" * 64)
    monkeypatch.setattr(
        f"{_DQ}.ensure_h264_rendition",
        lambda *a, **k: pytest.fail("transcoded a short that failed its fingerprint"),
    )
    with pytest.raises(TransientMediaError):
        _tiktok_rendition_path(record)


def test_routed_tiktok_handler_is_wired_with_the_rendition_loader():
    from gomoveshift.cli.handlers.dispatch_queue import (
        _build_routed_handlers,
        _tiktok_rendition_path,
    )

    config = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        drive=SimpleNamespace(publishing_tiktok_audit_approved=False),
    )
    with (
        mock.patch(
            "gomoveshift.publishing.tiktok_auth.get_valid_tiktok_credentials",
            return_value={"access_token": "tok"},
        ),
        mock.patch(f"{_DQ}.build_handlers") as build,
    ):
        _build_routed_handlers(None, None, config, ["tiktok"])
    assert build.call_args.kwargs["tiktok_rendition_path_loader"] is _tiktok_rendition_path


def _last_line(out):
    return [line for line in out.splitlines() if line.strip()][-1]


def test_the_pack_result_is_on_the_summary_line_the_cron_keeps(tmp_path, monkeypatch, capsys):
    result = PackSweepResult(
        considered=5,
        packed=2,
        already_packed=1,
        held_privacy=1,
        errors=["delivery 9: drive offline"],
    )
    rc, _, _, _ = _pack_run(tmp_path, monkeypatch, sweep=lambda *a, **k: result)
    last = _last_line(capsys.readouterr().out)
    assert rc == 0
    assert "tiktok_packed=2" in last
    assert "tiktok_pack_errors=1" in last
    assert "tiktok_already_packed=1" in last
    assert "tiktok_pack_held=1" in last


def test_a_clean_pass_adds_no_pack_fields_to_the_summary_line(tmp_path, monkeypatch, capsys):
    rc, _, _, _ = _pack_run(
        tmp_path, monkeypatch, sweep=lambda *a, **k: PackSweepResult(considered=0)
    )
    last = _last_line(capsys.readouterr().out)
    assert "tiktok_packed" not in last and "tiktok_pack_errors" not in last


def test_a_sweep_that_raises_counts_as_a_pack_error_on_the_summary_line(
    tmp_path, monkeypatch, capsys
):
    def boom(*a, **k):
        raise OSError("drive vanished")

    _pack_run(tmp_path, monkeypatch, sweep=boom)
    assert "tiktok_pack_errors=1" in _last_line(capsys.readouterr().out)


def test_the_sweep_gets_the_same_privacy_loader_as_the_drain(tmp_path, monkeypatch):
    from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader

    _, _, sweep, _ = _pack_run(tmp_path, monkeypatch)
    loader = sweep.call_args.kwargs["privacy_attestations"]
    assert isinstance(loader, PrivacyAttestationLoader)
    assert sweep.drain_kwargs[0]["privacy_attestations"] is loader


def test_an_unknown_tiktok_mode_skips_only_tiktok_and_says_so(tmp_path, monkeypatch, capsys):
    rc, calls, sweep, build = _pack_run(
        tmp_path,
        monkeypatch,
        mode="bogus",
        routes={"tiktok": "first_party", "youtube": "first_party"},
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == ["drain"]  # the other platforms still drain
    sweep.assert_not_called()
    assert build.call_args.args[3] == ["youtube"]  # TikTok never reaches the handler build
    assert "unknown GMS_TIKTOK_MODE" in out
    assert "tiktok_mode_invalid=1" in _last_line(out)


def test_pack_mode_builds_the_tiktok_handler_without_tiktok_credentials(monkeypatch):
    """Pack mode never calls TikTok, so an expired or unrefreshable TikTok
    token must not stop the hourly pass that also drains YT/IG/FB."""
    from gomoveshift.cli.handlers.dispatch_queue import _build_routed_handlers
    from gomoveshift.publishing.tiktok_auth import TikTokAuthError

    monkeypatch.setenv("GMS_TIKTOK_MODE", "pack")
    config = SimpleNamespace(
        seo=SimpleNamespace(credentials_dir="/creds/brand-a"),
        drive=SimpleNamespace(publishing_tiktok_audit_approved=False),
    )

    with mock.patch(
        "gomoveshift.publishing.tiktok_auth.get_valid_tiktok_credentials",
        side_effect=TikTokAuthError("refresh token expired"),
    ) as get_creds:
        handlers = _build_routed_handlers(None, None, config, ["tiktok"])

    get_creds.assert_not_called()
    assert "tiktok" in handlers


# --- pack reconcile (video.list) ---------------------------------------------


def _reconcile_run(tmp_path, monkeypatch, *, build, reconcile=None, mode=None):
    if mode is None:
        monkeypatch.delenv("GMS_TIKTOK_MODE", raising=False)
    else:
        monkeypatch.setenv("GMS_TIKTOK_MODE", mode)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())
    calls = []
    rec_mock = mock.Mock(
        side_effect=reconcile
        or (lambda *a, **k: calls.append("reconcile") or PackReconcileResult(considered=1))
    )
    with (
        mock.patch(f"{_DQ}.build_tiktok_reconciler", return_value=None),
        mock.patch(f"{_DQ}.build_pack_reconciler", build),
        mock.patch(f"{_DQ}.reconcile_packed", rec_mock),
        mock.patch(
            f"{_DQ}.sweep_queued",
            side_effect=lambda *a, **k: calls.append("sweep") or PackSweepResult(),
        ),
        mock.patch(
            f"{_DQ}.drain_queue",
            side_effect=lambda *a, **k: calls.append("drain") or DrainResult(),
        ),
        mock.patch(f"{_DQ}._build_routed_handlers"),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, tiktok="first_party"))
    return rc, calls, rec_mock


def test_pack_reconcile_runs_before_the_sweep_and_the_drain(tmp_path, monkeypatch, capsys):
    build = mock.Mock(return_value=(lambda cursor=None: ([], None, False), lambda v: v))
    rc, calls, rec = _reconcile_run(tmp_path, monkeypatch, build=build)
    assert rc == 0
    assert calls == ["reconcile", "sweep", "drain"]
    assert rec.call_args.kwargs["permalink_for"]("9") == "9"


def test_a_credentials_failure_does_not_stop_the_drain(tmp_path, monkeypatch, capsys):
    build = mock.Mock(side_effect=FileNotFoundError("no creds"))
    rc, calls, rec = _reconcile_run(tmp_path, monkeypatch, build=build)
    assert rc == 0
    assert calls == ["sweep", "drain"]
    rec.assert_not_called()
    assert "tiktok reconcile: skipped (FileNotFoundError" in capsys.readouterr().out


def test_a_reconcile_that_raises_does_not_stop_the_drain(tmp_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("video.list down")

    build = mock.Mock(return_value=(lambda cursor=None: ([], None, False), lambda v: v))
    rc, calls, _ = _reconcile_run(tmp_path, monkeypatch, build=build, reconcile=boom)
    assert rc == 0
    assert calls == ["sweep", "drain"]
    assert "tiktok reconcile: skipped (RuntimeError" in capsys.readouterr().out


def test_pack_reconcile_is_silent_about_nothing_to_do_and_reports_ambiguity(
    tmp_path, monkeypatch, capsys
):
    build = mock.Mock(return_value=(lambda cursor=None: ([], None, False), lambda v: v))
    rec = lambda *a, **k: PackReconcileResult(  # noqa: E731
        considered=2, published=1, ambiguous=1, ambiguous_rows=["delivery 5: 2 candidates"]
    )
    _reconcile_run(tmp_path, monkeypatch, build=build, reconcile=rec)
    out = capsys.readouterr().out
    assert "tiktok reconcile: considered=2 published=1 ambiguous=1" in out
    assert "delivery 5: 2 candidates" in out


@pytest.mark.parametrize("mode", ["inbox", "direct"])
def test_pack_reconcile_does_not_run_in_the_other_modes(tmp_path, monkeypatch, mode):
    build = mock.Mock()
    _, calls, rec = _reconcile_run(tmp_path, monkeypatch, build=build, mode=mode)
    build.assert_not_called()
    rec.assert_not_called()


# --- U-R5(b): parked Instagram rows are decided before the drain ---------------


def _ig_result(**over):
    base = dict(considered=0, published=0, retried=0, failed=0, unprovable=0, errors=[])
    return SimpleNamespace(**{**base, **over})


def test_ig_reconcile_runs_before_the_drain_when_instagram_is_routed(tmp_path, capsys):
    """A parked 'Media ID is not available' row is decided by the account
    scan BEFORE the drain reads the queue, so a proven-absent row re-queued
    this pass is drained this pass, and the result rides the cron log."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "instagram", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record(selected=("instagram",)))
    calls = []
    scan = mock.Mock(return_value=[])

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.build_ig_media_scanner",
            return_value=scan,
        ),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.reconcile_awaiting_ig",
            side_effect=lambda *a, **k: (
                calls.append(("ig", k["recent_media"])) or _ig_result(considered=1, unprovable=1)
            ),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.drain_queue",
            side_effect=lambda *a, **k: calls.append(("drain", None)) or DrainResult(),
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, instagram="first_party"))

    assert rc == 0
    assert calls == [("ig", scan), ("drain", None)]
    assert "ig reconcile: considered=1" in capsys.readouterr().out


def test_an_ig_scanner_that_cannot_be_built_does_not_stop_the_drain(tmp_path, capsys):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "instagram", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record(selected=("instagram",)))

    with (
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.build_ig_media_scanner",
            side_effect=FileNotFoundError("no creds"),
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.reconcile_awaiting_ig") as rec,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue") as drain,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, instagram="first_party"))

    assert rc == 0
    rec.assert_not_called()
    drain.assert_called_once()
    assert "ig reconcile: skipped (FileNotFoundError)" in capsys.readouterr().out


def test_an_unrouted_instagram_never_builds_the_scanner(tmp_path):
    """No first-party Instagram route means no Graph client and no SOPS
    decrypt for it."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "youtube", "HASH1")
    conn.close()
    _sidecar(tmp_path, _record())

    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.build_ig_media_scanner") as build,
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.reconcile_awaiting_ig") as rec,
        mock.patch(
            "gomoveshift.cli.handlers.dispatch_queue.drain_queue", return_value=DrainResult()
        ),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
    ):
        rc = cmd_dispatch_queue(_args(apply=True), _config(tmp_path, youtube="first_party"))

    assert rc == 0
    build.assert_not_called()
    rec.assert_not_called()
