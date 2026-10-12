"""The bounded-response ladder is a third rung on ``enqueue-plan``'s
rolling-week ceiling -- config -> brand-policy-cap -> ladder. Only a "live"
``platform_daily_caps`` row for youtube (the reference platform) actually
lowers the ceiling; a "shadow" row is logged only. Never raises the ceiling
above what the earlier rungs already set."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gomoveshift.cli.handlers.enqueue_plan import _healthy_days, cmd_enqueue_plan
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.pipeline.review_gate import compute_gate_cd_hash
from gomoveshift.publishing.models import (
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    YouTubePayload,
)
from gomoveshift.publishing.queue import (
    enqueue,
    log_response,
    open_queue_db,
    set_platform_daily_cap,
)

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)

_MEDIA_PATH = Path(tempfile.mkdtemp(prefix="gms-enqueue-plan-ladder-test-media-")) / "short.mp4"
_MEDIA_PATH.write_bytes(b"gms enqueue-plan ladder-ceiling test media fixture bytes")
_REAL_MEDIA = MediaFingerprint(
    str(_MEDIA_PATH),
    _MEDIA_PATH.stat().st_size,
    2.0,
    1080,
    1920,
    "h264",
    hashlib.sha256(_MEDIA_PATH.read_bytes()).hexdigest(),
)


def _record(short_index):
    return ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=short_index,
        media=_REAL_MEDIA,
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=("a",),
        parent_lf_video_id="vid",
        parent_lf_publish_at=SCHEDULED,
        payloads=PlatformPayloads(youtube=YouTubePayload("t", "d")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube",)),
    )


def _sidecar(out_dir):
    path = out_dir / "proj_publish_plan.json"
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=(_record(5),))), encoding="utf-8")
    return path


def _state():
    return SimpleNamespace(
        regular_lf_paths=[],
        gate_cd_confirmed=True,
        gate_cd_confirmation_hash=None,
    )


def _approved_state(out_dir):
    state = _state()
    state.gate_cd_confirmation_hash = compute_gate_cd_hash(state=state, output_dir=out_dir)
    return state


def _config(tmp_path, ceiling):
    return SimpleNamespace(
        seo=SimpleNamespace(storage_path=str(tmp_path), rolling_week_ceiling_per_platform=ceiling)
    )


def _args(state_path):
    return SimpleNamespace(
        state_path=str(state_path),
        apply=True,
        output_dir=None,
        shadow=False,
    )


def _seed_queue(tmp_path, existing_count):
    conn = open_queue_db(tmp_path)
    for i in range(existing_count):
        enqueue(conn, "other-proj", i, "youtube", f"hash{i}", scheduled_at=SCHEDULED.isoformat())
    conn.close()


def _run(tmp_path, out, ceiling, brand, caplog=None):
    _sidecar(out)
    state_path = out / "pipeline_state.json"
    state_path.write_text("{}", encoding="utf-8")
    with (
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan._load_state",
            return_value=_approved_state(out),
        ),
        mock.patch(
            "gomoveshift.cli.handlers.enqueue_plan.current_brand",
            return_value=brand,
        ),
    ):
        if caplog is not None:
            with caplog.at_level(logging.INFO, logger="gomoveshift.cli.handlers.enqueue_plan"):
                return cmd_enqueue_plan(_args(state_path), _config(tmp_path, ceiling))
        return cmd_enqueue_plan(_args(state_path), _config(tmp_path, ceiling))


def _rung_messages(caplog):
    return [r.message for r in caplog.records if "rolling week ceiling" in r.message]


def test_no_cap_row_leaves_ceiling_unchanged(tmp_path, caplog):
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=0)

    rc = _run(tmp_path, out, ceiling=5, brand="no-such-brand", caplog=caplog)

    assert rc == 0
    messages = _rung_messages(caplog)
    assert any("5" in m and "rung=config" in m for m in messages)


def test_shadow_row_leaves_ceiling_unchanged_and_logs_would_be(tmp_path, caplog):
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=0)

    conn = open_queue_db(tmp_path)
    set_platform_daily_cap(conn, "youtube", cap=1, state="RECOVERING", mode="shadow")
    log_response(conn, "youtube", "RECOVERING", 1, "cap", "shadow")
    conn.close()

    rc = _run(tmp_path, out, ceiling=5, brand="no-such-brand", caplog=caplog)

    assert rc == 0
    messages = _rung_messages(caplog)
    assert any("5" in m and "rung=config" in m for m in messages)
    would_be = [r.message for r in caplog.records if "ladder rung would set ceiling" in r.message]
    assert any("mode=shadow" in m and "state=RECOVERING" in m for m in would_be)


def test_live_row_lowers_ceiling_with_ladder_rung(tmp_path, caplog):
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=0)

    conn = open_queue_db(tmp_path)
    # RECOVERING, no healthy days yet -> rung 0 of the default ladder
    # (1, 2, 3) -> daily cap 1 -> weekly 7, below the config ceiling of 21.
    set_platform_daily_cap(conn, "youtube", cap=1, state="RECOVERING", mode="live")
    log_response(conn, "youtube", "RECOVERING", 1, "cap", "live")
    conn.close()

    rc = _run(tmp_path, out, ceiling=21, brand="no-such-brand", caplog=caplog)

    assert rc == 0
    messages = _rung_messages(caplog)
    assert any("7" in m and "rung=ladder" in m for m in messages)


def test_live_row_whose_ladder_volume_is_not_lower_leaves_ceiling_unchanged(tmp_path, caplog):
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=0)

    conn = open_queue_db(tmp_path)
    # HEALTHY -> uncapped by the ladder -> rungs[-1] * 7 = 21, equal to the
    # config ceiling -- "< ceiling" only, never "<=", so this must NOT win.
    set_platform_daily_cap(conn, "youtube", cap=3, state="HEALTHY", mode="live")
    log_response(conn, "youtube", "HEALTHY", 3, "cap", "live")
    conn.close()

    rc = _run(tmp_path, out, ceiling=21, brand="no-such-brand", caplog=caplog)

    assert rc == 0
    messages = _rung_messages(caplog)
    assert any("21" in m and "rung=config" in m for m in messages)


def _seed_brand_policy(tmp_path, monkeypatch, slug, policy_yaml):
    """Point the brand-policy loader at a brand.yaml carrying ``policy_yaml``."""
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    (brands_dir / slug).mkdir(parents=True)
    (brands_dir / slug / "brand.yaml").write_text(policy_yaml)
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)


def test_live_row_never_raises_the_ceiling_above_brand_policy_cap(tmp_path, monkeypatch, caplog):
    # The brand policy sets weekly_ceiling: 10; a HEALTHY ladder rung computes
    # 21, well above -- must not overwrite rung=brand-policy-cap.
    _seed_brand_policy(tmp_path, monkeypatch, "gomoveshift", "policy:\n  weekly_ceiling: 10\n")
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=0)

    conn = open_queue_db(tmp_path)
    set_platform_daily_cap(conn, "youtube", cap=3, state="HEALTHY", mode="live")
    log_response(conn, "youtube", "HEALTHY", 3, "cap", "live")
    conn.close()

    rc = _run(tmp_path, out, ceiling=21, brand="gomoveshift", caplog=caplog)

    assert rc == 0
    messages = _rung_messages(caplog)
    assert any("10" in m and "rung=brand-policy-cap" in m for m in messages)


def test_healthy_days_counts_consecutive_distinct_healthy_days():
    tmp_path = Path(tempfile.mkdtemp(prefix="gms-healthy-days-test-"))
    conn = open_queue_db(tmp_path)
    log_response(conn, "youtube", "HEALTHY", 3, "cap", "live", now="2026-09-01T10:00:00")
    log_response(conn, "youtube", "HEALTHY", 3, "cap", "live", now="2026-09-01T18:00:00")
    log_response(conn, "youtube", "RECOVERING", 2, "cap", "live", now="2026-09-02T10:00:00")
    log_response(conn, "youtube", "RECOVERING", 1, "cap", "live", now="2026-09-03T10:00:00")

    assert _healthy_days(conn, "youtube") == 3
    conn.close()


def test_healthy_days_stops_at_first_suppressed_row():
    tmp_path = Path(tempfile.mkdtemp(prefix="gms-healthy-days-test-"))
    conn = open_queue_db(tmp_path)
    log_response(conn, "youtube", "SUPPRESSED", 1, "cap", "live", now="2026-09-01T10:00:00")
    log_response(conn, "youtube", "RECOVERING", 1, "cap", "live", now="2026-09-02T10:00:00")
    log_response(conn, "youtube", "RECOVERING", 1, "cap", "live", now="2026-09-03T10:00:00")

    assert _healthy_days(conn, "youtube") == 2
    conn.close()


def test_healthy_days_zero_with_no_log_rows():
    tmp_path = Path(tempfile.mkdtemp(prefix="gms-healthy-days-test-"))
    conn = open_queue_db(tmp_path)

    assert _healthy_days(conn, "youtube") == 0
    conn.close()
