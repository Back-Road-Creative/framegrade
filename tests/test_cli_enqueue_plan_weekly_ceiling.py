"""A brand's ``policy.weekly_ceiling`` caps ``enqueue-plan``'s rolling-week
ceiling. It only ever trims the config value -- it never raises it -- and the
gate logs which rung set the effective ceiling."""

from __future__ import annotations

import hashlib
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gomoveshift.cli.handlers.enqueue_plan import cmd_enqueue_plan
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
from gomoveshift.publishing.queue import enqueue, open_queue_db

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)

_MEDIA_PATH = Path(tempfile.mkdtemp(prefix="gms-enqueue-plan-ceiling-test-media-")) / "short.mp4"
_MEDIA_PATH.write_bytes(b"gms enqueue-plan weekly-ceiling test media fixture bytes")
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


def _seed_brand_policy(tmp_path, monkeypatch, slug, policy_yaml):
    """Point the brand-policy loader at a brand.yaml carrying ``policy_yaml``."""
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    (brands_dir / slug).mkdir(parents=True)
    (brands_dir / slug / "brand.yaml").write_text(policy_yaml)
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)


def test_brand_weekly_ceiling_below_config_caps_the_ceiling(tmp_path, monkeypatch, capsys):
    # The brand policy sets weekly_ceiling: 10.
    _seed_brand_policy(tmp_path, monkeypatch, "gomoveshift", "policy:\n  weekly_ceiling: 10\n")
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=10)

    rc = _run(tmp_path, out, ceiling=21, brand="gomoveshift")

    assert rc == 1
    out_text = capsys.readouterr().out
    assert "ceiling=10" in out_text

    conn = open_queue_db(tmp_path)
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM deliveries WHERE project_name='proj'"
    ).fetchone()["n"]
    conn.close()
    assert count == 0


def test_brand_with_no_weekly_ceiling_leaves_config_value_unchanged(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=4)

    rc = _run(tmp_path, out, ceiling=5, brand="no-such-brand")

    assert rc == 0
    conn = open_queue_db(tmp_path)
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM deliveries WHERE project_name='proj'"
    ).fetchone()["n"]
    conn.close()
    assert count == 1


def test_brand_weekly_ceiling_above_config_never_raises_it(tmp_path, monkeypatch, capsys):
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    slug_dir = brands_dir / "highcap"
    slug_dir.mkdir(parents=True)
    (slug_dir / "brand.yaml").write_text("policy:\n  weekly_ceiling: 30\n")
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)

    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=21)

    rc = _run(tmp_path, out, ceiling=21, brand="highcap")

    assert rc == 1
    out_text = capsys.readouterr().out
    assert "ceiling=21" in out_text


def test_effective_ceiling_and_rung_are_logged(tmp_path, monkeypatch, caplog):
    _seed_brand_policy(tmp_path, monkeypatch, "gomoveshift", "policy:\n  weekly_ceiling: 10\n")
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=0)

    rc = _run(tmp_path, out, ceiling=21, brand="gomoveshift", caplog=caplog)

    assert rc == 0
    messages = [r.message for r in caplog.records if "rolling week ceiling" in r.message]
    assert any("10" in m and "rung=brand-policy-cap" in m for m in messages)


def test_effective_ceiling_rung_is_config_when_brand_does_not_cap(tmp_path, caplog):
    out = tmp_path / "out"
    out.mkdir()
    _seed_queue(tmp_path, existing_count=0)

    rc = _run(tmp_path, out, ceiling=5, brand="no-such-brand", caplog=caplog)

    assert rc == 0
    messages = [r.message for r in caplog.records if "rolling week ceiling" in r.message]
    assert any("5" in m and "rung=config" in m for m in messages)
