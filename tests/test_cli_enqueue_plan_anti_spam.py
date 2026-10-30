"""``cmd_enqueue_plan`` refuses the whole batch on an anti-spam violation
before anything is written to the queue -- same shape as the media/privacy/
rolling-ceiling gates it sits beside."""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from gomoveshift.cli.handlers.enqueue_plan import cmd_enqueue_plan
from gomoveshift.pipeline.drive_phases.phase_d import _serialize_publish_plan
from gomoveshift.pipeline.review_gate import compute_gate_cd_hash
from gomoveshift.publishing.anti_spam_gate import LEGACY_TITLE_POOL
from gomoveshift.publishing.models import (
    MediaFingerprint,
    PlatformPayloads,
    PublishPlan,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    YouTubePayload,
)
from gomoveshift.publishing.queue import open_queue_db

SCHEDULED = datetime(2026, 9, 1, 19, 0, tzinfo=timezone.utc)

_MEDIA_PATH = Path(tempfile.mkdtemp(prefix="gms-enqueue-plan-anti-spam-test-")) / "short.mp4"
_MEDIA_PATH.write_bytes(b"gms enqueue-plan anti-spam test media fixture bytes")
_REAL_MEDIA = MediaFingerprint(
    str(_MEDIA_PATH),
    _MEDIA_PATH.stat().st_size,
    2.0,
    1080,
    1920,
    "h264",
    hashlib.sha256(_MEDIA_PATH.read_bytes()).hexdigest(),
)


def _record(short_index, title, sampled_regions=("a",)):
    return ShortPublishRecord(
        project_name="proj",
        lf_index=0,
        short_index=short_index,
        media=_REAL_MEDIA,
        schedule=ScheduleWindow(SCHEDULED, "America/Chicago"),
        recording_date="2026-08-01",
        sampled_regions=sampled_regions,
        parent_lf_video_id="vid",
        parent_lf_publish_at=SCHEDULED,
        payloads=PlatformPayloads(youtube=YouTubePayload(title, "a real description")),
        consent_hash="c",
        reviewed_plan_hash="HASH1",
        routes=RouteSelection(selected=("youtube",)),
    )


def _sidecar(out_dir, *records):
    path = out_dir / "proj_publish_plan.json"
    path.write_text(_serialize_publish_plan(PublishPlan(shorts=records)), encoding="utf-8")
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


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _args(state_path):
    return SimpleNamespace(
        state_path=str(state_path),
        apply=True,
        output_dir=None,
        shadow=False,
    )


def _seed_brand_policy(tmp_path, monkeypatch, slug, policy_yaml):
    """Point the brand-policy loader at a brand.yaml carrying ``policy_yaml``."""
    import gomoveshift.publishing.brand_policy as brand_policy_mod

    brands_dir = tmp_path / "brands"
    (brands_dir / slug).mkdir(parents=True)
    (brands_dir / slug / "brand.yaml").write_text(policy_yaml)
    monkeypatch.setattr(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir)


def _run(tmp_path, out, brand):
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
        return cmd_enqueue_plan(_args(state_path), _config(tmp_path))


def _spooled_count(tmp_path):
    conn = open_queue_db(tmp_path)
    count = conn.execute(
        "SELECT COUNT(*) AS n FROM deliveries WHERE project_name='proj'"
    ).fetchone()["n"]
    conn.close()
    return count


def test_legacy_title_refuses_the_whole_batch(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    legacy_text = next(iter(LEGACY_TITLE_POOL))
    _sidecar(
        out,
        _record(1, legacy_text),
        _record(2, "A grounded, specific title naming a real place"),
    )

    rc = _run(tmp_path, out, brand="no-such-brand")

    assert rc == 1
    out_text = capsys.readouterr().out
    assert "index=0" in out_text
    assert "rule=legacy-title" in out_text
    assert _spooled_count(tmp_path) == 0


def test_shared_prefix_refuses_the_whole_batch(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(
        out,
        _record(1, "Driving the coast road toward Dingle"),
        _record(2, "Driving the coast highway near Tralee"),
    )

    rc = _run(tmp_path, out, brand="no-such-brand")

    assert rc == 1
    out_text = capsys.readouterr().out
    assert "rule=shared-prefix" in out_text
    assert "index=0" in out_text
    assert "index=1" in out_text
    assert _spooled_count(tmp_path) == 0


def test_location_cap_refuses_the_excess_shorts(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    out.mkdir()
    _seed_brand_policy(tmp_path, monkeypatch, "gomoveshift", "policy:\n  per_location_cap: 5\n")
    _titles = [
        "Killarney's coast road bends past three rivers",
        "Tralee's mountain pass climbs above the treeline",
        "Dingle's cliffside route hugs the Atlantic edge",
        "Kenmare's forest track winds through old oaks",
        "Cahersiveen's harbor lane skirts the fishing pier",
        "Waterville's dune road crosses the estuary bridge",
    ]
    records = [_record(i, title, sampled_regions=("kerry",)) for i, title in enumerate(_titles)]
    _sidecar(out, *records)

    rc = _run(tmp_path, out, brand="gomoveshift")

    assert rc == 1
    out_text = capsys.readouterr().out
    assert "rule=location-cap" in out_text
    assert "index=5" in out_text
    assert _spooled_count(tmp_path) == 0


def test_no_location_cap_for_a_brand_with_no_policy(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _titles = [
        "Killarney's coast road bends past three rivers",
        "Tralee's mountain pass climbs above the treeline",
        "Dingle's cliffside route hugs the Atlantic edge",
        "Kenmare's forest track winds through old oaks",
        "Cahersiveen's harbor lane skirts the fishing pier",
        "Waterville's dune road crosses the estuary bridge",
    ]
    records = [_record(i, title, sampled_regions=("kerry",)) for i, title in enumerate(_titles)]
    _sidecar(out, *records)

    rc = _run(tmp_path, out, brand="no-such-brand")

    assert rc == 0
    assert _spooled_count(tmp_path) == 6


def test_clean_batch_spools_normally(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(
        out,
        _record(1, "A grounded, specific title naming Killarney's coast road"),
        _record(2, "Another distinct title about a different mountain pass"),
    )

    rc = _run(tmp_path, out, brand="no-such-brand")

    assert rc == 0
    assert _spooled_count(tmp_path) == 2


def test_cross_batch_repeat_of_a_spooled_title_refuses_the_new_batch(tmp_path, capsys):
    import dataclasses

    title = "Killarney's coast road unwinds past three river bends"
    first = tmp_path / "first"
    first.mkdir()
    _sidecar(first, _record(1, title))
    assert _run(tmp_path, first, brand="no-such-brand") == 0

    second = tmp_path / "second"
    second.mkdir()
    _sidecar(second, dataclasses.replace(_record(1, title + " again"), project_name="other"))
    capsys.readouterr()

    rc = _run(tmp_path, second, brand="no-such-brand")

    assert rc == 1
    assert "rule=cross-batch-repeat" in capsys.readouterr().out
    assert _spooled_count(tmp_path) == 1


def test_rerunning_the_same_plan_is_not_a_cross_batch_repeat(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    _sidecar(out, _record(1, "Killarney's coast road unwinds past three river bends"))
    assert _run(tmp_path, out, brand="no-such-brand") == 0

    assert _run(tmp_path, out, brand="no-such-brand") == 0
