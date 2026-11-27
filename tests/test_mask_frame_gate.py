"""Publish gate: a day's blur mask frame count must equal its source's. Fail closed, one incident
per drive-day, no override, and every upload entry point (``health publish``, the backlog
republish, Phase D's upload run) calls the same check. No ffprobe, no network: the container
probe is patched."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.health import publish
from gomoveshift.pipeline.drive_phases import phase_d
from gomoveshift.publishing import mask_frame_gate as gate

from tests.test_publish_command import CAD, NOW, Rig

DATE = "2025-09-16"


@pytest.fixture
def incidents():
    """The conftest-isolated incidents dir (``GMS_HEALTH_INCIDENTS_DIR``)."""
    return Path(os.environ["GMS_HEALTH_INCIDENTS_DIR"])


def make_day(stages: Path, sidecar=306385, raw=True, date=DATE, fps=59.9386) -> Path:
    day = stages / f"day_{date}"
    day.mkdir(parents=True)
    if sidecar is not None:
        head = {"schema": "redactcam.timeline/v1", "source_hash": "x", "source_fps": fps}
        if sidecar:
            head["source_frame_count"] = sidecar
        text = json.dumps(head, indent=2)[:-2] + ',\n  "frame_boxes": {}\n}'
        (day / f"day_{date}_blur_regions.json").write_text(text)
    if raw:
        (day / f"day_{date}_raw.mp4").write_bytes(b"")
    return day


def probe(monkeypatch, count):
    monkeypatch.setattr(gate, "probe_frames", lambda path: count)


def rows(incidents: Path):
    return [json.loads(p.read_text()) for p in sorted(incidents.glob("*.json"))]


def test_mismatch_is_refused_with_both_counts_and_an_incident(tmp_path, monkeypatch, incidents):
    make_day(tmp_path / "st")
    probe(monkeypatch, 306377)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and DATE in why and "306385" in why and "306377" in why and "+8" in why
    (row,) = rows(incidents)
    assert row["kind"] == "mask_index_mismatch" and row["drive"] == "kerry"
    assert row["sidecar_frames"] == 306385 and row["container_frames"] == 306377
    assert row["difference"] == 8 and row["resolved_at"] is None


def test_equal_counts_are_allowed_and_write_nothing(tmp_path, monkeypatch, incidents):
    make_day(tmp_path / "st", sidecar=306377)
    probe(monkeypatch, 306377)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="kerry") is None
    assert rows(incidents) == []


@pytest.mark.parametrize("sidecar", [None, 0])  # no file / no count field
def test_missing_sidecar_or_count_is_refused_fail_closed(tmp_path, monkeypatch, sidecar):
    make_day(tmp_path / "st", sidecar=sidecar)
    probe(monkeypatch, 306377)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_unknown" in why


def test_container_that_cannot_be_probed_is_refused(tmp_path, monkeypatch):
    make_day(tmp_path / "st", raw=False)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_unknown" in why


# ── a finished day: the raw is deleted at encode, the master by cleanup's master-drop ──


def finished_day(stages: Path, episodes, sidecar=306385, master=False, record=True) -> Path:
    """A day with no raw: ``episodes`` maps each ``episode_NN`` stem to its frame count. The colour
    cache names their fingerprints, as the encode records them."""
    day = make_day(stages, sidecar=sidecar, raw=False)
    paths = []
    for i, name in enumerate(episodes, start=1):
        path = stages / f"{name}_{DATE}.mp4"
        path.write_bytes(b"x" * i)
        paths.append(path)
    fps = [f"{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in paths] if record else []
    (day / f"day_{DATE}_color_cache.json").write_text(json.dumps({"source_fingerprints": fps}))
    if master:
        (day / f"day_{DATE}_master.mp4").write_bytes(b"")
    return day


def probe_by_name(monkeypatch, counts, rate=None):
    monkeypatch.setattr(
        gate, "probe_frames", lambda path: counts.get(Path(path).name.split("_")[1])
    )
    monkeypatch.setattr(gate, "probe_rate", lambda path: rate, raising=False)


def test_a_finished_day_counts_the_episodes_the_colour_cache_recorded(tmp_path, monkeypatch):
    finished_day(tmp_path / "st", ["episode_01", "episode_02"], sidecar=306377)
    probe_by_name(monkeypatch, {"01": 200000, "02": 106377})
    assert gate.mask_frame_refusal(tmp_path / "st", drive="kerry", record=False) is None


def test_a_finished_day_whose_episodes_sum_short_is_a_mismatch(tmp_path, monkeypatch):
    finished_day(tmp_path / "st", ["episode_01", "episode_02"])
    probe_by_name(monkeypatch, {"01": 200000, "02": 106377})
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry", record=False)
    assert why and "mask_frame_count_mismatch" in why and "306377" in why and "+8" in why


def test_episodes_the_colour_cache_did_not_record_refuse_unknown(tmp_path, monkeypatch):
    stages = tmp_path / "st"
    finished_day(stages, ["episode_01", "episode_02"], sidecar=306377)
    (stages / f"episode_02_{DATE}.mp4").write_bytes(b"changed")
    probe_by_name(monkeypatch, {"01": 200000, "02": 106377})
    why = gate.mask_frame_refusal(stages, drive="kerry", record=False)
    assert why and "mask_frame_count_unknown" in why


def test_a_day_with_no_colour_cache_record_refuses_unknown(tmp_path, monkeypatch):
    finished_day(tmp_path / "st", ["episode_01"], sidecar=200000, record=False)
    probe_by_name(monkeypatch, {"01": 200000})
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry", record=False)
    assert why and "mask_frame_count_unknown" in why


def test_a_half_rate_master_is_not_the_masks_frame_space(tmp_path, monkeypatch):
    """Old Head: a 59.94 fps mask of 368,837 frames against a 29.97 fps master of 184,421."""
    finished_day(tmp_path / "st", [], sidecar=368837, master=True, record=False)
    probe_by_name(monkeypatch, {f"{DATE}": 184421}, rate=30000 / 1001)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="old-head", record=False)
    assert why and "mask_frame_count_unknown" in why and "mismatch" not in why


def test_a_half_rate_master_defers_to_the_recorded_episodes(tmp_path, monkeypatch):
    finished_day(tmp_path / "st", ["episode_01", "episode_02"], sidecar=368837, master=True)
    probe_by_name(monkeypatch, {"01": 300000, "02": 68837, f"{DATE}": 184421}, rate=30000 / 1001)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="old-head", record=False) is None


@pytest.mark.parametrize("master_frames, refused", [(306385, False), (306377, True)])
def test_a_master_at_the_masks_rate_still_counts(tmp_path, monkeypatch, master_frames, refused):
    finished_day(tmp_path / "st", [], master=True, record=False)
    probe_by_name(monkeypatch, {f"{DATE}": master_frames}, rate=60000 / 1001)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry", record=False)
    assert bool(why) is refused and (not why or "mask_frame_count_mismatch" in why)


def test_one_open_incident_across_repeated_ticks_then_resolved_when_counts_agree(
    tmp_path, monkeypatch, incidents
):
    day = make_day(tmp_path / "st")
    probe(monkeypatch, 306377)
    for _ in range(3):
        assert gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert len(rows(incidents)) == 1
    (day / f"day_{DATE}_blur_regions.json").write_text('{"source_frame_count": 306377}')
    assert gate.mask_frame_refusal(tmp_path / "st", drive="kerry") is None
    (row,) = rows(incidents)
    assert row["resolved_at"]


def test_dry_run_records_no_incident(tmp_path, monkeypatch, incidents):
    make_day(tmp_path / "st")
    probe(monkeypatch, 306377)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="kerry", record=False)
    assert rows(incidents) == []


def test_sidecar_count_reads_only_the_header(tmp_path):
    p = tmp_path / "s.json"
    p.write_text('{"source_frame_count": 12, "frame_boxes": {"source_frame_count": 99}}')
    assert gate.sidecar_frame_count(p) == 12


# ── entry points ─────────────────────────────────────────────────────────────


def test_health_publish_refuses_a_mismatched_day_before_any_write(tmp_path, monkeypatch, incidents):
    rig = Rig(tmp_path)
    rig.audit()
    make_day(rig.stages / "day-a" / "day-a" / "_stages")
    probe(monkeypatch, 306377)
    before = rig.snapshot()
    res = rig.run()
    assert res["action"] == "refused" and "306385" in res["reason"] and "306377" in res["reason"]
    assert rig.snapshot() == before and not rig.launches
    assert len(rows(incidents)) == 1


def test_health_publish_allows_an_agreeing_day(tmp_path, monkeypatch):
    rig = Rig(tmp_path)
    rig.audit()
    make_day(rig.stages / "day-a" / "day-a" / "_stages", sidecar=306377)
    probe(monkeypatch, 306377)
    assert rig.run()["action"] == "launched"


def test_backlog_republish_goes_through_the_same_refusal(tmp_path, monkeypatch):
    rig = Rig(tmp_path)
    rig.audit()
    make_day(rig.stages / "day-a" / "day-a" / "_stages")
    probe(monkeypatch, 306377)
    monkeypatch.setattr(publish, "open_schedule_store", lambda config: (rig.store, CAD))
    run = publish.backlog_publisher(
        None, region="usa", stages_root=rig.stages, code_root=tmp_path, jobs_dir=rig.jobs,
        now=NOW,
    )  # fmt: skip
    out = run("day-a", False)
    assert out["action"] == "refused" and "306385" in out["reason"]


def test_phase_d_upload_run_raises_before_anything_else(tmp_path, monkeypatch):
    make_day(tmp_path / "st")
    probe(monkeypatch, 306377)
    monkeypatch.setattr(phase_d, "publish_lf_paths", lambda state: [])
    monkeypatch.setattr(phase_d, "_reconcile_shorts_per_lf_against_disk", lambda state: None)
    monkeypatch.setattr(phase_d, "_run_sidecar_integrity_gate", lambda **kw: None)
    state = SimpleNamespace(
        drive_plan=None, shorts_per_lf={}, region="ireland", publish_day="wed", location="x",
        output_base=str(tmp_path / "out"), skip_upload=False, skip_drive_upload=False,
        skip_youtube_upload=False, skip_csv=False, force_sidecar_reset=False, youtube_upload=True,
        stages_base=str(tmp_path / "st"), source_dir=str(tmp_path / "Kerry To Cork"),
    )  # fmt: skip
    with pytest.raises(gate.MaskFrameCountError, match="306385"):
        phase_d.PhaseD(config=SimpleNamespace()).run(state)


def test_phase_d_audit_pass_is_not_gated(tmp_path, monkeypatch):
    make_day(tmp_path / "st")
    probe(monkeypatch, 306377)
    monkeypatch.setattr(phase_d, "publish_lf_paths", lambda state: [])
    monkeypatch.setattr(phase_d, "_reconcile_shorts_per_lf_against_disk", lambda state: None)
    monkeypatch.setattr(phase_d, "_run_sidecar_integrity_gate", lambda **kw: None)
    state = SimpleNamespace(
        drive_plan=None, shorts_per_lf={}, region="ireland", publish_day="wed", location="x",
        output_base=str(tmp_path / "out"), skip_upload=True, skip_drive_upload=True,
        skip_youtube_upload=True, skip_csv=False, force_sidecar_reset=False, youtube_upload=False,
        stages_base=str(tmp_path / "st"), source_dir=str(tmp_path / "Kerry To Cork"),
        metadata_per_lf=None, drive_schedule=None, social_posts_per_lf=None, upload_manifests=None,
    )  # fmt: skip
    phase_d.PhaseD(config=SimpleNamespace()).run(state)


# ── health check must survive the row; tests must not write into the checkout ──


def _run_check(tmp_path, incidents, notify):
    from datetime import datetime, timezone

    from gomoveshift.health.check import run_check

    return run_check(
        SimpleNamespace(seo=SimpleNamespace(storage_path="relative/seo")),
        incidents_dir=incidents, jobs_dir=tmp_path / "jobs", code_root=tmp_path,
        staging_sha="abc123", now=datetime(2026, 11, 20, 12, tzinfo=timezone.utc),
        notify_fn=notify,
    )  # fmt: skip


def test_health_check_pushes_a_mask_incident_with_its_detail_line(tmp_path, monkeypatch, incidents):
    make_day(tmp_path / "st")
    probe(monkeypatch, 306377)
    gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    sent = []
    res = _run_check(tmp_path, incidents, lambda ev, payload, text: sent.append(text) or True)
    (row,) = [r for r in rows(incidents) if r["kind"] == "mask_index_mismatch"]
    assert row["pushed"] is True and res["pushed"] == [row["key"]]
    assert sent == [row["detail"]]
    assert "kerry" in sent[0] and DATE in sent[0] and "306385" in sent[0] and "306377" in sent[0]


def test_incident_line_without_detail_keeps_the_due_deliveries_text():
    from gomoveshift.health.check import _line

    row = {"kind": "queue_stalled", "due_deliveries": 3, "quiet_hours": 5}
    assert _line(row) == "3 publishing deliveries are due and nothing has touched the queue in 5 h."


def test_tests_cannot_write_into_the_checkout_gms_dir(tmp_path, monkeypatch, incidents):
    make_day(tmp_path / "st")
    probe(monkeypatch, 306377)
    gate.mask_frame_refusal(tmp_path / "st", drive="isolation-probe")
    assert len(rows(incidents)) == 1
    repo = Path(__file__).resolve().parents[1]
    assert not list((repo / "gms" / "incidents").glob("*isolation-probe*"))
