"""Publish gate: a day's blur mask frame count must equal its source's. Fail closed, one incident
per drive-day, no override, and every upload entry point (``health publish``, the backlog
republish, Phase D's upload run) calls the same check. No ffprobe, no network: the container
probe is patched."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.health import publish
from gomoveshift.pipeline import blur_gate
from gomoveshift.pipeline.drive_phases import phase_d
from gomoveshift.publishing import mask_frame_gate as gate

from tests.test_publish_command import CAD, NOW, Rig

DATE = "2025-09-16"


@pytest.fixture
def incidents():
    """The conftest-isolated incidents dir (``GMS_HEALTH_INCIDENTS_DIR``)."""
    return Path(os.environ["GMS_HEALTH_INCIDENTS_DIR"])


def make_day(stages: Path, sidecar=306385, raw=True, date=DATE) -> Path:
    day = stages / f"day_{date}"
    day.mkdir(parents=True)
    if sidecar is not None:
        head = {"schema": "redactcam.timeline/v1", "source_hash": "x"}
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


# ── purged master: the persisted source frame count ──────────────────────────


def persist(day: Path, count=306377, *, date=DATE, **over) -> Path:
    """Write the mask and the file ``blur_gate.persist_source_frames`` would, as the pairing
    check leaves them (all three counts equal), then drop the raw so only the file speaks."""
    mask = day / f"day_{date}_blur_mask.mkv"
    mask.write_bytes(b"mask")
    sidecar = day / f"day_{date}_blur_regions.json"
    doc = {
        "schema": gate.SOURCE_FRAMES_SCHEMA,
        "date": date,
        "day_dir": day.name,
        "counts": {"container": count, "mask": count, "sidecar": count},
        "source_fingerprint": "fp",
        "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "mask": {"size": mask.stat().st_size, "mtime_ns": mask.stat().st_mtime_ns},
        "renderer": {"redactcam_version": "1", "decode_mode": "auto"},
        "redactcam_commit": None,
    }
    doc.update(over)
    path = day / f"day_{date}_source_frames.json"
    path.write_text(json.dumps(doc))
    (day / f"day_{date}_raw.mp4").unlink(missing_ok=True)
    return path


def no_probe(monkeypatch):
    def boom(path):
        raise AssertionError("no container is left to probe")

    monkeypatch.setattr(gate, "probe_frames", boom)


def test_purged_master_reads_the_persisted_source_count(tmp_path, monkeypatch, incidents):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    no_probe(monkeypatch)
    assert gate.mask_frame_refusal(tmp_path / "st", drive="kerry") is None
    assert rows(incidents) == []


def test_persisted_count_from_another_sidecar_is_unknown(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    sidecar = day / f"day_{DATE}_blur_regions.json"
    sidecar.write_text(sidecar.read_text() + "\n")  # a re-detect: a different sidecar
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_unknown" in why


def test_persisted_count_with_a_replaced_mask_is_unknown(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    (day / f"day_{DATE}_blur_mask.mkv").write_bytes(b"a different, longer mask")
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_unknown" in why


def test_persisted_count_disagreeing_with_sidecar_is_a_mismatch(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306385)
    persist(day, count=306377)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_mismatch" in why and "306385" in why and "306377" in why


def test_live_master_still_wins_over_the_persisted_count(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    (day / f"day_{DATE}_master.mp4").write_bytes(b"")
    probe(monkeypatch, 306000)  # the live container disagrees: the file must not vouch for it
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_mismatch" in why and "306000" in why


def test_an_unprobeable_raw_does_not_fall_back_to_the_persisted_count(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    (day / f"day_{DATE}_raw.mp4").write_bytes(b"")  # present but unreadable
    probe(monkeypatch, None)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_unknown" in why


def test_an_unreadable_persisted_file_stays_unknown(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    path = persist(day)
    path.write_text("{not json")
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_unknown" in why


@pytest.mark.parametrize(
    "over",
    [
        {"counts": {"container": 306377, "mask": 306385, "sidecar": 306377}},
        {"counts": {"container": 0, "mask": 0, "sidecar": 0}},
        {"counts": {"container": True, "mask": True, "sidecar": True}},
        {"schema": "other/v9"},
        {"sidecar_sha256": None},
        {"mask": None},
    ],
)
def test_a_malformed_or_unpaired_persisted_file_stays_unknown(tmp_path, over):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day, **over)
    why = gate.mask_frame_refusal(tmp_path / "st", drive="kerry")
    assert why and "mask_frame_count_unknown" in why


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
        now=NOW, drive_running=lambda jobs: False,
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


# ── the proof binds to its own day, survives no swap, never blocks ───────────


def real_persist(day: Path, monkeypatch, n=306377, date=DATE):
    """Drive the real ``persist_source_frames`` with only the frame probes patched."""
    mask = day / f"day_{date}_blur_mask.mkv"
    if not mask.exists():
        mask.write_bytes(b"maskbytes")
    sidecar = day / f"day_{date}_blur_regions.json"
    monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: n)
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: n)
    out = blur_gate.persist_source_frames(
        day / f"day_{date}_raw.mp4", mask, sidecar, source_fingerprint="fp"
    )
    (day / f"day_{date}_raw.mp4").unlink(missing_ok=True)
    return out


def is_unknown(stages: Path) -> bool:
    why = gate.mask_frame_refusal(stages, drive="kerry", record=False)
    return bool(why) and "mask_frame_count_unknown" in why


def test_real_persist_round_trips_through_the_gate(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    assert real_persist(day, monkeypatch) == day / f"day_{DATE}_source_frames.json"
    doc = json.loads((day / f"day_{DATE}_source_frames.json").read_text())
    assert doc["date"] == DATE and doc["day_dir"] == day.name
    assert gate.mask_frame_refusal(tmp_path / "st", drive="kerry", record=False) is None


def test_a_proof_copied_to_another_day_with_preserved_mtimes_is_unknown(tmp_path, monkeypatch):
    src = make_day(tmp_path / "a", sidecar=306377)
    real_persist(src, monkeypatch)
    other = "2025-09-17"
    dst = tmp_path / "st" / f"day_{other}"
    dst.mkdir(parents=True)
    for kind in ("blur_regions.json", "blur_mask.mkv", "source_frames.json"):
        shutil.copy2(src / f"day_{DATE}_{kind}", dst / f"day_{other}_{kind}")
    assert is_unknown(tmp_path / "st")


def test_a_v1_proof_is_unknown(tmp_path):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day, schema="gomoveshift.source_frames/v1")
    assert is_unknown(tmp_path / "st")


def test_a_proof_naming_another_day_dir_is_unknown(tmp_path):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day, day_dir="elsewhere")
    assert is_unknown(tmp_path / "st")


def test_a_dangling_master_symlink_is_present_not_absent(tmp_path):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    (day / f"day_{DATE}_master.mp4").symlink_to(tmp_path / "offline" / "master.mp4")
    assert is_unknown(tmp_path / "st")


def test_a_dangling_raw_symlink_is_present_not_absent(tmp_path):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    (day / f"day_{DATE}_raw.mp4").symlink_to(tmp_path / "offline" / "raw.mp4")
    assert is_unknown(tmp_path / "st")


def test_a_master_that_is_a_directory_stays_unknown(tmp_path):
    day = make_day(tmp_path / "st", sidecar=306377)
    persist(day)
    (day / f"day_{DATE}_master.mp4").mkdir()
    assert is_unknown(tmp_path / "st")


def test_same_bytes_rewritten_mask_with_a_new_mtime_is_unknown(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    real_persist(day, monkeypatch)
    mask = day / f"day_{DATE}_blur_mask.mkv"
    st = mask.stat()
    mask.write_bytes(mask.read_bytes())
    os.utime(mask, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
    assert is_unknown(tmp_path / "st")


def test_all_minus_one_counts_write_nothing(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    assert real_persist(day, monkeypatch, n=-1) is None
    assert not (day / f"day_{DATE}_source_frames.json").exists()


def test_positive_but_unequal_counts_write_nothing(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    (day / f"day_{DATE}_blur_mask.mkv").write_bytes(b"m")
    monkeypatch.setattr(blur_gate, "container_frame_count", lambda p: 306377)
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: 306385)
    sidecar = day / f"day_{DATE}_blur_regions.json"
    out = blur_gate.persist_source_frames(
        day / f"day_{DATE}_raw.mp4",
        day / f"day_{DATE}_blur_mask.mkv",
        sidecar,
        source_fingerprint="fp",
    )
    assert out is None
    assert not (day / f"day_{DATE}_source_frames.json").exists()


def test_a_failed_replace_leaves_no_proof_and_does_not_raise(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(blur_gate.os, "replace", boom)
    assert real_persist(day, monkeypatch) is None
    assert not (day / f"day_{DATE}_source_frames.json").exists()
    assert is_unknown(tmp_path / "st")


def test_an_unexpected_error_in_persist_never_blocks(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)

    def boom():
        raise RuntimeError("metadata backend exploded")

    monkeypatch.setattr(blur_gate, "renderer_identity", boom)
    assert real_persist(day, monkeypatch) is None
    assert not (day / f"day_{DATE}_source_frames.json").exists()


def test_a_mask_swapped_while_counting_is_not_vouched_for(tmp_path, monkeypatch):
    day = make_day(tmp_path / "st", sidecar=306377)
    mask = day / f"day_{DATE}_blur_mask.mkv"
    mask.write_bytes(b"maskbytes")

    def count(p):
        if Path(p) == mask:
            mask.write_bytes(b"a different mask of other length")
        return 306377

    monkeypatch.setattr(blur_gate, "container_frame_count", count)
    monkeypatch.setattr(blur_gate, "sidecar_frame_count", lambda p: 306377)
    out = blur_gate.persist_source_frames(
        day / f"day_{DATE}_raw.mp4",
        mask,
        day / f"day_{DATE}_blur_regions.json",
        source_fingerprint="fp",
    )
    assert out is None
    assert not (day / f"day_{DATE}_source_frames.json").exists()
