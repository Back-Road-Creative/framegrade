"""The publish plan carries its days' privacy attestations.

The encoder writes each day's attestation into the stage dir; dispatch reads
attestations from the publish plan's directory. Phase D bridges the two when
it writes the plan, so no plan exists without its attestation set.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader
from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.day_master_encoder import (
    PRIVACY_ATTESTATION_NAME,
    PRIVACY_ATTESTATION_SCHEMA,
    REQUIRED_PRIVACY_CLASSES,
)
from gomoveshift.pipeline.drive_phases.phase_d import _emit_publish_plan_sidecar
from gomoveshift.publishing.asset_plan import ASSET_ATTESTATION_FILENAME
from gomoveshift.publishing.dispatch import privacy_hold_reason
from gomoveshift.publishing.models import MediaFingerprint

FP = MediaFingerprint(
    path="/x/short_01.mp4",
    size_bytes=1,
    duration_seconds=9.0,
    width=1080,
    height=1920,
    codec="h264",
    sha256="a" * 64,
)
FINGERPRINT = "gomoveshift.publishing.media_fingerprint.fingerprint_media"


def _doc(date, state="verified"):
    return {
        "schema": PRIVACY_ATTESTATION_SCHEMA,
        "date": date,
        "required_classes": list(REQUIRED_PRIVACY_CLASSES),
        "classes": {c: {"state": state, "reason": "ok"} for c in REQUIRED_PRIVACY_CLASSES},
    }


def _stage_attestation(stages, date, state="verified"):
    d = stages / f"day_{date}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / PRIVACY_ATTESTATION_NAME.format(date=date)
    p.write_text(json.dumps(_doc(date, state), indent=2) + "\n", encoding="utf-8")
    return p


def _kwargs(dates, route=True):
    lfs = list(range(1, len(dates) + 1))
    cfg = DriveConfig(publishing_route_tiktok="first_party" if route else "off")
    return dict(
        project_base="p",
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=i) for i in lfs]),
        social_posts_per_lf={i: [SimpleNamespace(short_index=10 + i)] for i in lfs},
        drive_schedule=SimpleNamespace(
            shorts_slots=[
                SimpleNamespace(short_index=10 + i, date="2026-03-15", time="08:00:00") for i in lfs
            ]
        ),
        youtube_uploads={
            i: {"video_id": f"vid-{i}", "publish_at": "2026-03-14T08:00:00-05:00"} for i in lfs
        },
        per_lf_recording_date={i: d for i, d in zip(lfs, dates)},
        shorts_paths_by_lf={i: [f"/x/short_{i}.mp4"] for i in lfs},
        publish_pass=True,
        config=SimpleNamespace(drive=cfg),
    )


def _run(out, stages, dates, **kw):
    with patch(FINGERPRINT, return_value=FP):
        return _emit_publish_plan_sidecar(
            output_base=out, stages_base=stages, **_kwargs(dates, **kw)
        )


def _hold(plan_path):
    att = PrivacyAttestationLoader(plan_path.parent)({"sidecar_path": str(plan_path)})
    return privacy_hold_reason(att)


def _att_files(out):
    return sorted(p.name for p in out.glob("day_*_privacy_attestation.json"))


def test_verified_stage_attestation_lands_beside_plan_and_releases(tmp_path):
    out, stages = tmp_path / "out", tmp_path / "stages"
    out.mkdir()
    src = _stage_attestation(stages, "2026-03-14")
    plan = _run(out, stages, ["2026-03-14"])
    dst = out / "day_2026-03-14_privacy_attestation.json"
    assert dst.read_bytes() == src.read_bytes()
    assert _hold(plan) is None


def test_missing_stage_attestation_writes_inconclusive_and_warns(tmp_path, caplog):
    out, stages = tmp_path / "out", tmp_path / "stages"
    out.mkdir()
    with caplog.at_level(logging.WARNING):
        plan = _run(out, stages, ["2026-03-14"])
    doc = json.loads((out / "day_2026-03-14_privacy_attestation.json").read_text())
    assert doc["schema"] == PRIVACY_ATTESTATION_SCHEMA
    assert doc["date"] == "2026-03-14"
    assert set(doc["classes"]) == set(REQUIRED_PRIVACY_CLASSES)
    assert all(v["state"] == "inconclusive" for v in doc["classes"].values())
    assert "inconclusive" in _hold(plan)
    assert any(
        r.levelno == logging.WARNING and "day_2026-03-14" in r.getMessage() for r in caplog.records
    )


def test_none_stages_base_marks_every_day_inconclusive(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    with patch(FINGERPRINT, return_value=FP):
        plan = _emit_publish_plan_sidecar(
            output_base=out, stages_base=None, **_kwargs(["2026-03-14"])
        )
    assert "inconclusive" in _hold(plan)


def test_two_days_one_unattested_holds(tmp_path):
    out, stages = tmp_path / "out", tmp_path / "stages"
    out.mkdir()
    _stage_attestation(stages, "2026-03-14")
    plan = _run(out, stages, ["2026-03-14", "2026-03-15"])
    assert len(_att_files(out)) == 2
    assert _hold(plan) is not None


def test_stale_attestation_from_another_day_is_removed(tmp_path):
    out, stages = tmp_path / "out", tmp_path / "stages"
    out.mkdir()
    (out / "day_2020-01-01_privacy_attestation.json").write_text(
        json.dumps(_doc("2020-01-01")), encoding="utf-8"
    )
    _stage_attestation(stages, "2026-03-14")
    _run(out, stages, ["2026-03-14"])
    assert _att_files(out) == ["day_2026-03-14_privacy_attestation.json"]


def test_drvfs_copy_refuses_metadata_still_succeeds(tmp_path):
    out, stages = tmp_path / "out", tmp_path / "stages"
    out.mkdir()
    src = _stage_attestation(stages, "2026-03-14")

    def _deny(*a, **k):
        raise PermissionError("Operation not permitted")

    with patch.object(shutil, "copystat", _deny), patch.object(os, "utime", _deny):
        plan = _run(out, stages, ["2026-03-14"])
    assert (out / "day_2026-03-14_privacy_attestation.json").read_bytes() == src.read_bytes()
    assert _hold(plan) is None


def test_no_first_party_route_returns_none_and_writes_nothing(tmp_path):
    out, stages = tmp_path / "out", tmp_path / "stages"
    out.mkdir()
    _stage_attestation(stages, "2026-03-14")
    assert _run(out, stages, ["2026-03-14"], route=False) is None
    assert _att_files(out) == []


def test_asset_bundle_attestation_beside_the_plan_is_kept(tmp_path):
    # enqueue-asset's attestation matches the dispatch glob but names no day:
    # deleting it would let the asset's rows merge to the drive's verified days.
    out, stages = tmp_path / "out", tmp_path / "stages"
    out.mkdir()
    asset = out / ASSET_ATTESTATION_FILENAME
    asset.write_text(json.dumps(_doc("asset", "inconclusive")), encoding="utf-8")
    _stage_attestation(stages, "2026-03-14")
    plan = _run(out, stages, ["2026-03-14"])
    assert asset.is_file()
    assert "inconclusive" in _hold(plan)
