"""Standalone-asset enqueue bundle (``publishing/asset_plan.py``).

``subprocess.run`` is faked to return a valid ffprobe probe (matching
``tests/test_media_fingerprint.py``) so no real ffmpeg is required.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from gomoveshift.publishing import dispatch
from gomoveshift.publishing.asset_plan import (
    ASSET_ATTESTATION_FILENAME,
    build_asset_sidecar,
    review_hash,
    write_asset_bundle,
)
from gomoveshift.cli.handlers.dispatch_queue import PrivacyAttestationLoader
from gomoveshift.publishing.plan_io import load_publish_plan


@pytest.fixture(autouse=True)
def _fake_ffprobe(monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"duration": "12.5"},
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1080,
                            "height": 1920,
                        }
                    ],
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "route.mp4"
    path.write_bytes(b"route-story film bytes")
    return path


def _build(video, **over):
    kwargs = dict(
        title="Driving the Wicklow Way",
        description="A scenic loop through the mountains.",
        tags=("ireland", "driving"),
        recording_date="2026-09-01",
        platforms=("youtube",),
        scheduled_at_utc="2026-09-05T14:00:00+00:00",
    )
    kwargs.update(over)
    return build_asset_sidecar(video, **kwargs)


def test_sidecar_round_trips_through_load_publish_plan(tmp_path, video):
    doc = _build(video)
    sidecar_path, _ = write_asset_bundle(tmp_path, doc, None)

    plan = load_publish_plan(sidecar_path)

    assert len(plan.shorts) == 1
    record = plan.shorts[0]
    assert record.payloads.youtube.title == "Driving the Wicklow Way"
    assert record.routes.selected == ("youtube",)
    assert record.media.sha256


def test_review_hash_changes_when_title_changes(video):
    doc_a = _build(video, title="Title A")
    doc_b = _build(video, title="Title B")

    assert doc_a["shorts"][0]["reviewed_plan_hash"] != doc_b["shorts"][0]["reviewed_plan_hash"]


def test_review_hash_changes_when_file_changes(tmp_path, video):
    doc_a = _build(video)
    other = tmp_path / "other.mp4"
    other.write_bytes(b"a totally different file")
    doc_b = _build(other)

    assert doc_a["shorts"][0]["reviewed_plan_hash"] != doc_b["shorts"][0]["reviewed_plan_hash"]


def test_review_hash_is_pure_and_deterministic():
    h1 = review_hash("t", "d", ["a", "b"], "sha")
    h2 = review_hash("t", "d", ["a", "b"], "sha")
    assert h1 == h2


def test_missing_attestation_holds_every_required_class(tmp_path, video):
    doc = _build(video)
    _, attestation_path = write_asset_bundle(tmp_path, doc, None)

    written = json.loads(attestation_path.read_text(encoding="utf-8"))
    for name in dispatch.PRIVACY_REQUIRED_CLASSES:
        assert written["classes"][name]["state"] == dispatch.PRIVACY_INCONCLUSIVE

    merged = dispatch.merge_privacy_attestations([written])
    assert dispatch.privacy_hold_reason(merged) is not None


def test_receipt_attestation_with_all_verified_clears_the_hold(tmp_path, video):
    doc = _build(video)
    receipt_privacy = {
        "classes": {
            name: {"state": dispatch.PRIVACY_VERIFIED} for name in dispatch.PRIVACY_REQUIRED_CLASSES
        },
        "withheld": [],
    }
    _, attestation_path = write_asset_bundle(tmp_path, doc, receipt_privacy)

    written = json.loads(attestation_path.read_text(encoding="utf-8"))
    merged = dispatch.merge_privacy_attestations([written])
    assert dispatch.privacy_hold_reason(merged) is None
    assert attestation_path.name == ASSET_ATTESTATION_FILENAME


def test_privacy_attestation_loader_reads_the_bundle_dir(tmp_path, video):
    doc = _build(video)
    sidecar_path, _ = write_asset_bundle(tmp_path, doc, None)

    loader = PrivacyAttestationLoader(tmp_path)
    row = {"sidecar_path": str(sidecar_path)}
    attestation = loader(row)

    assert dispatch.privacy_hold_reason(attestation) is not None


def test_unknown_platform_raises(video):
    with pytest.raises(ValueError, match="unknown platform"):
        _build(video, platforms=("myspace",))
