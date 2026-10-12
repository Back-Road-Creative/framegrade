"""``gomoveshift enqueue-asset`` (cli/handlers/enqueue_asset.py)."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers.enqueue_asset import cmd_enqueue_asset
from gomoveshift.publishing.asset_plan import review_hash
from gomoveshift.publishing.queue import open_queue_db


@pytest.fixture(autouse=True)
def _fake_ffprobe(monkeypatch):
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"duration": "9.0"},
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
    path.write_bytes(b"route story bytes")
    return path


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _args(video, **over):
    fields = {
        "video": str(video),
        "platform": ["youtube"],
        "at": "2026-09-05T14:00:00+00:00",
        "display_timezone": "UTC",
        "metadata": None,
        "title": "A Drive Through the Hills",
        "description": "Scenic loop.",
        "tag": ["ireland"],
        "receipt": None,
        "bundle_dir": None,
        "confirm": None,
        "apply": False,
        "shadow": False,
    }
    fields.update(over)
    return SimpleNamespace(**fields)


def _sha256_of(video) -> str:
    import hashlib

    return hashlib.sha256(video.read_bytes()).hexdigest()


def test_dry_run_prints_hash_and_writes_nothing(tmp_path, video, capsys):
    rc = cmd_enqueue_asset(_args(video), _config(tmp_path))

    assert rc == 0
    out = capsys.readouterr().out
    assert "review hash:" in out
    assert "DRY RUN" in out
    assert list(video.parent.glob("*_publish_plan.json")) == []
    assert list(video.parent.glob("day_*_privacy_attestation.json")) == []


def test_apply_without_matching_confirm_refuses(tmp_path, video):
    rc = cmd_enqueue_asset(_args(video, apply=True, confirm="not-the-hash"), _config(tmp_path))

    assert rc == 2
    assert list(video.parent.glob("*_publish_plan.json")) == []


def test_apply_with_matching_confirm_spools_one_row_per_platform(tmp_path, video):
    expected_hash = review_hash(
        "A Drive Through the Hills", "Scenic loop.", ["ireland"], _sha256_of(video)
    )

    rc = cmd_enqueue_asset(
        _args(
            video,
            platform=["youtube", "facebook"],
            apply=True,
            confirm=expected_hash,
            bundle_dir=str(tmp_path / "bundle"),
        ),
        _config(tmp_path),
    )

    assert rc == 0
    conn = open_queue_db(tmp_path)
    try:
        rows = conn.execute("SELECT platform FROM deliveries ORDER BY platform").fetchall()
    finally:
        conn.close()
    assert sorted(r["platform"] for r in rows) == ["facebook", "youtube"]


def test_apply_with_no_receipt_spools_but_privacy_attestation_holds(tmp_path, video):
    from gomoveshift.publishing import dispatch
    from gomoveshift.publishing.asset_plan import ASSET_ATTESTATION_FILENAME

    expected_hash = review_hash(
        "A Drive Through the Hills", "Scenic loop.", ["ireland"], _sha256_of(video)
    )
    bundle_dir = tmp_path / "bundle"

    rc = cmd_enqueue_asset(
        _args(video, apply=True, confirm=expected_hash, bundle_dir=str(bundle_dir)),
        _config(tmp_path),
    )

    assert rc == 0
    attestation = json.loads((bundle_dir / ASSET_ATTESTATION_FILENAME).read_text())
    merged = dispatch.merge_privacy_attestations([attestation])
    assert dispatch.privacy_hold_reason(merged) is not None
