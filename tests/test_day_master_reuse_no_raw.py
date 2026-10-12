"""Cached day master reuse after the raw episode sources are purged.

The encoder persists the master's encode-time duration + fingerprint into
``per_day_completed[date]`` and validates a cached master against THAT when the
raw episodes are absent. Cases: (a) present → raw sum check runs; (b) absent +
match → validates without ffprobing absent episodes; (c) absent + mismatch →
RAISES; (+) legacy record → falls back to the raw path.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _blur_off(monkeypatch):
    """Cached-master-reuse MECHANICS, orthogonal to the privacy blur now default-ON in
    ``encode_day_master`` (its coverage lives in ``test_day_master_blur.py``). Force
    blur off so the mocked reuse path isn't gated by the ``.blurred`` marker."""
    monkeypatch.setattr("gomoveshift.pipeline.blur_gate.blur_required", lambda cfg: False)


def _make_day_batch(date: str, episode_paths: list, duration_s: float = 1800.0):
    from gomoveshift.pipeline.day_grouper import DayBatch

    episodes = []
    for i, p in enumerate(episode_paths):
        ep = MagicMock(index=i + 1, total_duration=duration_s, concat_path=p)
        episodes.append(ep)
    return DayBatch(date=date, episodes=episodes)


def _forbid_reencode(monkeypatch):
    """Re-encoding on the resume-skip path is a failure — the cached master
    must be returned untouched."""

    def boom(*a, **kw):
        raise AssertionError("re-encode (process/subprocess.run) must NOT run on resume skip")

    monkeypatch.setattr("gomoveshift.pipeline.single_pass.SinglePassProcessor.process", boom)
    monkeypatch.setattr("gomoveshift.pipeline.day_master_encoder.subprocess.run", boom)


def _patch_probe(monkeypatch, fn):
    monkeypatch.setattr(
        "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient.probe_duration", fn
    )


def _record(master_path: Path, batch, master_duration: float) -> dict:
    """Encode-time per_day_completed[date] record — built by the real
    ``build_master_record`` (probe stubbed to ``master_duration``)."""
    from gomoveshift.pipeline.day_master_encoder import build_master_record

    rec = build_master_record(batch, master_path, probe_fn=lambda p: master_duration)
    rec.update(master_path=str(master_path), master_sha="deadbeef")
    return rec


def _setup(tmp_path, *, episode_on_disk: bool):
    output_dir = tmp_path / "days_out"
    output_dir.mkdir(parents=True, exist_ok=True)
    ep = tmp_path / "ep_001.mp4"
    if episode_on_disk:
        ep.write_bytes(b"x" * 4096)
    batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)
    master = output_dir / "day_2026-04-18_master.mp4"
    master.write_bytes(b"M" * (16 * 1024))
    return output_dir, ep, batch, master


def _encode(batch, output_dir, record):
    from gomoveshift.config import Config
    from gomoveshift.pipeline.day_master_encoder import encode_day_master

    return encode_day_master(
        day_batch=batch, output_dir=output_dir, config=Config(), per_day_record=record
    )


# (a) Episodes present → raw-episode sum check still runs (no short-circuit).
def test_present_episodes_run_raw_sum_check(tmp_path, monkeypatch):
    output_dir, ep, batch, master = _setup(tmp_path, episode_on_disk=True)
    seen = []

    def probe(self, path):
        if Path(path) == ep:
            seen.append(path)
        return 1800.0

    _patch_probe(monkeypatch, probe)
    _forbid_reencode(monkeypatch)
    assert _encode(batch, output_dir, _record(master, batch, 1800.0)) == master
    assert seen, "raw-episode ffprobe sum check did not run"


# (b) Episodes absent + record match → validates WITHOUT probing absent episodes.
def test_absent_episodes_validate_against_record(tmp_path, monkeypatch):
    output_dir, _ep, batch, master = _setup(tmp_path, episode_on_disk=False)

    def probe(self, path):
        assert Path(path) == master, f"absent raw episode was ffprobed: {path!r}"
        return 1800.0

    _patch_probe(monkeypatch, probe)
    _forbid_reencode(monkeypatch)
    assert _encode(batch, output_dir, _record(master, batch, 1800.0)) == master


# (c) Episodes absent + duration OR fingerprint mismatch → RAISES.
@pytest.mark.parametrize(
    "kind, match",
    [("duration", "disagrees with recorded"), ("fingerprint", "fingerprint changed")],
)
def test_absent_episodes_mismatch_raises(tmp_path, monkeypatch, kind, match):
    from gomoveshift.pipeline.day_master_encoder import BatchDurationIntegrityError

    output_dir, _ep, batch, master = _setup(tmp_path, episode_on_disk=False)
    record = _record(master, batch, 1800.0)
    if kind == "duration":
        _patch_probe(monkeypatch, lambda self, path: 900.0)  # recorded 1800 → mismatch
    else:
        master.write_bytes(b"M" * (32 * 1024))  # rewrite → fingerprint no longer matches
        _patch_probe(monkeypatch, lambda self, path: 1800.0)
    _forbid_reencode(monkeypatch)
    with pytest.raises(BatchDurationIntegrityError, match=match):
        _encode(batch, output_dir, record)


# (+) Legacy record (no metadata) → verify_cached_master never raises (the
#     caller then falls back to the raw-episode path). Legacy back-compat unit.
def test_legacy_record_validates_as_fallback(tmp_path):
    from gomoveshift.pipeline.day_master_encoder import verify_cached_master

    _output_dir, _ep, _batch, master = _setup(tmp_path, episode_on_disk=False)
    assert verify_cached_master(master, {"master_path": str(master), "master_sha": "x"}) is True


# build_master_record records the day's raw source bytes so a resume's
# preflight disk gate can subtract a completed day from the peak estimate.
def test_build_master_record_captures_source_bytes(tmp_path):
    from gomoveshift.pipeline.day_grouper import DayBatch
    from gomoveshift.pipeline.day_master_encoder import build_master_record
    from gomoveshift.pipeline.ingest.gopro_parser import (
        Episode,
        GoProChapter,
        GoProRecording,
    )

    # A chapter-derived episode: chapter_files() are the raw source MP4s.
    c1 = tmp_path / "GX010001.MP4"
    c1.write_bytes(b"x" * 3000)
    c2 = tmp_path / "GX020001.MP4"
    c2.write_bytes(b"x" * 1000)
    rec = GoProRecording(recording_num=1)
    rec.chapters = [
        GoProChapter(path=c1, chapter_num=1, recording_num=1, duration=10.0),
        GoProChapter(path=c2, chapter_num=2, recording_num=1, duration=5.0),
    ]
    ep = Episode(index=1, recordings=[rec])
    batch = DayBatch(date="2026-04-18", episodes=[ep])
    master = tmp_path / "master.mp4"
    master.write_bytes(b"m" * 10)

    record = build_master_record(batch, master, probe_fn=lambda p: 15.0)
    assert record["source_bytes"] == 4000


# Fresh-encode duration integrity: build_master_record must COMPARE
# master_duration against expected_duration itself, not just record both for
# a later resume to check. Two-sided — both a SHORT and a LONG master raise.
@pytest.mark.parametrize("probed_duration", [900.0, 2200.0])
def test_build_master_record_raises_on_fresh_encode_mismatch(tmp_path, probed_duration):
    from gomoveshift.pipeline.day_master_encoder import (
        BatchDurationIntegrityError,
        build_master_record,
    )

    ep = tmp_path / "ep_001.mp4"
    ep.write_bytes(b"x" * 1024)
    batch = _make_day_batch("2026-04-18", [ep], duration_s=1800.0)
    master = tmp_path / "master.mp4"
    master.write_bytes(b"m" * 10)

    with pytest.raises(BatchDurationIntegrityError, match="freshly-encoded master"):
        build_master_record(batch, master, probe_fn=lambda p: probed_duration)


def test_completed_source_bytes_sums_per_day_records():
    from gomoveshift.pipeline.orchestrator_preflight import _completed_source_bytes

    state = {
        "per_day_completed": {
            "2026-04-18": {"source_bytes": 10, "master_path": "a"},
            "2026-04-19": {"source_bytes": 25, "master_path": "b"},
            # A legacy / recovery-marker record without source_bytes → 0.
            "2026-04-20": {"master_path": "c"},
        }
    }
    assert _completed_source_bytes(state) == 35.0
    assert _completed_source_bytes({}) == 0.0
    assert _completed_source_bytes({"per_day_completed": {}}) == 0.0
