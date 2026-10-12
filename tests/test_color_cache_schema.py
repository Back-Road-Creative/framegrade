"""Tests for color cache schema v2.

v1 cache files (with ``filter_str``) invalidate cleanly so the next run
re-analyzes. v2 cache files (with ``corrections``) round-trip through
the day_master_encoder stable-cache helpers.
"""

import json
from pathlib import Path

from gomoveshift.pipeline.cache import TRANSIENT_COLOR_CACHE_SCHEMA_VERSION
from gomoveshift.pipeline.color_correction import GRADE_ALGO_VERSION
from gomoveshift.pipeline.day_master_encoder import (
    COLOR_CACHE_SCHEMA_VERSION,
    _load_stable_color_cache,
    _preseed_transient_color_cache,
    _save_stable_color_cache,
    _source_fingerprints,
)


def _correction(**overrides) -> dict:
    base = dict(
        shadow_lift=5.0,
        highlight_pull=5.0,
        saturation_mult=1.05,
        contrast_curve=0.0,
        white_balance_r=0.0,
        white_balance_b=0.0,
        gamma=1.02,
        denoise_strength=0.0,
        sharpen_amount=0.0,
    )
    base.update(overrides)
    return base


def test_v1_cache_invalidates(tmp_path: Path) -> None:
    concat = tmp_path / "ep1.mp4"
    concat.write_bytes(b"x" * 100)
    fps = _source_fingerprints([concat])
    cache_path = tmp_path / "day_2026-05-19_color_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 1,
                "filter_str": "curves=m='0/0.02 1/0.97'",
                "method": "auto_global",
                "source_fingerprints": fps,
            }
        )
    )
    # v1 schema with matching fingerprints — should STILL invalidate.
    assert _load_stable_color_cache(tmp_path, "2026-05-19", [concat]) is None


def test_v2_cache_round_trips(tmp_path: Path) -> None:
    concat = tmp_path / "ep1.mp4"
    concat.write_bytes(b"y" * 200)
    raw_concat = tmp_path / "raw_concat.mp4"
    raw_concat.write_bytes(b"raw" * 50)

    # Simulate _analyze_color writing the current transient cache: merged-span
    # corrections + per-span durations.
    transient = raw_concat.parent / f"{raw_concat.stem}_color_analysis.json"
    transient.write_text(
        json.dumps(
            {
                "version": TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
                "corrections": [
                    _correction(shadow_lift=7.5, highlight_pull=6.0, saturation_mult=1.10),
                    _correction(shadow_lift=3.0, highlight_pull=12.0),
                ],
                "method": "auto_adaptive",
                "span_durations": [90.0, 60.0],
            }
        )
    )
    _save_stable_color_cache(tmp_path, "2026-05-19", raw_concat, [concat])
    loaded = _load_stable_color_cache(tmp_path, "2026-05-19", [concat])
    assert loaded is not None
    assert loaded["version"] == COLOR_CACHE_SCHEMA_VERSION
    assert loaded["grade_algo_version"] == GRADE_ALGO_VERSION
    assert loaded["method"] == "auto_adaptive"
    assert len(loaded["corrections"]) == 2
    assert loaded["corrections"][0]["shadow_lift"] == 7.5
    assert loaded["span_durations"] == [90.0, 60.0]


def test_fingerprint_mismatch_invalidates_v2(tmp_path: Path) -> None:
    concat = tmp_path / "ep1.mp4"
    concat.write_bytes(b"z" * 100)
    cache_path = tmp_path / "day_2026-05-19_color_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": 2,
                "corrections": [_correction()],
                "method": "auto_global",
                "segment_duration_sec": 0,
                "source_fingerprints": ["stale:fingerprint"],
            }
        )
    )
    assert _load_stable_color_cache(tmp_path, "2026-05-19", [concat]) is None


def test_preseed_writes_v2_transient_cache(tmp_path: Path) -> None:
    raw_concat = tmp_path / "raw.mp4"
    raw_concat.write_bytes(b"raw" * 10)
    payload = {
        "version": 2,
        "corrections": [_correction()],
        "method": "auto_global",
        "segment_duration_sec": 30,
    }
    _preseed_transient_color_cache(raw_concat, payload)
    transient_path = raw_concat.parent / f"{raw_concat.stem}_color_analysis.json"
    loaded = json.loads(transient_path.read_text())
    # Preseed must stamp the transient schema version the _analyze_color reader
    # accepts (v2), NOT the stable day-cache's COLOR_CACHE_SCHEMA_VERSION (v3) —
    # else resume re-runs the 35–90 min color analysis. The transient
    # schema carries no grade_algo_version (matching what _analyze_color writes).
    assert loaded["version"] == TRANSIENT_COLOR_CACHE_SCHEMA_VERSION
    assert "grade_algo_version" not in loaded
    assert loaded["method"] == "auto_global"
    assert loaded["corrections"] == payload["corrections"]
    # stores corrections, NOT filter_str (regenerated at use-time)
    assert "filter_str" not in loaded


def test_grade_algo_version_mismatch_invalidates(tmp_path: Path) -> None:
    """A cache written by a DIFFERENT grade algorithm — same schema version,
    matching fingerprints — must invalidate. This is the stale-dark-grade
    replay vector: a contrast_curve=0 fix once shipped without a schema
    bump, so an old cache would otherwise re-apply the dark S-curve on a
    same-source re-run."""
    concat = tmp_path / "ep1.mp4"
    concat.write_bytes(b"q" * 100)
    fps = _source_fingerprints([concat])
    cache_path = tmp_path / "day_2026-05-19_color_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "version": COLOR_CACHE_SCHEMA_VERSION,
                "grade_algo_version": "stale-algo-00",  # not the current algo hash
                "corrections": [_correction()],
                "method": "auto_global",
                "segment_duration_sec": 0,
                "source_fingerprints": fps,
            }
        )
    )
    assert _load_stable_color_cache(tmp_path, "2026-05-19", [concat]) is None


def test_hdr_day_tonemap_round_trips_stable_cache_and_preseed(tmp_path: Path) -> None:
    """An HDR day's stable cache must carry the tonemap chain through the
    preseed, so the transient reader (keyed on the chain) hits on resume
    instead of re-running the 35-90 min analysis."""
    chain = "zscale=t=linear:npl=100,tonemap=hable,zscale=t=bt709:m=bt709"
    concat = tmp_path / "ep1.mp4"
    concat.write_bytes(b"y" * 200)
    raw_concat = tmp_path / "raw_concat.mp4"
    raw_concat.write_bytes(b"raw" * 50)
    transient = raw_concat.parent / f"{raw_concat.stem}_color_analysis.json"
    transient.write_text(
        json.dumps(
            {
                "version": TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
                "corrections": [_correction()],
                "method": "auto_adaptive",
                "span_durations": [90.0],
                "tonemap": chain,
            }
        )
    )
    _save_stable_color_cache(tmp_path, "2026-05-19", raw_concat, [concat])
    loaded = _load_stable_color_cache(tmp_path, "2026-05-19", [concat])
    assert loaded is not None
    assert loaded["tonemap"] == chain

    transient.unlink()
    _preseed_transient_color_cache(raw_concat, loaded)
    assert json.loads(transient.read_text())["tonemap"] == chain
