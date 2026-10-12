"""Stale-artifact reuse gates.

A resumed / re-run drive must NOT ship an artifact rendered before a rule
change (parked footage / audible audio / unblurred faces). These close every
reuse path so a stale artifact is re-derived, invalidating ONLY when the
relevant inputs changed:

  * LF/short render fingerprints fold in segment windows + audio policy /
    the clip window, so a re-plan or audio-rule change re-renders.
  * the cached day master carries an encode-algo version; a pre-version or
    partial-metadata record is refused (re-encode) rather than reused.
  * phase_a/phase_bc code-version invalidation, the score_inputs resume
    re-validation, the fingerprint-sidecar recovery pairing, and the
    selector-version analysis cache.
  * the run lock is keyed to project_dir (dual-lock) so two runs with a
    different GMS_INTERMEDIATE_DIR still contend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ── LF render fingerprint (segment windows + audio policy) ───────────


class _Seg:
    def __init__(self, sfi: int, start: float, end: float) -> None:
        self.source_file_index = sfi
        self.start_time = start
        self.end_time = end


def test_lf_fingerprint_stable_for_unchanged_inputs():
    from gomoveshift.pipeline.day_pipeline import _lf_ost_fingerprint

    segs = [_Seg(0, 10.0, 40.0), _Seg(1, 5.0, 20.0)]
    a = _lf_ost_fingerprint("Cork, Ireland", 15.0, 0.5, segs)
    b = _lf_ost_fingerprint("Cork, Ireland", 15.0, 0.5, [_Seg(0, 10.0, 40.0), _Seg(1, 5.0, 20.0)])
    assert a == b


def test_lf_fingerprint_changes_when_segment_window_changes():
    from gomoveshift.pipeline.day_pipeline import _lf_ost_fingerprint

    base = _lf_ost_fingerprint("Cork", 15.0, 0.5, [_Seg(0, 10.0, 40.0)])
    # A stop mask trims the parked tail: same rank/text, different window.
    moved = _lf_ost_fingerprint("Cork", 15.0, 0.5, [_Seg(0, 10.0, 32.0)])
    assert base != moved


def test_lf_fingerprint_changes_when_audio_ceiling_changes(monkeypatch):
    from gomoveshift.pipeline import encoding_constants as ec
    from gomoveshift.pipeline.day_pipeline import _lf_ost_fingerprint

    segs = [_Seg(0, 10.0, 40.0)]
    before = _lf_ost_fingerprint("Cork", 15.0, 0.5, segs)
    monkeypatch.setattr(ec, "AUDIO_SILENCE_CEILING_LUFS", -20.0)  # the pre-fix boost target
    after = _lf_ost_fingerprint("Cork", 15.0, 0.5, segs)
    assert before != after


def test_lf_fingerprint_changes_when_audio_algo_version_bumps(monkeypatch):
    from gomoveshift.pipeline import encoding_constants as ec
    from gomoveshift.pipeline.day_pipeline import _lf_ost_fingerprint

    segs = [_Seg(0, 10.0, 40.0)]
    before = _lf_ost_fingerprint("Cork", 15.0, 0.5, segs)
    monkeypatch.setattr(ec, "AUDIO_FILTER_ALGO_VERSION", ec.AUDIO_FILTER_ALGO_VERSION + 1)
    after = _lf_ost_fingerprint("Cork", 15.0, 0.5, segs)
    assert before != after


# ── short render fingerprint (clip window) ───────────────────────────


def test_short_fingerprint_stable_for_unchanged_window():
    from gomoveshift.pipeline.shorts_extractor import _short_ost_fingerprint

    a = _short_ost_fingerprint("Healy Pass", "Subscribe", 12.0, 42.0)
    b = _short_ost_fingerprint("Healy Pass", "Subscribe", 12.0, 42.0)
    assert a == b


def test_short_fingerprint_changes_when_window_changes():
    from gomoveshift.pipeline.shorts_extractor import _short_ost_fingerprint

    base = _short_ost_fingerprint("Healy Pass", "Subscribe", 12.0, 42.0)
    # Same rank/text but a re-planned (possibly parked) window.
    moved = _short_ost_fingerprint("Healy Pass", "Subscribe", 100.0, 130.0)
    assert base != moved


# ── cached-master encode-algo version + partial-metadata hard-fail ───


def _master_and_record(tmp_path: Path):
    from gomoveshift.pipeline.day_master_encoder import (
        MASTER_ENCODE_ALGO_VERSION,
        master_fingerprint,
    )

    master = tmp_path / "day_2026-04-18_master.mp4"
    master.write_bytes(b"M" * 4096)
    record = {
        "master_fingerprint": master_fingerprint(master),
        "master_duration": 1800.0,
        "encode_algo_version": MASTER_ENCODE_ALGO_VERSION,
    }
    return master, record


def test_verify_cached_master_current_version_validates(tmp_path):
    from gomoveshift.pipeline.day_master_encoder import verify_cached_master

    master, record = _master_and_record(tmp_path)
    assert verify_cached_master(master, record, probe_fn=lambda p: 1800.0) is True


def test_verify_cached_master_missing_version_refuses(tmp_path):
    from gomoveshift.pipeline.day_master_encoder import (
        BatchDurationIntegrityError,
        verify_cached_master,
    )

    master, record = _master_and_record(tmp_path)
    record.pop("encode_algo_version")  # a master encoded before versioning
    with pytest.raises(BatchDurationIntegrityError, match="encode-algo version"):
        verify_cached_master(master, record, probe_fn=lambda p: 1800.0)


def test_verify_cached_master_older_version_refuses(tmp_path):
    from gomoveshift.pipeline.day_master_encoder import (
        BatchDurationIntegrityError,
        verify_cached_master,
    )

    master, record = _master_and_record(tmp_path)
    record["encode_algo_version"] = 1  # the pre-fingerprint hand-bumped integer
    with pytest.raises(BatchDurationIntegrityError, match="encode-algo version"):
        verify_cached_master(master, record, probe_fn=lambda p: 1800.0)


# ── the encode-algo version is DERIVED from the frozen encode constants ──


@pytest.mark.parametrize(
    "const, new_value",
    [
        ("LONGFORM_NVENC_CQ", 21),
        ("LONGFORM_NVENC_PRESET", "p5"),
        ("LONGFORM_PROFILE", "main"),
        ("LONGFORM_PIX_FMT", "yuv420p"),
        ("LONGFORM_MAXRATE_RATIO", 1.3),
        ("LONGFORM_MIN_BITRATE_4K", 30_000),
        ("AUDIO_SILENCE_CEILING_LUFS", -20.0),
    ],
)
def test_master_encode_version_moves_when_an_encode_constant_moves(monkeypatch, const, new_value):
    """A hand-bumped integer let a CQ/preset/profile/maxrate change reuse a
    stale cached master on resume. The version must be derived from the frozen
    constants the master encode actually consumes, so the bump is automatic."""
    from gomoveshift.pipeline import day_master_encoder as dme
    from gomoveshift.pipeline import encoding_constants as ec

    before = dme._master_encode_algo_version()
    monkeypatch.setattr(ec, const, new_value)
    assert dme._master_encode_algo_version() != before


def test_master_encode_version_ignores_shorts_only_constants(monkeypatch):
    """The day master carries no shorts parameters — re-deriving the shorts
    bitrate ladder must not force a ~6 h master re-encode on resume."""
    from gomoveshift.pipeline import day_master_encoder as dme
    from gomoveshift.pipeline import encoding_constants as ec

    before = dme._master_encode_algo_version()
    monkeypatch.setattr(ec, "SHORTS_MAXRATE", "20M")
    monkeypatch.setattr(ec, "SHORTS_BUFSIZE", "30M")
    assert dme._master_encode_algo_version() == before


def test_master_encode_version_is_stable_across_calls():
    from gomoveshift.pipeline import day_master_encoder as dme

    assert dme._master_encode_algo_version() == dme._master_encode_algo_version()
    assert dme.MASTER_ENCODE_ALGO_VERSION == dme._master_encode_algo_version()


def test_verify_cached_master_partial_metadata_hard_fails(tmp_path):
    """Fingerprint present but master_duration < 1.0 must NOT early-return
    True (which would skip the fingerprint + version checks) — it hard-fails."""
    from gomoveshift.pipeline.day_master_encoder import (
        BatchDurationIntegrityError,
        verify_cached_master,
    )

    master, record = _master_and_record(tmp_path)
    record["master_duration"] = 0.5  # encode-time ffprobe failure
    with pytest.raises(BatchDurationIntegrityError, match="PARTIAL"):
        verify_cached_master(master, record, probe_fn=lambda p: 1800.0)


def test_verify_cached_master_pre_fingerprint_record_still_falls_back(tmp_path):
    """A record with NO fingerprint at all (pre-fingerprint) still returns True so the
    caller falls back to the raw-episode path — unchanged behaviour."""
    from gomoveshift.pipeline.day_master_encoder import verify_cached_master

    master, _ = _master_and_record(tmp_path)
    assert verify_cached_master(master, {"master_sha": "x"}) is True


# ── phase code-version invalidation ──────────────────────────────────


def test_persist_phase_code_version_stamps_state():
    from gomoveshift.pipeline.drive_state_model import PipelineState
    from gomoveshift.pipeline.orchestrator_fingerprint import (
        PHASE_CODE_VERSIONS,
        persist_phase_code_version,
    )

    state = PipelineState()
    persist_phase_code_version(state, "phase_bc_day_first")
    assert state.phase_code_versions == {
        "phase_bc_day_first": PHASE_CODE_VERSIONS["phase_bc_day_first"]
    }
    # A phase with no tracked version is a no-op.
    persist_phase_code_version(state, "phase_b_preview")
    assert "phase_b_preview" not in (state.phase_code_versions or {})


def test_invalidate_phases_drops_stamped_version_mismatch():
    from gomoveshift.pipeline.orchestrator_fingerprint import (
        PHASE_CODE_VERSIONS,
        invalidate_phases_if_code_version_stale,
    )

    saved = {
        "phase_code_versions": {
            "phase_a": PHASE_CODE_VERSIONS["phase_a"],
            "phase_bc_day_first": 999,
        }
    }
    kept = invalidate_phases_if_code_version_stale(
        saved, ["phase_a", "phase_bc_day_first", "phase_d"]
    )
    # phase_bc stamped 999 != current → dropped; phase_a matches; phase_d untracked.
    assert kept == ["phase_a", "phase_d"]


def test_invalidate_phases_lenient_on_missing_stamp():
    from gomoveshift.pipeline.orchestrator_fingerprint import (
        invalidate_phases_if_code_version_stale,
    )

    # No phase_code_versions at all (pre-versioning checkpoint) → keep as-is.
    assert invalidate_phases_if_code_version_stale({}, ["phase_a", "phase_bc_day_first"]) == [
        "phase_a",
        "phase_bc_day_first",
    ]


# ── resume-path integration (code version + score_inputs) ────────────


def _make_source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "source"
    src.mkdir()
    (src / "CLIP_0001.MP4").write_bytes(b"")
    return src


def _episode_files(tmp_path: Path) -> list[str]:
    eps = []
    for i in range(2):
        p = tmp_path / f"ep_{i}.mp4"
        p.write_bytes(b"x" * (100 + i))
        eps.append(str(p))
    return eps


def _base_state_dict(source_dir: Path, output_base: Path, episode_files: list[str]) -> dict:
    return {
        "source_dir": str(source_dir),
        "output_base": str(output_base),
        "region": "ireland",
        "publish_day": "wed",
        "location": "Test Route",
        "phases_completed": ["phase_a"],
        "episode_files": episode_files,
        "gpx_files": [],
        "episode_dates": ["2026-03-24", "2026-03-25"],
        "file_durations": [300.0, 400.0],
        "file_boundaries": [[0.0, 300.0], [300.0, 700.0]],
        "run_start": "2026-04-19T08:00:00+00:00",
    }


def _write_scores(path: Path, metadata: dict) -> None:
    from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

    path.parent.mkdir(parents=True, exist_ok=True)
    scorer = UnifiedSegmentScorer(weights={"motion_energy": 1.0})
    scorer.save_scores(np.zeros((5, 9), dtype=np.float32), path, metadata)


def _orch_resume(tmp_path: Path, state_body: dict, *, extra_stage_files: dict) -> list[str]:
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
    from gomoveshift.pipeline.drive_state_model import PipelineState

    source_dir = Path(state_body["source_dir"])
    output_base = Path(state_body["output_base"])
    output_base.mkdir(parents=True, exist_ok=True)
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    state_file = stages_dir / "pipeline_state.json"
    state_file.write_text(json.dumps(state_body))
    for name, writer in extra_stage_files.items():
        writer(stages_dir / name)

    orch = DrivePipelineOrchestrator(config=Config())
    state = PipelineState(source_dir=source_dir, output_base=output_base)
    return orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)


def test_resume_reruns_phase_a_when_score_inputs_changed(tmp_path):
    """A re-ingest since the checkpoint (episode size/mtime changed) makes
    the persisted score_inputs stale — phase_a must re-run, not reuse."""
    source_dir = _make_source_dir(tmp_path)
    output_base = tmp_path / "output"
    eps = _episode_files(tmp_path)
    body = _base_state_dict(source_dir, output_base, eps)

    def write_scores(p: Path) -> None:
        # Cached score_inputs points at a DIFFERENT prior ingest.
        _write_scores(p, {"score_inputs": [["/old/ep_0.mp4", 1, 1], ["/old/ep_1.mp4", 2, 2]]})

    resumed = _orch_resume(tmp_path, body, extra_stage_files={"drive_scores.npz": write_scores})
    assert "phase_a" not in resumed


def test_resume_keeps_phase_a_when_score_inputs_match(tmp_path):
    """Unchanged episodes → the score_inputs key matches → phase_a stays reused."""
    from gomoveshift.pipeline.drive_phases.phase_a import _score_inputs_key

    source_dir = _make_source_dir(tmp_path)
    output_base = tmp_path / "output"
    eps = _episode_files(tmp_path)
    body = _base_state_dict(source_dir, output_base, eps)
    current_key = _score_inputs_key(eps)

    def write_scores(p: Path) -> None:
        _write_scores(p, {"score_inputs": current_key})

    resumed = _orch_resume(tmp_path, body, extra_stage_files={"drive_scores.npz": write_scores})
    assert "phase_a" in resumed


def test_resume_drops_phase_bc_on_code_version_bump(tmp_path):
    source_dir = _make_source_dir(tmp_path)
    output_base = tmp_path / "output"
    eps = _episode_files(tmp_path)
    body = _base_state_dict(source_dir, output_base, eps)
    body["phases_completed"] = ["phase_a", "phase_bc_day_first"]
    body["phase_code_versions"] = {"phase_bc_day_first": 999}  # stale

    resumed = _orch_resume(
        tmp_path,
        body,
        extra_stage_files={
            "drive_scores.npz": lambda p: _write_scores(p, {"num_files": 2}),
            "drive_plan.json": lambda p: p.write_text("{}"),
        },
    )
    assert "phase_bc_day_first" not in resumed
    assert "phase_a" in resumed


# ── recovery pairing uses the render-fingerprint sidecar, not a glob ──


def test_recovery_pairing_requires_fingerprint_sidecar(tmp_path):
    from gomoveshift.pipeline.day_pipeline import _maybe_fire_recovery_callback

    date = "2026-02-20"
    day_dir = tmp_path / f"day_{date}"
    lf_dir = day_dir / "regular_lfs"
    sh_dir = day_dir / "shorts"
    lf_dir.mkdir(parents=True)
    sh_dir.mkdir()
    # lf_03 is a current-code render (carries its .ost.sha256 sidecar);
    # lf_07 is a stale pre-fix render globbed into the dir with no sidecar.
    (lf_dir / "lf_03.mp4").write_bytes(b"lf")
    (lf_dir / "lf_03.ost.sha256").write_text("cafef00d")
    (lf_dir / "lf_07.mp4").write_bytes(b"stale")
    (sh_dir / "short_01.mp4").write_bytes(b"short")  # a final present → recovery fires
    # No master on disk → the recovery marker is written.

    captured = []
    _maybe_fire_recovery_callback(
        date, day_dir, state={}, on_day_complete=lambda art, st: captured.append(art)
    )
    assert len(captured) == 1
    names = [p.name for p in captured[0].regular_lf_paths]
    assert names == ["lf_03.mp4"]  # lf_07 (no sidecar) excluded


# ── selector-version analysis cache ──────────────────────────────────


def _seed_analysis(tmp_path: Path, *, version) -> tuple[Path, Path]:
    src = tmp_path / "master.mp4"
    src.write_bytes(b"v" * 64)
    cache = tmp_path / "shorts_analysis.json"
    payload = {"num_shorts_found": 3, "shorts": []}
    if version is not None:
        payload["selector_version"] = version
    cache.write_text(json.dumps(payload))
    return src, cache


def test_is_analysis_cached_hits_on_matching_selector_version(tmp_path):
    from gomoveshift.analysis.shorts_models import SHORTS_SELECTOR_VERSION
    from gomoveshift.pipeline.cache import is_analysis_cached

    src, cache = _seed_analysis(tmp_path, version=SHORTS_SELECTOR_VERSION)
    assert is_analysis_cached(src, cache, expected_version=SHORTS_SELECTOR_VERSION) is True


def test_is_analysis_cached_misses_on_selector_version_change(tmp_path):
    from gomoveshift.analysis.shorts_models import SHORTS_SELECTOR_VERSION
    from gomoveshift.pipeline.cache import is_analysis_cached

    src, cache = _seed_analysis(tmp_path, version=SHORTS_SELECTOR_VERSION - 1)
    assert is_analysis_cached(src, cache, expected_version=SHORTS_SELECTOR_VERSION) is False


def test_is_analysis_cached_versioned_does_not_trust_missing_source(tmp_path):
    from gomoveshift.analysis.shorts_models import SHORTS_SELECTOR_VERSION
    from gomoveshift.pipeline.cache import is_analysis_cached

    src, cache = _seed_analysis(tmp_path, version=SHORTS_SELECTOR_VERSION)
    src.unlink()  # source purged — must NOT trust the cache in the versioned path
    assert is_analysis_cached(src, cache, expected_version=SHORTS_SELECTOR_VERSION) is False


def test_is_analysis_cached_legacy_path_unchanged(tmp_path):
    """expected_version=None keeps the legacy mtime+size freshness (incl. the
    source-missing trust the color-analysis cache relies on)."""
    from gomoveshift.pipeline.cache import is_analysis_cached

    src, cache = _seed_analysis(tmp_path, version=None)
    src.unlink()  # legacy path DOES trust a cached output when the source is gone
    assert is_analysis_cached(src, cache) is True


# ── dual run lock keyed to project_dir ───────────────────────────────


def test_dual_lock_contends_across_differing_stages_base(tmp_path):
    """Two runs on the same project_dir but different GMS_INTERMEDIATE_DIR
    (stages_base) still contend for the ONE project-dir lock."""
    from gomoveshift.pipeline.orchestrator_lock import (
        ConcurrentRunError,
        _acquire_run_locks,
        _release_run_lock,
    )

    project = tmp_path / "proj"
    output = tmp_path / "out"
    stages_a = tmp_path / "ssd_a"
    stages_b = tmp_path / "ssd_b"
    for d in (project, output, stages_a, stages_b):
        d.mkdir()

    stage_fd, stage_path, proj_fd, proj_path = _acquire_run_locks(project, stages_a, output)
    try:
        with pytest.raises(ConcurrentRunError):
            _acquire_run_locks(project, stages_b, output)
    finally:
        _release_run_lock(stage_fd, stage_path)
        _release_run_lock(proj_fd, proj_path)

    # After release, a fresh dual-acquire (either staging) succeeds.
    s2, sp2, p2, pp2 = _acquire_run_locks(project, stages_b, output)
    _release_run_lock(s2, sp2)
    _release_run_lock(p2, pp2)
