"""V2: phase A measures each source's sharpness and refuses a soft one before merge.

One metric (``quality.sharpness_score``) is shared by the thumbnail probe, the
photo cull and this gate; the source's median over N sampled frames is stored in
``drive_scores.npz`` as ``source_sharpness`` (one entry per source, file order).
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from test_phase_a_score_cache import (
    _episode,
    _make_episode_files,
    _patch_phase_a,
    _SpyScorer,
    _state,
)

from gomoveshift.config import Config
from gomoveshift.pipeline import quality
from gomoveshift.pipeline.drive_phases import phase_a as phase_a_mod
from gomoveshift.pipeline.drive_phases.phase_a import PhaseA

SHARP = [900.0, 950.0, 1000.0, 980.0, 920.0]
SOFT = [4.0, 5.0, 6.0, 5.5, 4.5]


def _setup(tmp_path, monkeypatch, per_file):
    """Two fake sources; ``per_file`` maps file name -> the sampled sharpness list."""
    stages = tmp_path / "stages"
    stages.mkdir()
    files = _make_episode_files(tmp_path, n=2)
    _patch_phase_a(monkeypatch, [_episode(i, f) for i, f in enumerate(files)])
    monkeypatch.setattr(
        phase_a_mod,
        "_sample_source_sharpness",
        lambda path, duration: list(per_file[Path(path).name]),
    )
    return stages, files


def test_phase_a_refuses_soft_source_with_measurement(tmp_path, monkeypatch):
    stages, files = _setup(tmp_path, monkeypatch, {"ep_0.mp4": SHARP, "ep_1.mp4": SOFT})
    merged = []
    orig = _SpyScorer.merge_file_scores
    monkeypatch.setattr(
        _SpyScorer,
        "merge_file_scores",
        lambda self, fs, fd: merged.append(1) or orig(self, fs, fd),
    )
    with pytest.raises(RuntimeError) as exc:
        PhaseA(Config()).run(_state(tmp_path, stages_base=stages))
    msg = str(exc.value)
    assert "ep_1.mp4" in msg and "ep_0.mp4" not in msg
    assert f"{np.median(SOFT):.1f}" in msg
    assert f"{quality.SOURCE_SHARPNESS_FLOOR:.1f}" in msg
    assert "4.0" in msg and "4.5" in msg  # per-sample values
    assert merged == [], "merge must not run on a refused source"


def test_sharp_sources_record_measurement(tmp_path, monkeypatch):
    other = [700.0] * 5
    stages, _ = _setup(tmp_path, monkeypatch, {"ep_0.mp4": SHARP, "ep_1.mp4": other})
    state = _state(tmp_path, stages_base=stages)
    PhaseA(Config()).run(state)
    expected = [float(np.median(SHARP)), 700.0]
    assert list(state.source_sharpness) == expected
    with np.load(stages / "drive_scores.npz") as npz:
        assert npz["source_sharpness"].shape == (2,)
        assert list(npz["source_sharpness"]) == expected


def test_cached_npz_without_source_sharpness_is_rescored(tmp_path, monkeypatch, caplog):
    stages, files = _setup(tmp_path, monkeypatch, {"ep_0.mp4": SHARP, "ep_1.mp4": SHARP})
    PhaseA(Config()).run(_state(tmp_path, stages_base=stages))
    npz_path = stages / "drive_scores.npz"
    with np.load(npz_path) as npz:
        scores = npz["scores"]
    np.savez_compressed(npz_path, scores=scores)  # a pre-V2 cache: no source_sharpness

    monkeypatch.setattr(_SpyScorer, "score_file_calls", 0)
    with caplog.at_level("INFO"):
        PhaseA(Config()).run(_state(tmp_path, stages_base=stages))
    assert _SpyScorer.score_file_calls == len(files)
    assert "source_sharpness" in caplog.text


def test_one_sharpness_metric():
    root = Path(__file__).resolve().parent.parent / "gomoveshift" / "pipeline"
    owned = [
        root / "drive_phases" / "phase_a.py",
        root / "drive_phases" / "thumbnail_generator.py",
        root / "photos" / "cull.py",
    ]
    for path in owned:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "Laplacian":
                pytest.fail(
                    f"{path.name}:{node.lineno} calls a Laplacian; use quality.sharpness_score"
                )
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
                # hand-rolled 4-neighbour kernel: 4.0 * a[...] - a[...] ...
                if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Sub):
                    inner = node.left.left
                    if (
                        isinstance(inner, ast.BinOp)
                        and isinstance(inner.op, ast.Sub)
                        and isinstance(inner.left, ast.BinOp)
                        and isinstance(inner.left.op, ast.Mult)
                    ):
                        pytest.fail(f"{path.name}:{node.lineno} hand-rolls a Laplacian kernel")


def test_source_sharpness_round_trips_through_state_save_load(tmp_path):
    from gomoveshift.pipeline._drive_state import load_drive_state, save_drive_state
    from gomoveshift.pipeline.drive_state_model import PipelineState

    state = PipelineState(region="ireland", source_sharpness=[812.5, None, 64.0])
    path = tmp_path / "pipeline_state.json"
    save_drive_state(path, state, skip_invariants=True)
    assert load_drive_state(path).source_sharpness == [812.5, None, 64.0]
    assert PipelineState().source_sharpness == []
