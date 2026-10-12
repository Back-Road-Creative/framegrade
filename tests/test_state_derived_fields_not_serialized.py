"""``drive_scores`` and ``gps_per_episode`` are NOT serialized into
``pipeline_state.json``.

Both are large (numpy score matrix / full per-episode trackpoint traces),
rewritten on every checkpoint, and fully recoverable from on-disk artifacts:
``drive_scores`` from ``drive_scores.npz`` and ``gps_per_episode`` by
re-parsing the ``.gpx`` files whose paths live in ``state.gpx_files``.
Persisting them into the state JSON is pure I/O waste (tens of MB times
dozens of checkpoints).

Contract:
  * write side — ``PipelineState.to_dict`` omits both fields.
  * read side  — ``from_dict`` still reads a LEGACY checkpoint that carries
    them (backward compat), so an in-flight resume of an old run is safe.
  * resume     — ``gps_enrichment.rebuild_gps_per_episode`` reconstructs the
    per-episode trace from ``state.gpx_files`` (exercised end-to-end in
    ``test_state_roundtrip.test_gps_points_rebuilt_from_gpx_after_roundtrip``).
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline._drive_state import load_drive_state, save_drive_state
from gomoveshift.pipeline.drive_state_model import PipelineState


def _numpy_or_skip():
    return pytest.importorskip("numpy")


# ── write side: to_dict / on-disk JSON omit the two heavy fields ───────


def test_to_dict_omits_drive_scores_and_gps_per_episode():
    np = _numpy_or_skip()
    state = PipelineState()
    state["drive_scores"] = np.array([[1.0, 2.0], [3.0, 4.0]])
    state["gps_per_episode"] = [({"trackpoints": [{"lat": 1.0}]}, 0.0)]

    d = state.to_dict()

    assert "drive_scores" not in d, "drive_scores must not be serialized (recover from .npz)"
    assert "gps_per_episode" not in d, "gps_per_episode must not be serialized (recover from .gpx)"


def test_saved_state_json_carries_no_trackpoint_payload(tmp_path):
    """The on-disk JSON must contain neither a numpy score matrix nor a
    trackpoint blob — the two payloads that were pure I/O waste."""
    np = _numpy_or_skip()
    state_file = tmp_path / "pipeline_state.json"
    state = PipelineState(region="usa", publish_day="wed", phases_completed=["phase_a"])
    state["drive_scores"] = np.arange(90, dtype=np.float32).reshape(10, 9)
    state["gps_per_episode"] = [
        ({"trackpoints": [{"lat": 32.4, "lon": -80.7, "timestamp": 1.0}]}, 0.0)
    ]

    save_drive_state(state_file, state)
    raw = state_file.read_text()

    assert "drive_scores" not in raw
    assert "gps_per_episode" not in raw
    assert "trackpoints" not in raw
    assert "__numpy__" not in raw
    # sanity: real fields still round-trip
    loaded = load_drive_state(state_file)
    assert loaded.get("region") == "usa"
    assert loaded.get("phases_completed") == ["phase_a"]


# ── read side: legacy checkpoints carrying the fields still load ───────


def test_from_dict_still_reads_legacy_checkpoint_fields():
    """A checkpoint written before the exclusion carries both fields — ``from_dict`` must
    still rehydrate them so an in-flight resume of an old run is safe."""
    np = _numpy_or_skip()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    legacy = {
        "region": "usa",
        "drive_scores": {
            "__numpy__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.tolist(),
        },
        "gps_per_episode": [["gps_placeholder", 0.0], [None, 1.5]],
    }
    restored = PipelineState.from_dict(legacy)
    assert isinstance(restored.drive_scores, np.ndarray)
    np.testing.assert_array_equal(restored.drive_scores, arr)
    # list-of-list coerced to list-of-tuple by the existing from_dict path
    assert restored.gps_per_episode == [("gps_placeholder", 0.0), (None, 1.5)]


# ── resume helper: graceful on missing / GPS-less paths ────────────────


def test_rebuild_tolerates_missing_and_none_paths():
    """A GPS-less episode (None path) or a vanished .gpx yields ``(None, 0.0)``
    so index alignment with ``gpx_files`` is preserved."""
    from gomoveshift.pipeline.gps_enrichment import rebuild_gps_per_episode

    out = rebuild_gps_per_episode([None, "/does/not/exist.gpx"])
    assert out == [(None, 0.0), (None, 0.0)]
    assert rebuild_gps_per_episode(None) == []
