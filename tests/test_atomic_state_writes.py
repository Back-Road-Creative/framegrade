"""Tests for atomic state writes and .tmp corruption recovery.

Covers:
- state_manager.save_state uses mkstemp + os.replace (atomic pattern)
- state_manager.load_state recovers from .tmp when main file is corrupt
- _drive_state.save_drive_state uses mkstemp + os.replace
- _drive_state.load_drive_state recovers from .tmp when main file is corrupt
- Normal load still works for both modules
"""

import json
from unittest.mock import patch

import pytest
from _state_helpers import make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video_properties():
    """Minimal VideoProperties-like object for save_state."""
    from gomoveshift.pipeline.video_detect import VideoProperties

    return VideoProperties(
        width=1920,
        height=1080,
        fps=30.0,
        duration=120.0,
        bitrate_kbps=5000,
        codec="h264",
        audio_codec="aac",
        file_size_bytes=75_000_000,
    )


# ===================================================================
# state_manager tests
# ===================================================================


class TestStateManagerAtomicWrite:
    """save_state must use the atomic mkstemp → os.replace pattern."""

    def test_save_creates_file(self, tmp_path):
        from gomoveshift.pipeline.state_manager import save_state

        state_file = tmp_path / "state.json"
        src = tmp_path / "src.mp4"
        src.touch()

        save_state(
            state_file=state_file,
            project_name="test",
            source_video=src,
            source_hash="abc123",
            stages_completed=["stage_a"],
            current_video=src,
            video_properties=_make_video_properties(),
            timings={"stage_a": 1.0},
            params={"quality": "high"},
        )

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["project_name"] == "test"
        assert data["source_hash"] == "abc123"

    def test_save_uses_atomic_pattern(self, tmp_path):
        """Verify os.replace is called (atomic write, not bare open)."""
        import gomoveshift.pipeline.state_manager as _mod

        state_file = tmp_path / "state.json"
        src = tmp_path / "src.mp4"
        src.touch()
        original_replace = _mod.os.replace

        with patch.object(_mod.os, "replace", wraps=original_replace) as rep:
            _mod.save_state(
                state_file=state_file,
                project_name="test",
                source_video=src,
                source_hash="abc",
                stages_completed=[],
                current_video=src,
                video_properties=_make_video_properties(),
                timings={},
                params={},
            )

            rep.assert_called_once()

    def test_save_cleans_up_on_error(self, tmp_path):
        """On write failure, .tmp must be cleaned up (no leftover)."""
        import gomoveshift.pipeline.state_manager as _mod

        state_file = tmp_path / "state.json"
        src = tmp_path / "src.mp4"
        src.touch()

        with patch.object(_mod.json, "dump", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                _mod.save_state(
                    state_file=state_file,
                    project_name="test",
                    source_video=src,
                    source_hash="abc",
                    stages_completed=[],
                    current_video=src,
                    video_properties=_make_video_properties(),
                    timings={},
                    params={},
                )

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f".tmp not cleaned up: {tmp_files}"

    def test_no_leftover_tmp_on_success(self, tmp_path):
        """After a successful save, no .tmp files should remain."""
        from gomoveshift.pipeline.state_manager import save_state

        state_file = tmp_path / "state.json"
        src = tmp_path / "src.mp4"
        src.touch()

        save_state(
            state_file=state_file,
            project_name="test",
            source_video=src,
            source_hash="abc",
            stages_completed=[],
            current_video=src,
            video_properties=_make_video_properties(),
            timings={},
            params={},
        )

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"


class TestStateManagerLoad:
    """load_state: normal load, corrupt recovery, missing file."""

    def test_normal_load(self, tmp_path):
        from gomoveshift.pipeline.state_manager import load_state

        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"project_name": "test", "stages_completed": ["a"]}))

        result = load_state(state_file)
        assert result is not None
        assert result["project_name"] == "test"

    def test_missing_file_returns_none(self, tmp_path):
        from gomoveshift.pipeline.state_manager import load_state

        result = load_state(tmp_path / "nonexistent.json")
        assert result is None

    def test_corrupt_json_recovers_from_tmp(self, tmp_path):
        """Corrupt main file + valid .tmp sibling => recovery succeeds."""
        from gomoveshift.pipeline.state_manager import load_state

        state_file = tmp_path / "state.json"
        state_file.write_text("{corrupt data!!!}")

        # Create a valid .tmp sibling
        tmp_sibling = tmp_path / "state_recovery.tmp"
        tmp_sibling.write_text(json.dumps({"project_name": "recovered", "stages_completed": ["b"]}))

        result = load_state(state_file)
        assert result is not None
        assert result["project_name"] == "recovered"

    def test_corrupt_json_no_tmp_returns_none(self, tmp_path):
        """Corrupt main file + no .tmp sibling => returns None."""
        from gomoveshift.pipeline.state_manager import load_state

        state_file = tmp_path / "state.json"
        state_file.write_text("{corrupt data!!!}")

        result = load_state(state_file)
        assert result is None

    def test_corrupt_json_corrupt_tmp_returns_none(self, tmp_path):
        """Both main and .tmp are corrupt => returns None."""
        from gomoveshift.pipeline.state_manager import load_state

        state_file = tmp_path / "state.json"
        state_file.write_text("{corrupt}")

        tmp_sibling = tmp_path / "also_corrupt.tmp"
        tmp_sibling.write_text("{also corrupt}")

        result = load_state(state_file)
        assert result is None


# ===================================================================
# _drive_state tests
# ===================================================================


class TestDriveStateAtomicWrite:
    """save_drive_state must use the atomic mkstemp → os.replace pattern."""

    def test_save_creates_file(self, tmp_path):
        from gomoveshift.pipeline._drive_state import save_drive_state

        state_file = tmp_path / "drive_state.json"
        save_drive_state(state_file, {"phases_completed": ["phase_a"], "key": "value"})

        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert data["phases_completed"] == ["phase_a"]

    def test_save_uses_atomic_pattern(self, tmp_path):
        # The atomic write pattern now lives in _atomic_write; save_drive_state
        # delegates to it. Verify os.replace is invoked exactly once for the
        # state file.
        import gomoveshift.pipeline._atomic_write as _aw
        from gomoveshift.pipeline._drive_state import save_drive_state

        state_file = tmp_path / "drive_state.json"
        original_replace = _aw.os.replace

        with patch.object(_aw.os, "replace", wraps=original_replace) as rep:
            save_drive_state(state_file, {"key": "value"})
            rep.assert_called_once()

    def test_save_cleans_up_on_error(self, tmp_path):
        import gomoveshift.pipeline._atomic_write as _aw
        from gomoveshift.pipeline._drive_state import save_drive_state

        state_file = tmp_path / "drive_state.json"

        with patch.object(_aw.json, "dump", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                save_drive_state(state_file, {"key": "value"})

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f".tmp not cleaned up: {tmp_files}"

    def test_no_leftover_tmp_on_success(self, tmp_path):
        from gomoveshift.pipeline._drive_state import save_drive_state

        state_file = tmp_path / "drive_state.json"
        save_drive_state(state_file, {"phases_completed": ["phase_a"]})

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Leftover .tmp files: {tmp_files}"


class TestDriveStateLoad:
    """load_drive_state: normal load, corrupt recovery, missing file."""

    def test_normal_load(self, tmp_path):
        from gomoveshift.pipeline._drive_state import load_drive_state

        state_file = tmp_path / "drive_state.json"
        state_file.write_text(json.dumps({"phases_completed": ["phase_a"], "data": 42}))

        result = load_drive_state(state_file)
        assert result is not None
        assert result["phases_completed"] == ["phase_a"]

    def test_missing_file_returns_none(self, tmp_path):
        from gomoveshift.pipeline._drive_state import load_drive_state

        result = load_drive_state(tmp_path / "nonexistent.json")
        assert result is None

    def test_corrupt_json_recovers_from_tmp(self, tmp_path):
        from gomoveshift.pipeline._drive_state import load_drive_state

        state_file = tmp_path / "drive_state.json"
        state_file.write_text("{corrupt data!!!}")

        tmp_sibling = tmp_path / "recovery.tmp"
        tmp_sibling.write_text(
            json.dumps({"phases_completed": ["phase_bc_day_first"], "region": "usa"})
        )

        result = load_drive_state(state_file)
        assert result is not None
        assert result["phases_completed"] == ["phase_bc_day_first"]
        assert result["region"] == "usa"

    def test_corrupt_json_no_tmp_raises(self, tmp_path):
        """Corrupt state + no .tmp → raise loudly.

        Previously this returned None and the pipeline silently started
        fresh. Now load_drive_state raises CorruptStateError so the CLI
        can surface the failure instead of discarding prior work.
        """
        import pytest

        from gomoveshift.pipeline._drive_state import load_drive_state
        from gomoveshift.pipeline.exceptions import CorruptStateError

        state_file = tmp_path / "drive_state.json"
        state_file.write_text("{corrupt}")

        with pytest.raises(CorruptStateError):
            load_drive_state(state_file)

    def test_corrupt_json_corrupt_tmp_raises(self, tmp_path):
        """Corrupt main + corrupt .tmp → raise."""
        import pytest

        from gomoveshift.pipeline._drive_state import load_drive_state
        from gomoveshift.pipeline.exceptions import CorruptStateError

        state_file = tmp_path / "drive_state.json"
        state_file.write_text("{corrupt}")

        tmp_sibling = tmp_path / "bad.tmp"
        tmp_sibling.write_text("{also corrupt}")

        with pytest.raises(CorruptStateError):
            load_drive_state(state_file)

    def test_corrupt_json_orphan_sibling_tmp_raises(self, tmp_path):
        """A non-state sibling .tmp must NOT hydrate a near-default state.

        A crashed sibling ``atomic_write_json`` writer (e.g. drive_schedule.json)
        can leave an orphaned ``*.tmp`` in the stages dir. It parses cleanly as
        JSON but is not pipeline state — ``PipelineState.from_dict`` ignores its
        unknown keys and requires none, so recovery used to silently return a
        near-default state (empty phases_completed/youtube_uploads/OST prior
        hash) on a WARNING, turning a resume into a fresh-ish run. Recovery must
        skip the schema-mismatched orphan and raise ``CorruptStateError``.
        """
        import pytest

        from gomoveshift.pipeline._drive_state import load_drive_state
        from gomoveshift.pipeline.exceptions import CorruptStateError

        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text("{corrupt}")

        # Orphaned drive_schedule.json .tmp shape: carries ``region`` (which a
        # real state also has) but NO pipeline-state signature key.
        orphan = tmp_path / "tmpABC123.tmp"
        orphan.write_text(
            json.dumps({"region": "usa", "lf_slots": [], "shorts_slots": [], "computed_at": None})
        )

        with pytest.raises(CorruptStateError):
            load_drive_state(state_file)


class TestDriveStateExistingSerializationTests:
    """Verify existing roundtrip tests still pass with atomic writes."""

    def test_roundtrip_with_phases(self, tmp_path):
        from gomoveshift.pipeline._drive_state import load_drive_state, save_drive_state

        state_file = tmp_path / "state.json"
        state = make_state(
            phases_completed=["phase_a", "phase_bc_day_first"],
            regular_lf_paths=["/tmp/lf_01.mp4"],
            shorts_per_lf={1: []},
            route_profiles={"all": "/tmp/video.mp4"},
        )
        save_drive_state(state_file, state)
        loaded = load_drive_state(state_file)

        assert loaded is not None
        assert loaded["phases_completed"] == ["phase_a", "phase_bc_day_first"]
        assert loaded["route_profiles"]["all"] == "/tmp/video.mp4"
