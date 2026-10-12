"""Perceptual spot-check on the FINAL artifact.

Every pre-upload gate read container/stream METADATA only, so a long-form that
encoded to a black picture passed all of them and shipped. HARD FAIL only when
EVERY sampled window is essentially all black; partial black or freeze runs are
reported, not fatal — a tunnel or a stop is real footage and this runs
immediately before an irreversible upload.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.validation.longform_validator import (
    PERCEPTUAL_WINDOW_SEC,
    LongformValidator,
)


def _validator(stderr_per_window):
    """LongformValidator whose ffmpeg pass returns canned stderr per window."""
    calls: list[list[str]] = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        text = stderr_per_window[min(len(calls) - 1, len(stderr_per_window) - 1)]
        return SimpleNamespace(returncode=0, stdout="", stderr=text)

    ffprobe = MagicMock()
    ffprobe.probe_or_none.return_value = {"format": {"duration": "1800.0"}}
    ffprobe._subprocess = SimpleNamespace(run=_run)
    return LongformValidator(ffprobe), calls


def _black_stderr(duration: float) -> str:
    return f"[blackdetect @ 0x1] black_start:0 black_end:{duration} black_duration:{duration}\n"


def _freeze_stderr(duration: float) -> str:
    return f"[freezedetect @ 0x1] lavfi.freezedetect.freeze_duration: {duration}\n"


class TestCheckPerceptualSpot:
    def test_clean_final_passes(self):
        validator, calls = _validator(["frame= 300 fps=90 q=-0.0 Lsize=N/A\n"])
        # No explicit duration: it must fall back to the probed one and still sample.
        gate = validator.check_perceptual_spot(Path("/fake/lf.mp4"))
        assert gate.passed, gate.message
        assert len(calls) == 3, "three windows should be sampled across the final"
        assert all("blackdetect" in " ".join(c) and "freezedetect" in " ".join(c) for c in calls)

    def test_all_black_final_hard_fails(self):
        """A final with no picture anywhere — the unambiguous render fault."""
        validator, _ = _validator([_black_stderr(PERCEPTUAL_WINDOW_SEC)])
        gate = validator.check_perceptual_spot(Path("/fake/lf.mp4"), duration=1800.0)
        assert not gate.passed
        assert "black" in gate.message.lower()

    def test_single_black_window_is_reported_not_fatal(self):
        """One dark window (a tunnel) must not abort a run before upload."""
        validator, _ = _validator([_black_stderr(PERCEPTUAL_WINDOW_SEC), "clean\n", "clean\n"])
        gate = validator.check_perceptual_spot(Path("/fake/lf.mp4"), duration=1800.0)
        assert gate.passed, gate.message
        assert gate.measured_value and "black" in gate.measured_value

    def test_freeze_run_is_reported_not_fatal(self):
        validator, _ = _validator([_freeze_stderr(6.0), "clean\n", "clean\n"])
        gate = validator.check_perceptual_spot(Path("/fake/lf.mp4"), duration=1800.0)
        assert gate.passed, gate.message
        assert gate.measured_value and "freeze 6.0s" in gate.measured_value

    def test_unprobeable_file_degrades_to_a_pass(self):
        """file_integrity already hard-fails a corrupt final; this belt
        degrades rather than double-failing (momentary-loudness policy)."""
        validator, calls = _validator(["clean\n"])
        validator._ffprobe.probe_or_none.return_value = None
        gate = validator.check_perceptual_spot(Path("/fake/lf.mp4"), duration=1800.0)
        assert gate.passed
        assert "unavailable" in gate.message
        assert calls == [], "no ffmpeg pass on a file that will not probe"

    def test_ffmpeg_failure_degrades_to_a_pass(self):
        validator, _ = _validator(["clean\n"])
        validator._ffprobe._subprocess = SimpleNamespace(run=MagicMock(side_effect=OSError("gone")))
        gate = validator.check_perceptual_spot(Path("/fake/lf.mp4"), duration=1800.0)
        assert gate.passed
        assert "error" in gate.message.lower()


# ── Phase D wiring ────────────────────────────────────────────────────────


class TestPhaseDPerceptualWiring:
    def _run(self, tmp_path, perceptual_check_fn):
        from gomoveshift.pipeline.drive_phases.phase_d_gates import _run_final_validation_gate

        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        ok = SimpleNamespace(passed=True, gates=[], summary=lambda: "ok")
        return _run_final_validation_gate(
            validator_cls=lambda **kw: SimpleNamespace(validate=lambda **k: ok),
            config=SimpleNamespace(
                drive=SimpleNamespace(skip_audio=True),
                output=SimpleNamespace(audio_max_lufs=-18.0, audio_target_tp=-1.5),
            ),
            drive_plan=SimpleNamespace(
                regular_lfs=[SimpleNamespace(index=1, total_duration=1800.0)]
            ),
            regular_lf_paths=[big],
            metadata_per_lf={1: {}},
            perceptual_check_fn=perceptual_check_fn,
        )

    def test_all_black_final_aborts_before_upload(self, tmp_path):
        fail = SimpleNamespace(gate="perceptual_spot", passed=False, message="all black")
        with pytest.raises(RuntimeError, match="Perceptual spot-check HARD FAIL"):
            self._run(tmp_path, lambda path, duration: fail)

    def test_passing_check_lets_the_run_through_and_sees_each_lf(self, tmp_path):
        seen: list[tuple[str, float]] = []

        def _fn(path, duration):
            seen.append((Path(path).name, duration))
            return SimpleNamespace(gate="perceptual_spot", passed=True, message="clean")

        self._run(tmp_path, _fn)
        assert seen == [("lf.mp4", 1800.0)]

    def test_default_check_is_the_real_longform_validator(self):
        """No injection ⇒ production runs the real ffmpeg-backed check."""
        import inspect

        from gomoveshift.pipeline.drive_phases import phase_d_gates

        source = inspect.getsource(phase_d_gates._default_perceptual_check)
        assert "check_perceptual_spot" in source and "LongformValidator" in source
