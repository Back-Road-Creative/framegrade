"""Phase D audio-loudness HARD FAIL gate.

Under the near-silent policy every published
LF is attenuate-only, so the only audio defect is being too LOUD. The gate now
raises ``RuntimeError`` (aborting before any Drive/YouTube upload) when at least
one LF is LOUDER than the near-silent ceiling after attenuation; a too-QUIET LF
is the goal and never produces a failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import assert_audio_loudness_hard_fail
from gomoveshift.pipeline.pipeline_validator import GateResult, PipelineValidationResult


class TestAssertAudioLoudnessHardFail:
    def test_no_failures_is_a_silent_noop(self):
        # Empty list → no raise.
        assert_audio_loudness_hard_fail([])

    def test_single_lf_too_loud_raises(self):
        with pytest.raises(RuntimeError) as exc:
            assert_audio_loudness_hard_fail(
                [
                    "  LF 3 (project_lf03_FINAL.mp4): loudness -18.0 is LOUDER than "
                    "the near-silent ceiling -43 LUFS"
                ]
            )
        msg = str(exc.value)
        assert "HARD FAIL" in msg
        assert "LF 3" in msg
        assert "-18.0" in msg
        # New wording: the gate fires on too-LOUD LFs, never too-quiet.
        assert "LOUDER" in msg or "near-silent" in msg

    def test_multiple_lfs_all_listed(self):
        failures = [
            "  LF 1 (a.mp4): loudness -18.0 is LOUDER than the near-silent ceiling -43 LUFS",
            "  LF 4 (d.mp4): loudness -12.0 is LOUDER than the near-silent ceiling -43 LUFS",
        ]
        with pytest.raises(RuntimeError) as exc:
            assert_audio_loudness_hard_fail(failures)
        msg = str(exc.value)
        assert "LF 1" in msg
        assert "LF 4" in msg


class TestAudioFailureDetectionFromGateResult:
    """Ensures the failure-extraction shape Phase D builds matches the
    GateResult contract (regression guard against future refactors that
    rename the gate or drop it from ``gates``)."""

    def test_gate_result_contains_audio_loudness_when_failing(self):
        # Mirrors what PhaseD constructs from gate_validator.validate(...).
        result = PipelineValidationResult(
            passed=False,
            gates=[
                GateResult(
                    gate="audio_loudness",
                    passed=False,
                    message="Audio too loud: loudness -18.0 is LOUDER than the "
                    "near-silent ceiling -43 LUFS",
                ),
                GateResult(gate="video_quality", passed=True, message="ok"),
            ],
        )
        audio_gate = next((g for g in result.gates if g.gate == "audio_loudness"), None)
        assert audio_gate is not None
        assert not audio_gate.passed
        assert "-18.0" in audio_gate.message

    def test_gate_result_passing_audio_skips_failure_collection(self):
        result = PipelineValidationResult(
            passed=False,
            gates=[
                GateResult(gate="audio_loudness", passed=True, message="ok"),
                GateResult(gate="duration", passed=False, message="duration mismatch"),
            ],
        )
        audio_gate = next((g for g in result.gates if g.gate == "audio_loudness"), None)
        assert audio_gate is not None
        assert audio_gate.passed
        # PhaseD's branch must NOT collect this as an audio_failure even
        # though the overall result.passed=False.
