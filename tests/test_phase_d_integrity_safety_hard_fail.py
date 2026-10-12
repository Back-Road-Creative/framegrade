"""Phase D file-integrity + content-safety HARD FAIL escalation.

Before this gate, ``_run_final_validation_gate`` escalated only
``audio_loudness`` and ``location_invariant`` sub-gates to ``RuntimeError``;
every other failing gate (including ``file_integrity`` and ``content_safety``)
was demoted to ``logger.warning`` and the upload proceeded.

``file_integrity`` is deterministic (missing / <1 MiB / ffprobe-unreadable
FINAL.mp4 — genuinely corrupt) and ``content_safety`` blocks a curated
blocklist violation in title/description/captions; both have block-or-bust
intent, so they now HARD-fail. ``metadata_quality`` escalates too
(title/location mismatch, business-name-in-title, fragmented location tags —
the one place a title and its own description/location are compared).
The last WARN-only pair escalates too: ``video_quality`` and
``bitrate_preservation``: a regressed 8-bit, sub-4K, or under-floor-bitrate
long-form used to log a warning and ship.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    assert_content_safety_hard_fail,
    assert_file_integrity_hard_fail,
    assert_metadata_quality_hard_fail,
)
from gomoveshift.pipeline.drive_phases.phase_d_gates import (
    _run_final_validation_gate,
    assert_video_quality_hard_fail,
)


def _lf(idx, *, total_duration=1800.0):
    return SimpleNamespace(index=idx, total_duration=total_duration)


def _mk_config(*, skip_audio=True):
    return SimpleNamespace(
        drive=SimpleNamespace(skip_audio=skip_audio),
        output=SimpleNamespace(audio_max_lufs=-18.0, audio_target_tp=-1.5),
    )


# ── assert_file_integrity_hard_fail ──────────────────────────────────────


class TestAssertFileIntegrityHardFail:
    def test_no_failures_is_silent_noop(self):
        assert_file_integrity_hard_fail([])

    def test_single_failure_raises(self):
        with pytest.raises(RuntimeError) as exc:
            assert_file_integrity_hard_fail(
                ["  LF 2 (project_lf02_FINAL.mp4): File corrupt or unreadable: ffprobe failed"]
            )
        msg = str(exc.value)
        assert "HARD FAIL" in msg
        assert "LF 2" in msg


# ── assert_content_safety_hard_fail ──────────────────────────────────────


class TestAssertContentSafetyHardFail:
    def test_no_failures_is_silent_noop(self):
        assert_content_safety_hard_fail([])

    def test_single_failure_raises(self):
        with pytest.raises(RuntimeError) as exc:
            assert_content_safety_hard_fail(
                ["  LF 3 (project_lf03_FINAL.mp4): 1 violation(s): blocklisted term in title"]
            )
        msg = str(exc.value)
        assert "HARD FAIL" in msg
        assert "LF 3" in msg


# ── assert_metadata_quality_hard_fail ─────────────────────────


class TestAssertMetadataQualityHardFail:
    def test_no_failures_is_silent_noop(self):
        assert_metadata_quality_hard_fail([])

    def test_single_failure_raises(self):
        with pytest.raises(RuntimeError) as exc:
            assert_metadata_quality_hard_fail(
                ["  LF 4 (project_lf04_FINAL.mp4): title has no location keywords from 'X, SC'"]
            )
        msg = str(exc.value)
        assert "Metadata quality HARD FAIL" in msg
        assert "LF 4" in msg


# ── drive path RAISES on the new escalations ─────────────────────────────


class TestDrivePathEscalates:
    def test_failing_file_integrity_raises_on_drive_path(self, tmp_path):
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        gate = SimpleNamespace(
            gate="file_integrity",
            passed=False,
            message="File corrupt or unreadable: ffprobe failed",
        )
        result = SimpleNamespace(passed=False, gates=[gate], summary=lambda: "integrity fail")
        validator = SimpleNamespace(validate=lambda **kw: result)
        with pytest.raises(RuntimeError, match="File integrity HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: validator,
                config=_mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[big],
                metadata_per_lf={1: {}},
            )

    def test_failing_content_safety_raises_on_drive_path(self, tmp_path):
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        gate = SimpleNamespace(
            gate="content_safety",
            passed=False,
            message="1 violation(s): blocklisted term in title",
        )
        result = SimpleNamespace(passed=False, gates=[gate], summary=lambda: "content fail")
        validator = SimpleNamespace(validate=lambda **kw: result)
        with pytest.raises(RuntimeError, match="Content safety HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: validator,
                config=_mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[big],
                metadata_per_lf={1: {}},
            )

    def test_failing_metadata_quality_raises_on_drive_path(self, tmp_path):
        # metadata_quality escalates WARN → hard fail on the drive path.
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        gate = SimpleNamespace(
            gate="metadata_quality",
            passed=False,
            message="Metadata quality issues: title has no location keywords from 'X, SC'",
        )
        result = SimpleNamespace(passed=False, gates=[gate], summary=lambda: "meta fail")
        validator = SimpleNamespace(validate=lambda **kw: result)
        with pytest.raises(RuntimeError, match="Metadata quality HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: validator,
                config=_mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[big],
                metadata_per_lf={1: {}},
            )

    def test_failing_video_quality_raises_on_drive_path(self, tmp_path):
        # video_quality is deterministic (codec / resolution / bitrate /
        # profile / pix_fmt against frozen constants) — a failure means the
        # final is off-spec, so it must abort ahead of any upload.
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        gate = SimpleNamespace(
            gate="video_quality",
            passed=False,
            message="Quality issues: pix_fmt 'yuv420p' should be yuv420p10le or p010le",
        )
        result = SimpleNamespace(passed=False, gates=[gate], summary=lambda: "video fail")
        validator = SimpleNamespace(validate=lambda **kw: result)
        with pytest.raises(RuntimeError, match="Video quality HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: validator,
                config=_mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[big],
                metadata_per_lf={1: {}},
            )

    def test_failing_bitrate_preservation_raises_on_drive_path(self, tmp_path):
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        gate = SimpleNamespace(
            gate="bitrate_preservation",
            passed=False,
            message="Output 22000kbps below 40000kbps floor for 2160p",
        )
        result = SimpleNamespace(passed=False, gates=[gate], summary=lambda: "bitrate fail")
        validator = SimpleNamespace(validate=lambda **kw: result)
        with pytest.raises(RuntimeError, match="Video quality HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: validator,
                config=_mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[big],
                metadata_per_lf={1: {}},
            )

    def test_drive_path_needs_no_bitrate_sentinel(self, tmp_path):
        """``check_bitrate_preservation`` — the only ec-sourced bitrate floor —
        runs on every ``PipelineValidator.validate`` call because its floor is
        absolute. Pin that the drive path threads no stand-in value
        to switch it on: a flag posing as a measurement is the state this
        removes, and it can drift back to zero.
        """
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        seen: list[dict] = []
        result = SimpleNamespace(passed=True, gates=[], summary=lambda: "ok")

        def _validate(**kw):
            seen.append(kw)
            return result

        _run_final_validation_gate(
            validator_cls=lambda **kw: SimpleNamespace(validate=_validate),
            config=_mk_config(),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            regular_lf_paths=[big],
            metadata_per_lf={1: {}},
        )
        assert seen, "no validate() call"
        assert "source_bitrate_kbps" not in seen[0], (
            "drive path must not pass a source_bitrate_kbps sentinel — the "
            "bitrate_preservation gate is unconditional; got "
            f"{seen[0]['source_bitrate_kbps']!r}"
        )


class TestAssertVideoQualityHardFail:
    def test_no_failures_is_silent_noop(self):
        assert_video_quality_hard_fail([])

    def test_single_failure_raises(self):
        with pytest.raises(RuntimeError) as exc:
            assert_video_quality_hard_fail(
                ["  LF 2 (project_lf02_FINAL.mp4): profile 'main' should be main10"]
            )
        msg = str(exc.value)
        assert "Video quality HARD FAIL" in msg
        assert "LF 2" in msg


# ── missing / truncated FINAL.mp4 HARD-fails instead of skipping ──────


class TestMissingOrTruncatedFinalHardFails:
    """A missing or <1 MiB FINAL.mp4 is exactly the corrupt class
    the ``file_integrity`` gate exists to block, yet the pre-upload loop used
    to silently ``continue`` past it — so a truncated LF sailed through every
    pre-upload gate and Drive/YouTube uploads proceeded on the other LFs.
    These pin the flip from skip → HARD FAIL, upstream of any upload.

    The validator is stubbed to PASS so the raise can only come from the new
    missing/truncated detection, not from ``validate`` itself.
    """

    @staticmethod
    def _passing_validator():
        result = SimpleNamespace(passed=True, gates=[], summary=lambda: "ok")
        return SimpleNamespace(validate=lambda **kw: result)

    def test_missing_final_raises_before_validate(self, tmp_path):
        missing = tmp_path / "lf_missing.mp4"  # never created
        with pytest.raises(RuntimeError, match="File integrity HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: self._passing_validator(),
                config=_mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[missing],
                metadata_per_lf={1: {}},
            )

    def test_truncated_final_under_1mib_raises(self, tmp_path):
        small = tmp_path / "lf_small.mp4"
        small.write_bytes(b"x" * 1024)  # 1 KiB — well under the 1 MiB floor
        with pytest.raises(RuntimeError, match="File integrity HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: self._passing_validator(),
                config=_mk_config(),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[small],
                metadata_per_lf={1: {}},
            )


def test_missing_final_blocks_all_uploads_no_partial_ship(tmp_path, monkeypatch):
    """Regression: when one LF's FINAL.mp4 is missing, the pre-upload
    integrity gate must abort the whole run — NEITHER the valid sibling LF's
    Drive shorts NOR its YouTube LF may be uploaded (no partial irreversible
    ship). Driven through the real ``PhaseD.run`` upload wiring.
    """
    from gomoveshift.config import get_config
    from gomoveshift.pipeline.drive_phases.phase_d import PhaseD

    from test_phase_d_split_skip_upload import (
        _build_state,
        _install_fake_youtube_uploader,
        _phase_d_patches,
        _setup_mocks,
    )

    state = _build_state(tmp_path, num_lfs=2)
    # Corrupt LF 1 by deleting its FINAL.mp4; LF 2 stays valid (2 MiB).
    Path(state["regular_lf_paths"][0]).unlink()

    upload_tracker: dict = {}
    _install_fake_youtube_uploader(monkeypatch, upload_tracker)
    with (
        _phase_d_patches()[0] as m0,
        _phase_d_patches()[1] as m1,
        _phase_d_patches()[2] as m2,
        _phase_d_patches()[3] as m3,
        _phase_d_patches()[4] as m4,
        _phase_d_patches()[5] as m5,
        _phase_d_patches()[6] as m6,
        _phase_d_patches()[7] as m7,
    ):
        uploader = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7), num_lfs=2)
        with pytest.raises(RuntimeError, match="File integrity HARD FAIL"):
            PhaseD(get_config()).run(state)

    assert uploader.upload_shorts.call_count == 0, (
        "Drive shorts upload must NOT run when a sibling LF's FINAL is missing; "
        f"got {uploader.upload_shorts.call_count} call(s)"
    )
    assert upload_tracker.get("calls", []) == [], (
        "YouTube LF upload must NOT run when a sibling LF's FINAL is missing; "
        f"got calls for {upload_tracker.get('calls')}"
    )
