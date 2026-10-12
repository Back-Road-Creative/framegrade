"""Resume-correctness regression tests for the OST burn stage.

The OST burn stage (``ost_applier.apply_ost``) produces the final
``lf_NN.mp4`` with on-screen titles, logo overlay, and website end-card.
It uses NVENC or libx265 and writes with ``-movflags +faststart`` which
rewrites the moov atom at the *end* of the encode. A mid-encode kill
(SIGTERM, power loss, container OOM) can leave one of several bad
states the resume logic must distinguish from a clean artifact:

1. **Clean** — ``lf_NN.mp4`` complete with moov atom at front.
2. **Half-written** — ``lf_NN.mp4`` has frames but the moov rewrite
   never finished. ffprobe may return partial duration or fail outright.
3. **State says done, file missing** — checkpoint claims the LF is
   done but the file was deleted (or never flushed to disk).
4. **Intermediate input missing** — the ``lf_NN_with_montage.mp4``
   that ``apply_ost`` takes as input is gone at resume time.

These tests are regressions against re-using a half-written MP4 on
resume. They use at least one REAL truncated-file fixture (raw bytes
on disk) so the test exercises the real filesystem path rather than
stubs only.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ost_subprocess_harness import patch_ost_run

# ── Helpers ────────────────────────────────────────────────────────────

# Minimal MP4 header (ftyp box) — 32 bytes of "looks like an MP4 but
# is truncated". Enough that a file-type sniff would say mp4; far too
# short to contain a moov atom.
_TRUNCATED_MP4_BYTES = (
    b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00"
    b"isomiso2avc1mp41"  # compatible brands
    b"\x00\x00\x00\x00"
)


def _mock_ffprobe_stdout(
    duration: float = 600.0,
    width: int = 3840,
    height: int = 2160,
    fps: float = 29.97,
    bitrate_kbps: int = 53000,
) -> bytes:
    data = {
        "streams": [
            {
                "width": width,
                "height": height,
                "r_frame_rate": f"{int(fps * 1000)}/1000",
            }
        ],
        "format": {"duration": str(duration), "bit_rate": str(bitrate_kbps * 1000)},
    }
    return json.dumps(data).encode("utf-8")


def _write_truncated_mp4(path: Path, extra_bytes: int = 0) -> Path:
    """Write a REAL truncated MP4 fixture (10-100 bytes, no moov).

    Simulates a mid-encode kill where the file has an ftyp header and
    some stray bytes but no trailing moov atom — i.e. technically not
    fully unplayable by every probe but structurally broken.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _TRUNCATED_MP4_BYTES + (b"\x00" * extra_bytes)
    assert 10 <= len(payload) <= 100, "Fixture must be 10-100 bytes per spec"
    path.write_bytes(payload)
    return path


# ── Tests ──────────────────────────────────────────────────────────────


class TestResumeCorrectnessOST:
    """Regression scenarios for resuming an OST burn after a crash."""

    # ── Scenario 1: Clean output ─────────────────────────────────────
    def test_clean_output_is_accepted_as_complete(self, tmp_path):
        """A well-formed lf_NN.mp4 with moov/faststart must be usable as-is.

        ffprobe returns a sane duration; no re-encode required on resume.
        """
        from gomoveshift.pipeline.ost_applier import _probe_video_info

        clean_lf = tmp_path / "lf_01.mp4"
        # Write a realistic-size placeholder — real encode would be MB.
        clean_lf.write_bytes(b"\x00" * 4096)

        def fake_run(cmd, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = _mock_ffprobe_stdout(duration=612.34)
            result.stderr = b""
            return result

        with patch_ost_run(side_effect=fake_run):
            info = _probe_video_info(clean_lf)

        # A clean artifact probes to a reasonable duration; resume logic
        # can trust this value and skip the OST re-encode.
        assert info["duration"] == pytest.approx(612.34)
        assert info["width"] == 3840
        assert info["height"] == 2160

    # ── Scenario 2: Half-written MP4 (REAL truncated fixture) ────────
    def test_half_written_output_is_not_silently_reused(self, tmp_path, monkeypatch):
        """A truncated mp4 (partial bytes, no moov) must not pass as complete.

        Uses a REAL 32-byte file written to disk — not just a stub. When
        apply_ost runs and its INPUT is a truncated file, ffprobe may
        return degenerate values or fail; the downstream behaviour
        should not treat the half-written artifact as valid.
        """
        from gomoveshift.pipeline.ost_applier import _probe_video_info, apply_ost

        # Create a real truncated fixture on disk (simulates a mid-encode
        # kill leaving a half-written lf_01.mp4).
        half_written = tmp_path / "lf_01_halfwritten.mp4"
        _write_truncated_mp4(half_written)
        assert half_written.exists()
        assert half_written.stat().st_size < 100  # real truncated file

        # 2a. ffprobe returns empty / partial output for a truncated file.
        # _probe_video_info raises ProbeFailedError
        # instead of returning hardcoded 4K@29.97/300s defaults — those
        # defaults silently passed truncated artifacts as "complete" on
        # resume. Resume callers must now decide
        # explicitly: apply_ost traps and returns None.
        from gomoveshift.pipeline.exceptions import ProbeFailedError

        def ffprobe_partial(cmd, *args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            # No streams, no format — simulates ffprobe giving up on a
            # file with no moov atom.
            result.stdout = b'{"streams": [], "format": {}}'
            result.stderr = b""
            return result

        with patch_ost_run(
            side_effect=ffprobe_partial,
        ):
            with pytest.raises(ProbeFailedError) as exc_info:
                _probe_video_info(half_written)

        # The exception carries the offending path so the operator can
        # locate it without grepping the log.
        assert exc_info.value.video_path == half_written

        # 2b. When a fresh apply_ost() runs with a truncated INPUT but
        # the output doesn't exist yet, ffmpeg will fail (rc != 0) and
        # apply_ost should return None — not silently produce a bad file.
        truncated_input = tmp_path / "lf_01_with_montage.mp4"
        _write_truncated_mp4(truncated_input)
        output_path = tmp_path / "lf_01.mp4"

        # Make the font exist so we hit the real encode path.
        fake_font = tmp_path / "font.ttf"
        fake_font.write_bytes(b"fake font")

        call_log = []

        def fake_ffmpeg_fail(cmd, *args, **kwargs):
            call_log.append(cmd[0])
            result = MagicMock()
            if "ffprobe" in str(cmd[0]):
                result.returncode = 0
                result.stdout = _mock_ffprobe_stdout(duration=600.0)
                result.stderr = b""
                return result
            # ffmpeg encode fails on a truncated input.
            result.returncode = 1
            result.stdout = b""
            result.stderr = b"[mov,mp4,m4a] moov atom not found\n"
            return result

        with patch_ost_run(side_effect=fake_ffmpeg_fail):
            # detect_nvenc is lazy-imported inside _resolve_encoding_args
            # — patch the source module so the import lands on the stub.
            # NVENC succeeds; the truncated-input failure mode
            # we're exercising here happens during the encode itself.
            with patch(
                "gomoveshift.pipeline.color.detect_nvenc",
                return_value=True,
            ):
                result = apply_ost(
                    input_path=truncated_input,
                    output_path=output_path,
                    location="Test Place via N51",
                    layout="regular_lf",
                    font_path=str(fake_font),
                    prefer_nvenc=False,
                )

        # Must return None on encode failure — never a partial path.
        assert result is None, (
            "apply_ost must return None when ffmpeg fails on a "
            "truncated input; otherwise resume logic would accept a "
            "half-written output as 'done'."
        )

    # ── Scenario 3: State says done, file missing ────────────────────
    def test_state_claims_done_but_lf_file_missing(self, tmp_path):
        """If checkpoint state records lf_NN.mp4 as done but the file is
        gone, a subsequent apply_ost() call must fail loud (return None)
        rather than succeed on a non-existent input.
        """
        from gomoveshift.pipeline.ost_applier import apply_ost

        # Simulate a state file claiming the upstream stage is done.
        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text(
            json.dumps(
                {
                    "day_2026_04_01": {
                        "regular_lfs_built": ["lf_01.mp4", "lf_02.mp4"],
                    }
                }
            )
        )

        # But the input the OST stage expects is missing.
        missing_input = tmp_path / "lf_01_with_montage.mp4"
        assert not missing_input.exists()

        output_path = tmp_path / "lf_01.mp4"

        # Should return None immediately — apply_ost checks input_path.exists()
        # at the top before spawning any subprocess.
        def fake_run_should_not_be_called(*args, **kwargs):  # pragma: no cover
            raise AssertionError("subprocess.run must not be invoked when input is missing")

        with patch_ost_run(
            side_effect=fake_run_should_not_be_called,
        ):
            result = apply_ost(
                input_path=missing_input,
                output_path=output_path,
                location="Somewhere via M7",
                layout="regular_lf",
            )

        assert result is None, (
            "apply_ost must return None when the input file is missing, "
            "even if an upstream state file claims the LF is done."
        )
        assert not output_path.exists(), "No output must be written when input is missing"

    # ── Scenario 4: Intermediate input missing at resume ─────────────
    def test_intermediate_with_montage_missing_on_resume(self, tmp_path):
        """On resume, lf_NN_with_montage.mp4 may have been garbage-
        collected from staging. apply_ost must refuse to run rather
        than creating a zero-byte output.
        """
        from gomoveshift.pipeline.ost_applier import apply_ost

        # Put a previous (possibly stale) final output in place — the
        # resume path must not rely on it when the *input* is gone.
        stale_final = tmp_path / "lf_02.mp4"
        _write_truncated_mp4(stale_final)  # real 32-byte truncated file

        missing_montage = tmp_path / "lf_02_with_montage.mp4"
        # Deliberately not created.
        assert not missing_montage.exists()

        result = apply_ost(
            input_path=missing_montage,
            output_path=stale_final,
            location="Glencolumbkille via R263",
            layout="regular_lf",
        )

        assert result is None, (
            "apply_ost must return None when its intermediate input "
            "(lf_NN_with_montage.mp4) is missing at resume time"
        )
        # Stale final remains untouched (we didn't accidentally truncate it).
        assert stale_final.exists()
        assert stale_final.read_bytes().startswith(b"\x00\x00\x00\x20ftyp")

    # ── Scenario 5: ffprobe fails on input → apply_ost returns None ──
    def test_apply_ost_returns_none_when_probe_fails(self, tmp_path, caplog):
        """_probe_video_info must not fall back to silent defaults.

        When the input exists but ffprobe cannot read it (truncated /
        half-written / corrupt), apply_ost must trap ProbeFailedError
        and return None — NOT continue with the legacy hardcoded
        3840x2160@29.97/300s defaults that silently re-encoded
        truncated artifacts on resume.
        """
        from gomoveshift.pipeline.ost_applier import apply_ost

        # Real truncated input on disk; the file exists so the
        # input_path.exists() guard passes, but ffprobe will reject it.
        truncated_input = tmp_path / "lf_03_with_montage.mp4"
        _write_truncated_mp4(truncated_input)
        output_path = tmp_path / "lf_03.mp4"

        fake_font = tmp_path / "font.ttf"
        fake_font.write_bytes(b"fake font")

        ffmpeg_calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):
            cmd_str = " ".join(str(c) for c in cmd)
            result = MagicMock()
            if cmd[0] == "ffprobe":
                # Empty payload — _probe_video_info will raise.
                result.returncode = 0
                result.stdout = b'{"streams": [], "format": {}}'
                result.stderr = b""
                return result
            # ffmpeg must NOT be invoked when probe fails.
            ffmpeg_calls.append(list(cmd))
            raise AssertionError(f"ffmpeg must not run when ffprobe failed; cmd={cmd_str}")

        with patch_ost_run(side_effect=fake_run):
            with caplog.at_level("ERROR", logger="gomoveshift.pipeline.ost_applier"):
                result = apply_ost(
                    input_path=truncated_input,
                    output_path=output_path,
                    location="Frogmore via N51",
                    layout="regular_lf",
                    font_path=str(fake_font),
                    prefer_nvenc=False,
                )

        assert result is None, (
            "apply_ost must return None when ffprobe rejects the input — "
            "the legacy hardcoded-defaults fallback is removed"
        )
        assert ffmpeg_calls == [], "ffmpeg must not run after probe failure"
        assert not output_path.exists(), (
            "No output must be written when ffprobe could not read the input"
        )
        # Operator-visible ERROR with actionable hint.
        messages = [r.getMessage() for r in caplog.records]
        assert any(
            "OST apply: aborting encode" in m and "ffprobe could not read" in m for m in messages
        ), f"expected aborting-encode error message, got: {messages!r}"
