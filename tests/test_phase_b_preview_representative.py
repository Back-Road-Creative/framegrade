"""Phase B preview clips must carry representative (near-silent) audio,
and the grade the drive will actually ship.

Previously previews were encoded straight from the RAW source with no audio
filter, so a reviewer heard LOUD source audio that never ships — every shipped
surface is attenuated to near-silence (the operator near-silent rule). The
preview encode now applies the SAME attenuate-only ``volume=`` filter, so the
audio a reviewer hears matches what ships.

The preview is the ONLY human visual check ahead of the multi-hour Phase C
master encode, and it used to render ``color_filter="null"`` — so a grading
defect was invisible to the one gate placed to catch it. The grade is now
analysed from the same 90s window the reviewer watches (a stream-copy cut, so
no extra encode) and rendered into the preview.

(Cutting from the planned LF/short windows is still infeasible in --preview
mode: that depends on the phase_bc day-master, which --preview intentionally
does not build.)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.single_pass import SinglePassProcessor

from _state_helpers import make_state


def _run_preview_capture(tmp_path, monkeypatch, *, make_outputs=False):
    ep_file = tmp_path / "ep_000.mp4"
    ep_file.touch()
    (tmp_path / "_stages").mkdir(exist_ok=True)
    state = make_state(episode_files=[ep_file], output_base=tmp_path)

    captured = []

    def fake_run(cmd, *args, **kwargs):
        cmd = [str(a) for a in cmd]
        captured.append(cmd)
        # ``monkeypatch.setattr(subprocess, "run", ...)`` is GLOBAL — every
        # module shares that one function object — so this can intercept a call
        # from anywhere in the window. Writing to ``cmd[-1]`` unguarded would
        # then clobber whatever an intercepted probe happened to name last,
        # including a source video. Write only inside this test's own tmp dir,
        # and only for an ffmpeg command whose last argument is an output path.
        target = Path(cmd[-1])
        writes_output = cmd[0] == "ffmpeg" and not cmd[-1].startswith("-")
        if make_outputs and writes_output and target.is_relative_to(tmp_path):
            target.write_bytes(b"\x00" * 32)
        result = MagicMock()
        result.returncode = 0
        result.stdout = b"3840\n300.0"
        result.stderr = b""
        return result

    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", fake_run)

    orch = DrivePipelineOrchestrator(config=Config())
    orch._phase_b_preview(state)
    return captured


def _encode_cmd(captured):
    """The ENCODE call, matched by its OUTPUT rather than a flag: the phase also
    takes a grade-window cut (carries "-t") and reaches the NVENC probe (carries
    "-c:v"), so only the destination file identifies it unambiguously."""
    for cmd in captured:
        if "ffmpeg" in cmd[0] and cmd[-1].endswith("_preview.mp4"):
            return cmd
    return None


def test_preview_encode_applies_attenuate_audio_filter(tmp_path, monkeypatch):
    captured = _run_preview_capture(tmp_path, monkeypatch)
    encode = _encode_cmd(captured)
    assert encode is not None, f"no 90s encode cmd captured: {captured}"
    assert "-af" in encode, "preview encode must apply an audio filter (near-silent)"
    af = encode[encode.index("-af") + 1]
    assert af.startswith("volume="), f"audio filter must be an attenuate-only volume gain: {af!r}"
    # Attenuate-only: the gain must never be positive (would amplify to audible).
    gain = float(af.removeprefix("volume=").removesuffix("dB"))
    assert gain <= 0.0, f"preview audio gain must not amplify, got {gain:+.2f} dB"


GRADE = "curves=all='0/0.02 0.5/0.55 1/1'"


def test_preview_renders_the_real_grade_for_the_sampled_window(tmp_path, monkeypatch):
    """The preview is the only human visual check ahead of the multi-hour master
    encode. Rendering it ungraded meant a grading defect could not be caught
    there — the one place it was cheap to catch."""
    seen = {}

    def fake_color_step(self, video_path, skip_color, lut_path):
        seen["analysed"] = Path(video_path)
        seen["skip_color"] = skip_color
        return (GRADE, "global", None)

    monkeypatch.setattr(SinglePassProcessor, "_run_color_step", fake_color_step)
    captured = _run_preview_capture(tmp_path, monkeypatch, make_outputs=True)

    encode = _encode_cmd(captured)
    assert encode is not None, f"no 90s encode cmd captured: {captured}"
    assert "-vf" in encode, "preview must render through a video filter chain"
    assert GRADE in encode[encode.index("-vf") + 1], "preview must carry the analysed grade"

    # The grade must come from the window the reviewer watches, not from the
    # whole episode and not from some other episode's lighting.
    assert seen.get("skip_color") is False
    cut = next((c for c in captured if "-c" in c and c[c.index("-c") + 1] == "copy"), None)
    assert cut is not None, f"grade window must be a stream copy, not a re-encode: {captured}"
    assert cut[cut.index("-t") + 1].startswith("90"), "grade window must be the previewed 90s"
    assert seen.get("analysed") == Path(cut[-1])


def test_preview_still_renders_when_the_grade_cannot_be_analysed(tmp_path, monkeypatch):
    """A preview is a review aid; a failed analysis must degrade it, never stop
    the drive. Nothing writes the cut here, so the window guard trips."""
    captured = _run_preview_capture(tmp_path, monkeypatch, make_outputs=False)
    encode = _encode_cmd(captured)
    assert encode is not None, f"no 90s encode cmd captured: {captured}"
    assert GRADE not in " ".join(encode)
