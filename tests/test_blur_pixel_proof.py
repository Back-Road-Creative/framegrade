"""Blur proof reads pixels, not just a marker file.

``check_applied`` accepted any FINAL sitting next to a ``.blurred`` marker. The
marker is an attestation: it says an encode once claimed to composite the mask,
and a stale, copied, or hand-created one says it just as convincingly. Nothing
ever looked at the artefact.

The spot-check closes that by measuring the FINAL against its own mask — the
masked region must carry markedly less high-frequency detail than the rest of
the same frame. It is deliberately one-sided: a clear negative fails, and an
unmeasurable frame (no mask coverage, an unreadable file) falls back to
marker-only with a warning. A privacy gate that hard-fails a six-hour encode on
a measurement it could not take would be worse than the hole it closes.
"""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline import blur_gate

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

BOX = "x=100:y=30:w=120:h=120"


def _render(path, filter_complex: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=1",
         "-filter_complex", filter_complex, "-c:v", "libx264", "-preset", "ultrafast",
         "-qp", "0", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip


@pytest.fixture
def mask(tmp_path):
    """White where the composite should have blurred, black everywhere else."""
    out = tmp_path / "mask.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c=black:s=320x180:r=10:d=1,drawbox={BOX}:color=white:t=fill",
         "-c:v", "ffv1", str(out)],
        check=True,
    )  # fmt: skip
    return out


@pytest.fixture
def blurred(tmp_path):
    _render(
        tmp_path / "blurred.mp4",
        "[0:v]split=2[base][src];[src]crop=120:120:100:30,boxblur=20[b];[base][b]overlay=100:30",
    )
    return tmp_path / "blurred.mp4"


@pytest.fixture
def unblurred(tmp_path):
    _render(tmp_path / "raw.mp4", "[0:v]null")
    return tmp_path / "raw.mp4"


def _ctx(tmp_path, mask_path, *, blur_enabled=True, skip_blur=False):
    return SimpleNamespace(
        config=SimpleNamespace(
            drive=SimpleNamespace(blur_enabled=blur_enabled, skip_blur=skip_blur)
        ),
        blur_mask_path=mask_path,
        paths=SimpleNamespace(stages_dir=tmp_path),
        project_name="proj",
    )


@needs_ffmpeg
def test_a_blurred_final_passes_the_pixel_check(blurred, mask):
    ok, why = blur_gate.masked_regions_look_blurred(blurred, mask)
    assert ok, why


@needs_ffmpeg
def test_an_unblurred_final_fails_the_pixel_check(unblurred, mask):
    """The failure this whole gate exists for: a FINAL whose masked region is
    every bit as sharp as the rest of the frame."""
    ok, why = blur_gate.masked_regions_look_blurred(unblurred, mask)
    assert not ok
    assert "sharp" in why.lower() or "detail" in why.lower()


@needs_ffmpeg
def test_a_marker_alone_no_longer_proves_an_unblurred_final(tmp_path, unblurred, mask):
    """A stale, copied or hand-written marker used to be the whole proof."""
    blur_gate.mark_applied(unblurred)
    err = blur_gate.check_applied(_ctx(tmp_path, mask), unblurred)
    assert err and "BLUR GATE" in err


@needs_ffmpeg
def test_marker_plus_corroborating_pixels_passes(tmp_path, blurred, mask):
    blur_gate.mark_applied(blurred)
    assert blur_gate.check_applied(_ctx(tmp_path, mask), blurred) is None


@needs_ffmpeg
def test_a_missing_marker_still_fails_before_any_decode(tmp_path, blurred, mask):
    err = blur_gate.check_applied(_ctx(tmp_path, mask), blurred)
    assert err and "no blur marker" in err


def test_an_unmeasurable_frame_falls_back_to_the_marker(tmp_path):
    """No mask, unreadable files, a mask that covers nothing — none of these are
    evidence of an unblurred ship, and none may fail a six-hour encode."""
    final = tmp_path / "FINAL.mp4"
    final.write_bytes(b"not a video")
    blur_gate.mark_applied(final)
    ctx = _ctx(tmp_path, None)
    assert blur_gate.check_applied(ctx, final) is None
    ok, _ = blur_gate.masked_regions_look_blurred(final, tmp_path / "gone.mkv")
    assert ok


def test_skip_blur_still_skips_everything(tmp_path):
    ctx = _ctx(tmp_path, None, skip_blur=True)
    assert blur_gate.check_applied(ctx, tmp_path / "FINAL.mp4") is None
