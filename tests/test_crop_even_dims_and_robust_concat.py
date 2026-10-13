"""Even crop dimensions + partial-output cleanup on the montage concat.

1. ``build_crop_filter`` computed an ODD crop width for every common source
   resolution (3840x2160 -> 1215). 4:2:0 chroma is subsampled 2x horizontally,
   so ffmpeg cannot honour an odd width and silently rounds it down. Relying on
   that round-down means the real crop window is not the one the code computed.

2. ``prepend_montage``'s final concat used a bare ``subprocess.run``. On
   timeout or a crash it left a truncated ``output_path`` on disk. Every other
   long ffmpeg call in the pipeline goes through the shared runner, which
   deletes partial output before surfacing the failure.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import montage_builder
from gomoveshift.pipeline.ffmpeg_filters import build_crop_filter


def _crop_dims(filter_str: str) -> tuple[int, int]:
    m = re.match(r"crop=(\d+):(\d+)", filter_str)
    assert m, f"unexpected crop filter shape: {filter_str!r}"
    return int(m.group(1)), int(m.group(2))


class TestCropDimensionsAreEven:
    """4:2:0 needs even crop dimensions on both axes."""

    @pytest.mark.parametrize(
        "src_w,src_h",
        [
            (3840, 2160),  # GoPro 4K — the common case, crop_w was 1215
            (2704, 1520),  # GoPro 2.7K
            (1920, 1080),  # 1080p
            (5312, 2988),  # Insta360 stitched
            (4000, 3000),  # 4:3 source — exercises the crop-height branch
            (1080, 1920),  # already vertical — no crop needed
        ],
    )
    def test_crop_dimensions_are_even(self, src_w, src_h):
        crop_w, crop_h = _crop_dims(build_crop_filter(src_w, src_h))
        assert crop_w % 2 == 0, f"{src_w}x{src_h} -> odd crop width {crop_w}"
        assert crop_h % 2 == 0, f"{src_w}x{src_h} -> odd crop height {crop_h}"

    def test_crop_never_exceeds_the_source(self):
        """Rounding to even must round DOWN — a crop wider than the source
        makes ffmpeg fail outright."""
        for src_w, src_h in [(3840, 2160), (2704, 1520), (1921, 1081), (4000, 3000)]:
            crop_w, crop_h = _crop_dims(build_crop_filter(src_w, src_h))
            assert crop_w <= src_w, f"crop width {crop_w} exceeds source {src_w}"
            assert crop_h <= src_h, f"crop height {crop_h} exceeds source {src_h}"

    def test_aspect_ratio_still_lands_on_9_16(self):
        """Evening the dimensions must not move the crop off the shorts aspect.

        Shorts are rejected by vertical platforms if they are not 9:16, so the
        crop window must stay within a pixel of the target ratio.
        """
        crop_w, crop_h = _crop_dims(build_crop_filter(3840, 2160))
        assert crop_w / crop_h == pytest.approx(1080 / 1920, abs=0.002)

    def test_scaler_survives(self):
        """Regression guard: the explicit scaler on the shorts crop must not be
        lost. The scaler is the one declared upscaler, ``zscale`` spline36."""
        chain = build_crop_filter(3840, 2160)
        assert "zscale=" in chain and "filter=spline36" in chain


class TestPrependMontageCleansPartialOutput:
    """The final concat must not leave a truncated file behind."""

    @pytest.fixture
    def scratch(self, tmp_path):
        raw_lf = tmp_path / "raw_lf.mp4"
        raw_lf.write_bytes(b"RAW_LF")
        work = tmp_path / "work"
        work.mkdir()
        montage = work / "out_montage.mp4"
        montage.write_bytes(b"MONTAGE")
        return SimpleNamespace(
            raw_lf=raw_lf,
            work=work,
            out=tmp_path / "out.mp4",
            master=tmp_path / "master.mp4",
            peaks=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=40.0)],
            boundaries=[(0.0, 600.0)],
        )

    def _run(self, scratch, fake_run):
        """Drive prepend_montage with build_montage stubbed to succeed."""

        def _fake_build(master_path, peaks, file_boundaries, output_path, work_dir, **kw):
            Path(output_path).write_bytes(b"MONTAGE")
            return Path(output_path)

        # prepend_montage's concat now runs through ffmpeg_runner, so that is
        # the seam — montage_builder no longer calls subprocess itself.
        with (
            patch.object(montage_builder, "build_montage", side_effect=_fake_build),
            patch("gomoveshift.pipeline.ffmpeg_runner.subprocess.run", side_effect=fake_run),
        ):
            return montage_builder.prepend_montage(
                raw_lf_path=scratch.raw_lf,
                peaks=scratch.peaks,
                output_path=scratch.out,
                work_dir=scratch.work,
                master_path=scratch.master,
                file_boundaries=scratch.boundaries,
            )

    def test_timeout_removes_the_partial_output(self, scratch):
        """A concat killed by the timeout leaves a truncated mp4 on disk.

        Left behind, a resume finds a plausible-looking output and ships a
        half-length long-form.
        """

        def fake_run(cmd, *a, **kw):
            Path(cmd[-1]).write_bytes(b"PARTIAL")  # ffmpeg wrote some output
            raise subprocess.TimeoutExpired(cmd, 1800)

        assert self._run(scratch, fake_run) is None
        assert not scratch.out.exists(), "truncated concat output was left on disk"

    def test_nonzero_exit_removes_the_partial_output(self, scratch):
        def fake_run(cmd, *a, **kw):
            Path(cmd[-1]).write_bytes(b"PARTIAL")
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="boom")

        assert self._run(scratch, fake_run) is None
        assert not scratch.out.exists(), "failed concat output was left on disk"

    def test_success_keeps_the_output(self, scratch):
        def fake_run(cmd, *a, **kw):
            Path(cmd[-1]).write_bytes(b"JOINED")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        assert self._run(scratch, fake_run) == scratch.out
        assert scratch.out.read_bytes() == b"JOINED"
