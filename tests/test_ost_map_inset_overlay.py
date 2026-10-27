"""apply_ost(map_inset=...): composite an optional RGBA clip inside the OST encode.

The inset rides the SAME ffmpeg command as the drawtext burn (no second
full-length transcode) and must keep the master's 10-bit precision.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.ffmpeg_runner import VideoValidation

from ost_subprocess_harness import patch_ost_run


def _stub_probe():
    return {"width": 3840, "height": 2160, "fps": 29.97, "bitrate_kbps": 52000, "duration": 60.0}


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(
        "gomoveshift.pipeline.ost_applier.validate_video",
        lambda *_a, **_kw: VideoValidation(valid=True),
    )
    monkeypatch.setattr("gomoveshift.pipeline.color.detect_nvenc", lambda: False)


def _run(tmp_path, layout="regular_lf", logo=False, **kwargs):
    """Run apply_ost with ffmpeg mocked; return the burn argv."""
    in_mp4 = tmp_path / "in.mp4"
    in_mp4.write_bytes(b"\x00" * 1024)
    logo_path = None
    if logo:
        logo_path = tmp_path / "logo.png"
        logo_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    with (
        patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
        patch("gomoveshift.pipeline.ost_applier._has_audio_stream", return_value=False),
        patch_ost_run() as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr=b"")
        ost_applier.apply_ost(
            input_path=in_mp4,
            output_path=tmp_path / "out.mp4",
            location="Longwood to Fort Fremont",
            layout=layout,
            brand_logo_path=str(logo_path) if logo_path else None,
            **kwargs,
        )
    return mock_run.call_args[0][0]


def _inset(tmp_path, **kw):
    clip = tmp_path / "inset.mov"
    clip.write_bytes(b"\x00" * 64)
    return ost_applier.MapInsetOverlay(clip_path=clip, x=120, y=80, **kw)


def test_off_is_identical_for_every_layout_and_logo(mocked, tmp_path, monkeypatch):
    custom = dict(ost_applier.LAYOUTS["regular_lf"], show_logo=True)
    monkeypatch.setitem(ost_applier.LAYOUTS, "_logo_optin", custom)
    for layout, logo in [
        ("regular_lf", False),
        ("short", False),
        ("_logo_optin", True),
    ]:
        a = tmp_path / f"a_{layout}"
        b = tmp_path / f"b_{layout}"
        a.mkdir()
        b.mkdir()
        without = _run(a, layout=layout, logo=logo)
        explicit = _run(b, layout=layout, logo=logo, map_inset=None)
        norm = lambda cmd, d: [t.replace(str(d), "<T>") for t in cmd]  # noqa: E731
        assert norm(without, a) == norm(explicit, b)
        # And the legacy shape is intact: -vf for no logo, filter_complex for logo.
        assert ("-filter_complex" in explicit) is logo
        assert ("-vf" in explicit) is (not logo)
        assert explicit.count("-i") == (2 if logo else 1)
        if logo:
            fc = explicit[explicit.index("-filter_complex") + 1]
            assert fc.endswith("overlay=x=W-w-40:y=H-h-40[vout]")


def test_on_adds_second_input_overlay_and_explicit_formats(mocked, tmp_path):
    inset = _inset(tmp_path)
    cmd = _run(tmp_path, map_inset=inset)
    ins = [i for i, a in enumerate(cmd) if a == "-i"]
    assert len(ins) == 2
    assert cmd[ins[1] + 1] == str(inset.clip_path)
    assert "-vf" not in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "overlay=x=120:y=80" in fc
    assert "eof_action=pass" in fc
    assert "format=yuv420p10le" in fc  # base, before overlay
    assert "format=yuva420p10le" in fc  # inset, keeps alpha at 10 bits
    assert "overlay=" in fc and "format=yuv420p10" in fc.split("overlay=")[1]
    # Final stage pins the encoder's own pix_fmt, and it is a 10-bit one.
    enc_fmt = cmd[cmd.index("-pix_fmt") + 1]
    assert enc_fmt in ("yuv420p10le", "p010le")
    assert fc.rstrip().endswith(f"format={enc_fmt}[vout]")
    assert cmd[cmd.index("-map") + 1] == "[vout]"
    # Drawtext runs before the overlay.
    assert fc.index("drawtext") < fc.index("overlay=")


def test_on_with_logo_composites_inset_after_logo(mocked, tmp_path, monkeypatch):
    monkeypatch.setitem(
        ost_applier.LAYOUTS, "_logo_optin", dict(ost_applier.LAYOUTS["regular_lf"], show_logo=True)
    )
    inset = _inset(tmp_path)
    cmd = _run(tmp_path, layout="_logo_optin", logo=True, map_inset=inset)
    ins = [i for i, a in enumerate(cmd) if a == "-i"]
    assert [cmd[i + 1] for i in ins][-1] == str(inset.clip_path)
    assert len(ins) == 3
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert fc.index("W-w-40") < fc.index("overlay=x=120:y=80")
    assert "[2:v]" in fc


def test_missing_clip_returns_none_logs_and_writes_nothing(mocked, tmp_path, caplog):
    in_mp4 = tmp_path / "in.mp4"
    in_mp4.write_bytes(b"\x00" * 1024)
    out = tmp_path / "out.mp4"
    inset = ost_applier.MapInsetOverlay(clip_path=tmp_path / "nope.mov", x=1, y=2)
    with (
        patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
        patch_ost_run() as mock_run,
        caplog.at_level("ERROR", logger="gomoveshift.pipeline.ost_applier"),
    ):
        result = ost_applier.apply_ost(
            in_mp4, out, "Longwood to Fort Fremont", "regular_lf", map_inset=inset
        )
    assert result is None
    assert not mock_run.called
    assert not out.exists()
    assert any("nope.mov" in r.getMessage() for r in caplog.records)


def test_missing_font_does_not_silently_drop_the_inset(mocked, tmp_path):
    inset = _inset(tmp_path)
    in_mp4 = tmp_path / "in.mp4"
    in_mp4.write_bytes(b"\x00" * 1024)
    with (
        patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
        patch_ost_run() as mock_run,
    ):
        result = ost_applier.apply_ost(
            in_mp4,
            tmp_path / "out.mp4",
            "Longwood to Fort Fremont",
            "regular_lf",
            font_path=str(tmp_path / "no-such-font.ttf"),
            map_inset=inset,
        )
    assert result is None
    assert not mock_run.called


# --- real ffmpeg: the inset must not cost the master its 10 bits -------------


def _ffmpeg_can_encode_main10() -> bool:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=128x128:d=0.1", "-pix_fmt",
         "yuv420p10le", "-c:v", "libx265", "-f", "null", "-"],
        capture_output=True,
    )  # fmt: skip
    return r.returncode == 0


needs_main10 = pytest.mark.skipif(
    not _ffmpeg_can_encode_main10(), reason="ffmpeg/libx265 main10 not available"
)


def _gray10_frame(path: Path, w: int, h: int) -> object:
    import numpy as np

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-vf", "format=gray10le",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype="<u2").reshape(h, w)


@needs_main10
def test_inset_keeps_ten_bit_gradient_and_is_visible(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    monkeypatch.setattr("gomoveshift.pipeline.color.detect_nvenc", lambda: False)
    w, h = 1280, 360
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=30:d=2",
         "-vf", "format=yuv420p10le,geq=lum='64+(X/W)*876':cb=512:cr=512",
         "-c:v", "libx265", "-preset", "ultrafast", "-x265-params", "lossless=1", "-pix_fmt",
         "yuv420p10le", str(src)],
        check=True, capture_output=True,
    )  # fmt: skip
    clip = tmp_path / "inset.mov"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=white@0.6:s=160x160:r=10:d=1",
         "-vf", "format=rgba", "-c:v", "png", str(clip)],
        check=True, capture_output=True,
    )  # fmt: skip
    out = tmp_path / "out.mp4"
    inset = ost_applier.MapInsetOverlay(clip_path=clip, x=100, y=100)
    result = ost_applier.apply_ost(src, out, "Longwood", "regular_lf", prefer_nvenc=False,
                                   map_inset=inset)  # fmt: skip
    assert result == out and out.exists()

    pix = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert pix == "yuv420p10le"

    got, ref = _gray10_frame(out, w, h), _gray10_frame(src, w, h)
    # A row below the inset (rows 100-259) keeps its ramp. Distinct-level count
    # alone is not enough (encoder noise adds levels), so also pin the error
    # against the source: an 8-bit hop costs ~5 levels MAE, 10-bit stays under 1.
    row = got[300]
    assert len(np.unique(row)) > 256, "gradient collapsed to 8-bit"
    mae = float(np.abs(row.astype(int) - ref[300].astype(int)).mean())
    assert mae < 1.5, f"gradient drifted from the master (MAE {mae:.2f} 10-bit levels)"
    # Inset region is visibly different from the source there.
    region = (slice(100, 260), slice(100, 260))
    assert abs(float(got[region].mean()) - float(ref[region].mean())) > 40
