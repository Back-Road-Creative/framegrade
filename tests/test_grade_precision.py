"""HDR / gamut conversion keeps its precision into the 16-bit grade, and a
degraded grade is recorded (and refusable) instead of silent."""

import json
import shutil
import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoveshift.pipeline import single_pass as sp
from gomoveshift.pipeline.color_correction import CorrectionParams

_EIGHT_BIT = ("yuv420p,", "yuv420p\n", "format=rgb24", "format=nv12")


def _ends_8bit(chain: str) -> bool:
    last = chain.rsplit("format=", 1)[-1].split(":")[0].split(",")[0]
    return last in {"yuv420p", "yuv422p", "yuv444p", "nv12", "rgb24"}


def _have(*filters: str) -> bool:
    return bool(shutil.which("ffmpeg")) and set(filters) <= sp._ffmpeg_filter_names()


def _proc(**drive):
    config = MagicMock()
    config.output.audio_bitrate = "192k"
    config.drive.blur_strength = 12
    config.drive.no_sharpen = False
    config.drive.no_denoise = False
    for k, v in drive.items():
        setattr(config.drive, k, v)
    return sp.SinglePassProcessor(config)


@pytest.fixture(autouse=True)
def _no_nvenc_probe(monkeypatch):
    monkeypatch.setattr(sp, "detect_nvenc", lambda: True)


# ── chains ──────────────────────────────────────────────────────────────


def test_no_chain_ends_in_an_8bit_format():
    chains = [c[3] for c in sp._HDR_TONEMAP_CHAINS] + [sp._GAMUT_CHAIN]
    for chain in chains:
        assert not _ends_8bit(chain), chain
        assert chain.endswith(sp.TONEMAP_OUTPUT_FORMAT) or sp.TONEMAP_OUTPUT_FORMAT in chain


def test_terminator_keeps_at_least_ten_bits():
    assert sp.TONEMAP_OUTPUT_FORMAT.endswith(("10le", "12le", "16le"))


def test_sampler_and_encode_share_the_resolved_chain(monkeypatch):
    from gomoveshift.pipeline import color_analysis as ca

    chain = sp._GAMUT_CHAIN
    vf = _proc()._build_video_filter(
        color_filter="null",
        location="",
        font_path="",
        duration=60.0,
        skip_ost=True,
        source_range="limited",
        tonemap_filter=chain,
    )
    assert vf.startswith(chain + ",")
    cmds = []

    class FakeProc:
        def __init__(self, cmd, **kw):
            cmds.append(cmd)
            self.stdout = MagicMock(read=lambda n=-1: b"")
            self.stderr = MagicMock(read=lambda n=-1: b"")
            self.returncode = 0

        def communicate(self, *a, **k):
            return b"", b""

        def wait(self, *a, **k):
            return 0

    monkeypatch.setattr(ca.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(
        ca.subprocess,
        "run",
        lambda cmd, **kw: cmds.append(cmd) or MagicMock(stdout=b"", returncode=1),
    )
    try:
        ca.sample_frames("v.mkv", 10.0, tonemap_filter=chain, source_range="limited")
    except Exception:  # noqa: BLE001 — only the command line matters here
        pass
    assert cmds, "sampler built no ffmpeg command"
    vf_arg = cmds[0][cmds[0].index("-vf") + 1]
    assert chain in vf_arg
    # the sampler's own 8-bit RGB conversion is at its END, after the shared chain
    assert vf_arg.index(chain) < vf_arg.index("scale=")
    assert cmds[0][cmds[0].index("-pix_fmt") + 1] == "rgb24"


# ── real ffmpeg: the 8-bit hop is gone ──────────────────────────────────


def _distinct_levels(chain: str, trc: str, primaries: str, matrix: str) -> int:
    """Run a 10-bit luma ramp through ``chain`` into gbrp16le; count G levels."""
    src = (
        "color=c=gray:s=1024x16:d=0.1:r=1,format=yuv420p10le,"
        "geq=lum='X':cb=512:cr=512,"
        f"setparams=color_trc={trc}:color_primaries={primaries}:colorspace={matrix}:range=tv"
    )
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", src]
        + ["-vf", f"{chain},format={sp.GRADE_WORKING_FORMAT}", "-frames:v", "1"]
        + ["-f", "rawvideo", "-pix_fmt", "gbrp16le", "-"],
        capture_output=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode()[-400:]
    plane = np.frombuffer(proc.stdout, dtype="<u2")[: 1024 * 16]  # G plane first
    return len(np.unique(plane))


@pytest.mark.skipif(not _have("zscale"), reason="ffmpeg without zscale")
def test_gamut_chain_keeps_more_than_256_levels():
    n = _distinct_levels(sp._GAMUT_CHAIN, "bt709", "bt2020", "bt2020nc")
    assert n > 256, f"only {n} distinct levels: an 8-bit hop is still in the chain"


@pytest.mark.skipif(not _have("zscale", "tonemap"), reason="ffmpeg without zscale+tonemap")
def test_zscale_tonemap_chain_keeps_more_than_256_levels():
    chain = next(c[3] for c in sp._HDR_TONEMAP_CHAINS if c[0] == "zscale")
    n = _distinct_levels(chain, "arib-std-b67", "bt2020", "bt2020nc")
    assert n > 256, f"only {n} distinct levels: an 8-bit hop is still in the chain"


@pytest.mark.skipif(not _have("zscale"), reason="ffmpeg without zscale")
def test_gamut_chain_still_probes_bt709_limited():
    tags = sp._probe_converted_sample("bt709", "bt2020", sp._GAMUT_CHAIN)
    assert tags == {"color_trc": "bt709", "color_primaries": "bt709", "color_range": "tv"}


# ── grade fallback ──────────────────────────────────────────────────────

GRADE = [CorrectionParams(shadow_lift=8.0, gamma=1.1)]


def _break_bake(monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(sp, "params_to_lut3d_cube", boom)


def test_lut_grade_records_no_fallback():
    proc = _proc()
    filt, _method, lut_dir = proc._corrections_to_filter(GRADE, None)
    try:
        assert "lut3d=" in filt
        assert proc._grade_fallback is None
    finally:
        if lut_dir:
            shutil.rmtree(lut_dir, ignore_errors=True)


def test_failed_bake_records_the_fallback(monkeypatch):
    _break_bake(monkeypatch)
    proc = _proc()
    filt, _method, lut_dir = proc._corrections_to_filter(GRADE, None)
    assert lut_dir is None and "lut3d=" not in filt
    assert proc._grade_fallback == "eq_filter_chain"


def test_strict_grade_refuses_the_fallback(monkeypatch):
    _break_bake(monkeypatch)
    with pytest.raises(ValueError, match="GRADE-FALLBACK-REFUSED"):
        _proc(strict_grade=True)._corrections_to_filter(GRADE, None)


def test_strict_grade_defaults_off():
    from gomoveshift.config import DriveConfig

    assert DriveConfig().strict_grade is False


def _run_process(proc, monkeypatch, tmp_path, info):
    monkeypatch.setattr(proc, "_run_color_step", lambda *a, **k: ("null", "auto_global", None))
    monkeypatch.setattr(proc, "_run_audio_step", lambda *a, **k: {})
    monkeypatch.setattr(proc, "_find_font", lambda: None)
    monkeypatch.setattr(
        proc,
        "_run_encode_step",
        lambda v, o, *a, **k: sp.SinglePassResult(success=True, output_path=o),
    )
    return proc.process(
        video_path=tmp_path / "in.mp4",
        output_path=tmp_path / "day.tmp.mp4",
        location="",
        skip_ost=True,
        skip_audio=True,
        video_info=info,
    )


SDR = {
    "duration": 60.0,
    "height": 1080,
    "width": 1920,
    "fps": 30.0,
    "color_trc": "bt709",
    "color_primaries": "bt709",
}


def test_result_and_sidecar_carry_the_fallback(monkeypatch, tmp_path):
    proc = _proc()
    proc._grade_fallback = "eq_filter_chain"
    result = _run_process(proc, monkeypatch, tmp_path, dict(SDR))
    assert result.grade_fallback == "eq_filter_chain"
    assert result.to_dict()["grade_fallback"] == "eq_filter_chain"
    side = json.loads((tmp_path / "day.tonemap.json").read_text())
    assert side["grade_fallback"] == "eq_filter_chain"


def test_clean_sdr_run_has_null_fallback_and_no_sidecar(monkeypatch, tmp_path):
    result = _run_process(_proc(), monkeypatch, tmp_path, dict(SDR))
    assert result.grade_fallback is None
    assert result.to_dict()["grade_fallback"] is None
    assert not list(tmp_path.glob("*.tonemap.json"))
