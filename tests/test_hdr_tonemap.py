"""HDR sources are tonemapped to BT.709 ahead of the grade."""

import json
import shutil
import subprocess
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline import single_pass as sp

HLG = {"color_trc": "arib-std-b67", "color_primaries": "bt2020", "color_range": "tv"}


_REAL_PROBE = getattr(sp, "_probe_converted_sample", None)
_BT709_TAGS = {"color_trc": "bt709", "color_primaries": "bt709", "color_range": "tv"}


def _real_zscale() -> bool:
    return bool(shutil.which("ffmpeg")) and {"zscale", "tonemap"} <= sp._ffmpeg_filter_names()


needs_zscale = pytest.mark.skipif(not _real_zscale(), reason="ffmpeg without zscale+tonemap")


@pytest.fixture(autouse=True)
def _no_nvenc_probe(monkeypatch):
    monkeypatch.setattr(sp, "detect_nvenc", lambda: True)
    # Unit tests with faked filter names cannot run ffmpeg; the real-sample
    # tests below call _REAL_PROBE / unpatched behaviour explicitly.
    if _REAL_PROBE is not None:
        monkeypatch.setattr(sp, "_probe_converted_sample", lambda *a, **k: dict(_BT709_TAGS))


def _proc():
    config = MagicMock()
    config.output.audio_bitrate = "192k"
    config.drive.blur_strength = 12
    return sp.SinglePassProcessor(config)


def test_hlg_source_tonemapped_and_tagged_bt709(monkeypatch):
    # Runs on every runner: filter inventory and the converted-sample probe are faked.
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    info, tonemap, operator = _proc()._resolve_hdr_source(dict(HLG))
    assert tonemap.startswith("zscale=")
    assert operator == "zscale:hable"
    assert info["color_trc"] == "bt709"
    assert info["color_primaries"] == "bt709"
    # the SDR gate re-checks the result
    _proc()._assert_sdr_source(info)
    vf = _proc()._build_video_filter(
        color_filter="null",
        location="",
        font_path="",
        duration=60.0,
        skip_ost=True,
        source_range="limited",
        tonemap_filter=tonemap,
    )
    assert vf.startswith(tonemap + ",setrange=limited")


@needs_zscale
def test_hlg_real_ffmpeg_sample_is_bt709(monkeypatch):
    # Real ffmpeg: the chosen chain runs on a sample and the OUTPUT is probed.
    monkeypatch.setattr(sp, "_probe_converted_sample", _REAL_PROBE)
    info, tonemap, operator = _proc()._resolve_hdr_source(dict(HLG))
    assert tonemap.startswith("zscale=")
    assert operator == "zscale:hable"
    assert info["color_trc"] == "bt709"
    assert info["color_primaries"] == "bt709"
    # the SDR gate re-checks the result
    _proc()._assert_sdr_source(info)
    assert sp._probe_converted_sample("arib-std-b67", "bt2020", tonemap)["color_trc"] == "bt709"
    vf = _proc()._build_video_filter(
        color_filter="null",
        location="",
        font_path="",
        duration=60.0,
        skip_ost=True,
        source_range="limited",
        tonemap_filter=tonemap,
    )
    assert vf.startswith(tonemap + ",setrange=limited")


def test_libplacebo_fallback_by_filter_name(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: frozenset({"libplacebo"}))
    monkeypatch.setattr(sp, "_libplacebo_usable", lambda: True)
    _info, tonemap, operator = _proc()._resolve_hdr_source(dict(HLG))
    assert tonemap.startswith("libplacebo=")
    assert operator == "libplacebo:bt.2390"


def test_hdr_refusal_names_conversion(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: frozenset())
    with pytest.raises(ValueError, match="ffmpeg -i"):
        _proc()._resolve_hdr_source(dict(HLG))


def test_sdr_source_untouched(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: frozenset({"zscale", "tonemap"}))
    info = {"color_trc": "bt709"}
    assert _proc()._resolve_hdr_source(info) == (info, "", "")


def test_result_records_operator():
    assert sp.SinglePassResult(success=True, tonemap="zscale:hable").to_dict()["tonemap"] == (
        "zscale:hable"
    )


# ── Review follow-ups ────────────────────────────────────────────────────

ZS = frozenset({"zscale", "tonemap"})
INFO = {"duration": 60.0, "height": 1080, "width": 1920, "fps": 30.0, **HLG}


def test_primaries_only_hdr_is_refused_early(monkeypatch):
    # bt2020 primaries with an unknown transfer: zscale has no path, so refuse
    # up front rather than failing inside the encode.
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    with pytest.raises(ValueError, match="HDR / non-BT.709"):
        _proc()._resolve_hdr_source({"color_primaries": "bt2020", "color_trc": ""})


def test_refusal_command_uses_filters_this_ffmpeg_lacks_only_as_advice(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: frozenset())
    with pytest.raises(ValueError) as exc:
        _proc()._resolve_hdr_source(dict(HLG))
    msg = str(exc.value)
    assert "ffmpeg -i" in msg
    assert "lacks" in msg and "zscale" in msg  # tells the operator to use another ffmpeg


def test_libplacebo_needs_a_working_device(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: frozenset({"libplacebo"}))
    monkeypatch.setattr(sp, "_libplacebo_usable", lambda: False)
    with pytest.raises(ValueError, match="ffmpeg -i") as exc:
        _proc()._resolve_hdr_source(dict(HLG))
    assert "Vulkan" in str(exc.value)


def test_filter_names_parsed_from_ffmpeg_listing(monkeypatch):
    listing = " ..C zscale V->V x\n .S tonemap V->V y\n libplacebo N->V z\n"
    monkeypatch.setattr(sp, "_filter_names_cache", None)
    monkeypatch.setattr(
        sp.subprocess, "run", lambda *a, **k: MagicMock(stdout=listing, returncode=0)
    )
    names = sp._ffmpeg_filter_names()
    assert {"zscale", "tonemap"} <= names
    monkeypatch.setattr(sp, "_filter_names_cache", None)

    def boom(*a, **k):
        raise OSError("no ffmpeg")

    monkeypatch.setattr(sp.subprocess, "run", boom)
    assert sp._ffmpeg_filter_names() == frozenset()
    monkeypatch.setattr(sp, "_filter_names_cache", None)


def test_libplacebo_probe_follows_ffmpeg_exit(monkeypatch):
    monkeypatch.setattr(sp, "_libplacebo_ok_cache", None)
    monkeypatch.setattr(sp.subprocess, "run", lambda *a, **k: MagicMock(returncode=0))
    assert sp._libplacebo_usable() is True
    monkeypatch.setattr(sp, "_libplacebo_ok_cache", None)
    monkeypatch.setattr(sp.subprocess, "run", lambda *a, **k: MagicMock(returncode=1))
    assert sp._libplacebo_usable() is False
    monkeypatch.setattr(sp, "_libplacebo_ok_cache", None)


def _wire_process(monkeypatch, tmp_path, *, blur=False):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    seen = {}
    proc = _proc()

    def fake_color(video_path, skip_color, lut_path, tonemap_filter=""):
        seen["color_tonemap"] = tonemap_filter
        return "null", "none", None

    def fake_encode(video_path, output_path, vf, *a, filter_complex=None, **k):
        seen["vf"] = vf
        seen["fc"] = filter_complex
        return sp.SinglePassResult(success=True, output_path=output_path)

    monkeypatch.setattr(proc, "_run_color_step", fake_color)
    monkeypatch.setattr(proc, "_run_audio_step", lambda *a, **k: {})
    monkeypatch.setattr(proc, "_run_encode_step", fake_encode)
    monkeypatch.setattr(proc, "_find_font", lambda: None)
    out = tmp_path / "day_master.tmp.mp4"
    result = proc.process(
        video_path=tmp_path / "in.mkv",
        output_path=out,
        location="",
        skip_ost=True,
        skip_audio=True,
        video_info=dict(INFO),
        blur_mask_path=(tmp_path / "mask.mkv") if blur else None,
    )
    return result, seen, tmp_path


def test_process_wires_tonemap_into_analysis_vf_and_state(monkeypatch, tmp_path):
    result, seen, _ = _wire_process(monkeypatch, tmp_path)
    assert result.tonemap == "zscale:hable"
    assert seen["vf"].startswith(sp._HDR_TONEMAP_CHAINS[0][3] + ",setrange=limited")
    assert seen["color_tonemap"] == sp._HDR_TONEMAP_CHAINS[0][3]  # measured post-tonemap
    sidecar = tmp_path / "day_master.tonemap.json"
    assert json.loads(sidecar.read_text())["tonemap"] == "zscale:hable"


def test_process_wires_tonemap_into_blur_filter_complex(monkeypatch, tmp_path):
    _result, seen, _ = _wire_process(monkeypatch, tmp_path, blur=True)
    assert sp._HDR_TONEMAP_CHAINS[0][3] in seen["fc"]


def test_sdr_process_writes_no_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    proc = _proc()
    monkeypatch.setattr(proc, "_run_color_step", lambda *a, **k: ("null", "none", None))
    monkeypatch.setattr(proc, "_run_audio_step", lambda *a, **k: {})
    monkeypatch.setattr(proc, "_find_font", lambda: None)
    monkeypatch.setattr(
        proc,
        "_run_encode_step",
        lambda v, o, *a, **k: sp.SinglePassResult(success=True, output_path=o),
    )
    proc.process(
        video_path=tmp_path / "in.mp4",
        output_path=tmp_path / "o.mp4",
        location="",
        skip_ost=True,
        skip_audio=True,
        video_info={**INFO, "color_trc": "bt709", "color_primaries": "bt709"},
    )
    assert not list(tmp_path.glob("*.tonemap.json"))


def test_analyze_color_samples_through_the_tonemap(monkeypatch, tmp_path):
    captured = {}

    def fake_sample(path, duration, **kw):
        captured.update(kw)
        return []

    monkeypatch.setattr(sp, "sample_frames", fake_sample)
    proc = _proc()
    monkeypatch.setattr(proc, "_get_video_info", lambda p: dict(INFO))
    proc._analyze_color(tmp_path / "v.mkv", tonemap_filter="TM")
    assert captured["tonemap_filter"] == "TM"
    assert captured["source_range"] == "limited"


def test_sample_frames_cmd_carries_tonemap(monkeypatch, tmp_path):
    from gomoveshift.pipeline import color_analysis as ca

    cmds = []

    class FakeProc:
        returncode = 1

        def communicate(self, timeout=None):
            return b"", b""

    def fake_popen(cmd, **k):
        cmds.append(cmd)
        return FakeProc()

    monkeypatch.setattr(ca.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ca, "_sample_frames_sequential", lambda *a, **k: [])
    ca.sample_frames(tmp_path / "v.mkv", 30.0, source_range="limited", tonemap_filter="TM")
    vf = cmds[0][cmds[0].index("-vf") + 1]
    assert vf.index("TM") < vf.index("setrange=limited")


# ── Review round 2 ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("trc", "primaries"),
    [
        ("bt2020-10", "bt2020"),
        ("bt2020-12", "bt2020"),
        ("bt709", "bt2020"),
        ("smpte170m", "smpte432"),
    ],
)
def test_sdr_transfer_wide_gamut_is_gamut_mapped_not_tonemapped(monkeypatch, trc, primaries):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    src = {"color_trc": trc, "color_primaries": primaries, "color_range": "tv"}
    info, chain, operator = _proc()._resolve_hdr_source(src)
    assert "tonemap=" not in chain
    assert "zscale=p=bt709" in chain
    assert operator == "zscale:gamut"
    assert info["color_primaries"] == "bt709"


def test_gamut_chain_needs_only_zscale(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: frozenset({"zscale"}))
    _info, _chain, operator = _proc()._resolve_hdr_source(
        {"color_trc": "bt2020-10", "color_primaries": "bt2020"}
    )
    assert operator == "zscale:gamut"


def test_gamut_conversion_without_zscale_is_refused(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: frozenset())
    with pytest.raises(ValueError, match="zscale"):
        _proc()._resolve_hdr_source({"color_trc": "bt709", "color_primaries": "bt2020"})


def test_resolve_refuses_when_converted_sample_is_not_bt709(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    monkeypatch.setattr(
        sp,
        "_probe_converted_sample",
        lambda *a, **k: {
            "color_trc": "arib-std-b67",
            "color_primaries": "bt2020",
            "color_range": "tv",
        },
    )
    with pytest.raises(ValueError, match="HDR / non-BT.709"):
        _proc()._resolve_hdr_source(dict(HLG))


def test_resolve_refuses_when_sample_probe_fails(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    monkeypatch.setattr(sp, "_probe_converted_sample", lambda *a, **k: {})
    with pytest.raises(ValueError, match="converted sample"):
        _proc()._resolve_hdr_source(dict(HLG))


def _mid_gray_luma(trc, primaries, chain):
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
        f"color=c=gray:s=64x64:d=0.1,format=yuv420p,setparams=color_trc={trc}:"
        f"color_primaries={primaries}:colorspace=bt2020nc:range=tv",
        "-vf", chain + ",format=gray", "-frames:v", "1", "-f", "rawvideo", "-",
    ]  # fmt: skip
    return subprocess.run(cmd, capture_output=True, timeout=60, check=True).stdout[0]


@needs_zscale
def test_bt2020_10_mid_gray_is_not_darkened(monkeypatch):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    _info, chain, _op = _proc()._resolve_hdr_source(
        {"color_trc": "bt2020-10", "color_primaries": "bt2020"}
    )
    assert abs(_mid_gray_luma("bt2020-10", "bt2020", chain) - 128) <= 3


@needs_zscale
def test_real_probe_reports_converted_tags():
    tags = _REAL_PROBE("bt2020-10", "bt2020", sp._GAMUT_CHAIN)
    assert tags["color_trc"] == "bt709"
    assert tags["color_primaries"] == "bt709"
    assert tags["color_range"] == "tv"


def test_color_cache_is_keyed_on_the_tonemap_chain(monkeypatch, tmp_path):
    proc = _proc()
    video = tmp_path / "v.mkv"
    video.write_bytes(b"x" * 64)
    calls = []

    def fake_sample(path, duration, **kw):
        calls.append(kw.get("tonemap_filter", ""))
        return []

    monkeypatch.setattr(sp, "sample_frames", fake_sample)
    monkeypatch.setattr(proc, "_get_video_info", lambda p: dict(INFO))
    cache = tmp_path / "v_color_analysis.json"
    cache.write_text(
        json.dumps(
            {
                "version": sp.TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
                "corrections": [],
                "tonemap": "OLD",
            }
        )
    )
    proc._analyze_color(video, tonemap_filter="NEW")
    assert calls == ["NEW"]  # stale chain: re-analyzed, not served from cache


def test_sdr_master_removes_stale_tonemap_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr(sp, "_ffmpeg_filter_names", lambda: ZS)
    (tmp_path / "o.tonemap.json").write_text("{}")
    proc = _proc()
    monkeypatch.setattr(proc, "_run_color_step", lambda *a, **k: ("null", "none", None))
    monkeypatch.setattr(proc, "_run_audio_step", lambda *a, **k: {})
    monkeypatch.setattr(proc, "_find_font", lambda: None)
    monkeypatch.setattr(
        proc,
        "_run_encode_step",
        lambda v, o, *a, **k: sp.SinglePassResult(success=True, output_path=o),
    )
    proc.process(
        video_path=tmp_path / "in.mp4",
        output_path=tmp_path / "o.tmp.mp4",
        location="",
        skip_ost=True,
        skip_audio=True,
        video_info={**INFO, "color_trc": "bt709", "color_primaries": "bt709"},
    )
    assert not (tmp_path / "o.tonemap.json").exists()


def test_analyze_color_hits_preseeded_hdr_cache(tmp_path, monkeypatch):
    """Day-master resume: a stable cache served through the preseed must hit
    _analyze_color for an HDR chain without re-running the analysis."""
    from gomoveshift.pipeline import day_master_encoder as dme

    chain = "zscale=t=linear,tonemap=hable,zscale=t=bt709:m=bt709"
    video = tmp_path / "raw.mp4"
    video.write_bytes(b"v" * 10)
    stable = {
        "corrections": [
            dict(
                shadow_lift=5.0,
                highlight_pull=5.0,
                saturation_mult=1.0,
                contrast_curve=0.0,
                white_balance_r=0.0,
                white_balance_b=0.0,
                gamma=1.0,
                denoise_strength=0.0,
                sharpen_amount=0.0,
            )
        ],
        "method": "auto_adaptive",
        "span_durations": [10.0],
        "tonemap": chain,
    }
    dme._preseed_transient_color_cache(video, stable)
    proc = _proc()
    proc.config.drive.no_sharpen = False
    proc.config.drive.no_denoise = False
    monkeypatch.setattr(sp, "is_analysis_cached", lambda *a, **k: True)
    monkeypatch.setattr(
        proc, "_corrections_to_filter", lambda c, s: ("null", "auto_adaptive", None)
    )
    result = proc._analyze_color(video, tonemap_filter=chain)
    assert result[1] == "auto_adaptive"
