"""A sub-rendition source is resampled up with one scaler, and the fact is recorded."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import single_pass as sp
from gomoveshift.pipeline.day_master_encoder import _record_upscale, build_master_record
from gomoveshift.pipeline.ffmpeg_filters import build_crop_filter, crop_upscaled_from
from gomoveshift.pipeline.quality import (
    HONEST_RESOLUTION_RULE,
    UpscaleRefusedError,
    claim_native_resolution_allowed,
    picture_inputs_from_state,
    rendition_for,
    upscale_plan,
)
from gomoveshift.pipeline.validation.longform_validator import LongformValidator

LF4K = rendition_for("drive_longform_4k")
ZSCALE_4K = "zscale=w=3840:h=2160:filter=spline36"


def _process(tmp_path: Path, width: int, height: int, profile) -> tuple[str, object]:
    """Run ``process`` with every subprocess mocked; return (-vf chain, result)."""
    with patch.object(sp, "detect_nvenc", return_value=True):
        proc = sp.SinglePassProcessor(Config(), profile=profile)
    captured: dict = {}

    def fake_encode(video_path, output_path, vf, *args, **kwargs):
        captured["vf"] = vf
        return sp.SinglePassResult(success=True)

    info = {
        "duration": 60.0,
        "width": width,
        "height": height,
        "fps": 30.0,
        "color_range": "tv",
    }
    with (
        patch.object(proc, "_assert_sdr_source"),
        patch.object(proc, "_has_nvenc", return_value=True),
        patch.object(proc, "_run_color_step", return_value=("null", "none", None)),
        patch.object(proc, "_run_audio_step", return_value={}),
        patch.object(proc, "_find_font", return_value=None),
        patch.object(proc, "_run_encode_step", side_effect=fake_encode),
    ):
        result = proc.process(
            tmp_path / "in.mp4",
            tmp_path / "out.mp4",
            "",
            skip_ost=True,
            skip_audio=True,
            video_info=info,
        )
    return captured["vf"], result


def test_sub4k_source_upscaled_and_recorded(tmp_path):
    profile = SimpleNamespace(upscale="resample", claim_native_resolution_only=True)
    vf, result = _process(tmp_path, 1920, 1080, profile)
    assert vf.split(",")[0] == ZSCALE_4K
    assert result.upscale_plan.source == (1920, 1080)

    master = tmp_path / "day_2026-02-20_master.mp4"
    master.write_bytes(b"x")
    _record_upscale(master, result.upscale_plan, profile)
    batch = SimpleNamespace(date="2026-02-20", total_duration=60.0, episodes=[])
    record = build_master_record(batch, master, probe_fn=lambda p: 60.0)
    assert record["upscaled_from"] == [1920, 1080]
    assert record["native_resolution_claim"] is False

    art = SimpleNamespace(
        date="2026-02-20",
        master_path=None,
        master_record=record,
        regular_lf_indices=[0, 1],
    )
    _, changed = picture_inputs_from_state(
        SimpleNamespace(source_sharpness=[], day_artifacts=[art])
    )
    assert changed == {0: True, 1: True}


def test_native_source_chain_unchanged(tmp_path):
    baseline, native = _process(tmp_path, 3840, 2160, None)
    profile = SimpleNamespace(upscale="resample", claim_native_resolution_only=True)
    vf, result = _process(tmp_path, 3840, 2160, profile)
    assert vf == baseline
    assert result.upscale_plan is None
    assert "zscale" not in vf


def _gate(**kwargs):
    ffprobe = MagicMock()
    ffprobe.probe.return_value = {
        "format": {"bit_rate": "60000000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 3840,
                "height": 2160,
                "profile": "Main 10",
                "pix_fmt": "yuv420p10le",
                "color_range": "tv",
                "color_space": "bt709",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    return LongformValidator(ffprobe).check_video_quality(Path("/fake/lf.mp4"), **kwargs)


def test_4k_claim_blocked_when_policy_native_only():
    record = {"upscaled_from": [1920, 1080], "native_resolution_claim": True}
    blocked = _gate(day_record=record, claim_native_resolution_only=True)
    assert not blocked.passed
    assert HONEST_RESOLUTION_RULE in str(blocked.issues if hasattr(blocked, "issues") else blocked)
    assert _gate(day_record=record, claim_native_resolution_only=False).passed
    assert _gate(
        day_record={"upscaled_from": [1920, 1080], "native_resolution_claim": False},
        claim_native_resolution_only=True,
    ).passed


def test_upscale_refused_when_policy_off():
    with pytest.raises(UpscaleRefusedError) as exc:
        upscale_plan(1920, 1080, LF4K, "off")
    msg = str(exc.value)
    assert "1920x1080" in msg and "3840x2160" in msg and HONEST_RESOLUTION_RULE in msg
    assert upscale_plan(3840, 2160, LF4K, "off") is None


def test_plan_shapes_and_claim_policy():
    plan = upscale_plan(1920, 1080, LF4K, "resample")
    assert (plan.scaler, plan.filter, plan.source, plan.target) == (
        "zscale",
        ZSCALE_4K,
        (1920, 1080),
        (3840, 2160),
    )
    assert upscale_plan(1920, 1080, LF4K, "ml").filter == ZSCALE_4K
    wide = upscale_plan(2560, 1080, LF4K, "resample")
    assert wide.filter.endswith(",crop=3840:2160") and wide.crop == (3840, 2160)
    strict = SimpleNamespace(claim_native_resolution_only=True)
    loose = SimpleNamespace(claim_native_resolution_only=False)
    assert claim_native_resolution_allowed(strict, None)
    assert not claim_native_resolution_allowed(strict, plan)
    assert claim_native_resolution_allowed(loose, plan)


def test_short_from_narrow_crop_scaled_through_one_scaler():
    chain = build_crop_filter(1600, 900)  # 9:16 crop is 506 wide: an upscale
    assert chain == "crop=506:900,zscale=w=1080:h=1920:filter=spline36"
    assert "scale=" not in chain.replace("zscale=", "")
    assert crop_upscaled_from(1600, 900) == [506, 900]
    assert crop_upscaled_from(3840, 2160) is None


def _run_claim_gate(tmp_path, record, native_only):
    from gomoveshift.pipeline.drive_phases.phase_d_gates import _run_final_validation_gate

    big = tmp_path / "lf.mp4"
    big.write_bytes(b"x" * (1024 * 1024 + 1))

    def make_validator(**_kw):
        def validate(**kwargs):
            gate = _gate(
                day_record=kwargs.get("day_record"),
                claim_native_resolution_only=kwargs.get("claim_native_resolution_only", True),
            )
            return SimpleNamespace(passed=gate.passed, gates=[gate], summary=lambda: "")

        return SimpleNamespace(validate=validate)

    return _run_final_validation_gate(
        validator_cls=make_validator,
        config=SimpleNamespace(
            drive=SimpleNamespace(skip_audio=True),
            output=SimpleNamespace(audio_max_lufs=-18.0, audio_target_tp=-1.5),
        ),
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1, total_duration=1800.0)]),
        regular_lf_paths=[big],
        metadata_per_lf={1: {}},
        perceptual_check_fn=lambda p, d: None,
        picture_check_fn=lambda *a: None,
        day_records={1: record},
        claim_native_resolution_only=native_only,
    )


def test_phase_d_gate_aborts_upscaled_day_claiming_native(tmp_path):
    record = {"upscaled_from": [1920, 1080], "native_resolution_claim": True}
    with pytest.raises(RuntimeError, match="native resolution"):
        _run_claim_gate(tmp_path, record, True)


def test_phase_d_gate_passes_when_claim_policy_allows(tmp_path):
    record = {"upscaled_from": [1920, 1080], "native_resolution_claim": True}
    _run_claim_gate(tmp_path, record, False)


def test_pipeline_validator_forwards_claim_inputs(monkeypatch):
    from gomoveshift.pipeline.pipeline_validator import PipelineValidator

    pv = object.__new__(PipelineValidator)
    pv._longform = MagicMock()
    pv._check_file_integrity = lambda p: SimpleNamespace(passed=True, gate="file_integrity")
    pv._longform.check_video_quality.return_value = SimpleNamespace(
        passed=True, gate="video_quality"
    )
    record = {"upscaled_from": [1, 1]}
    pv.validate(
        Path("/f.mp4"), skip_audio=True, day_record=record, claim_native_resolution_only=False
    )
    kw = pv._longform.check_video_quality.call_args.kwargs
    assert kw["day_record"] is record and kw["claim_native_resolution_only"] is False


def test_short_upscaled_from_recorded_per_master_size():
    from gomoveshift.pipeline import day_pipeline as dp

    with patch.object(dp, "get_video_dimensions", return_value=(1600, 900)):
        assert dp._short_upscaled_from(Path("m.mp4")) == [506, 900]
    with patch.object(dp, "get_video_dimensions", return_value=(3840, 2160)):
        assert dp._short_upscaled_from(Path("m.mp4")) is None
    with patch.object(dp, "get_video_dimensions", side_effect=RuntimeError("no stream")):
        assert dp._short_upscaled_from(Path("m.mp4")) is None
