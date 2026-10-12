"""Encoder ladder registry, probe and selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline import encoders
from gomoveshift.pipeline.encoders import (
    AMF,
    LADDER,
    NVENC,
    QSV,
    VAAPI,
    VIDEOTOOLBOX,
    X265,
    EncoderUnavailableError,
    probe_encoder,
    probe_ladder,
    select_encoder,
)


def _fake_run(failing=(), exc=()):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        name = cmd[cmd.index("-c:v") + 1]
        if name in exc:
            raise OSError(f"{name} boom")
        rc = 1 if name in failing else 0
        return SimpleNamespace(returncode=rc, stderr=f"{name} no device".encode())

    run.calls = calls
    return run


def test_ladder_order_x265_last():
    assert [b.name for b in LADDER] == [
        "hevc_nvenc",
        "hevc_videotoolbox",
        "hevc_qsv",
        "hevc_vaapi",
        "hevc_amf",
        "libx265",
    ]
    assert LADDER[-1] is X265 and not X265.hardware
    assert all(b.hardware for b in LADDER if b is not X265)


@pytest.mark.parametrize("backend", [NVENC, X265])
def test_args_carry_encoder_name(backend):
    args = backend.args("29.97", 60)
    assert args[args.index("-c:v") + 1] == backend.name


def test_describe():
    assert NVENC.describe() == "hevc_nvenc (NVIDIA NVENC, hardware)"
    assert "cpu" in X265.describe().lower()


def test_probe_passes_encoder_name():
    run = _fake_run()
    res = probe_encoder(X265, run=run)
    assert res.ok and res.backend is X265
    assert run.calls[0][run.calls[0].index("-c:v") + 1] == "libx265"
    assert "testsrc=duration=1:size=256x256:rate=1" in run.calls[0]


def test_probe_nonzero_rc_does_not_raise():
    res = probe_encoder(NVENC, run=_fake_run(failing=("hevc_nvenc",)))
    assert not res.ok and "no device" in res.reason


def test_probe_exception_does_not_raise():
    res = probe_encoder(NVENC, run=_fake_run(exc=("hevc_nvenc",)))
    assert not res.ok and "boom" in res.reason


def test_probe_ladder_in_order():
    res = probe_ladder(run=_fake_run(failing=("hevc_nvenc",)))
    assert [r.backend.name for r in res] == [b.name for b in LADDER]
    assert [r.ok for r in res] == [False] + [True] * (len(LADDER) - 1)


def test_required_fails_without_fallback():
    run = _fake_run(failing=("hevc_nvenc",))
    with pytest.raises(EncoderUnavailableError) as ei:
        select_encoder(required="hevc_nvenc", run=run)
    assert "no device" in str(ei.value)
    assert "CPU fallback is not permitted" in str(ei.value)
    assert len(run.calls) == 1


def test_required_ok_returns_backend():
    assert select_encoder(required="libx265", run=_fake_run()) is X265


def test_required_unknown_name():
    with pytest.raises(EncoderUnavailableError):
        select_encoder(required="hevc_nope", run=_fake_run())


def test_select_first_ok():
    assert select_encoder(run=_fake_run()) is NVENC
    hardware = tuple(b.name for b in LADDER if b is not X265)
    assert select_encoder(run=_fake_run(failing=hardware)) is X265


def test_select_none_lists_all_reasons():
    run = _fake_run(failing=tuple(b.name for b in LADDER))
    with pytest.raises(EncoderUnavailableError) as ei:
        select_encoder(run=run)
    msg = str(ei.value)
    assert "hevc_nvenc" in msg and "libx265" in msg


def test_calibration_encoder_follows_registry():
    from gomoveshift.pipeline import calibration

    assert calibration._ENCODER == encoders.NVENC.name


@pytest.mark.parametrize("backend", [VIDEOTOOLBOX, QSV, VAAPI, AMF])
def test_vendor_args_shape(backend):
    args = backend.args("29.97", 60)
    assert args[args.index("-c:v") + 1] == backend.name
    assert args[args.index("-r") + 1] == "29.97"
    assert args[args.index("-profile:v") + 1] == "main10"
    assert args[-2:] == ["-g", "60"]


def test_select_walk_honours_new_order():
    run = _fake_run(failing=("hevc_nvenc",))
    assert select_encoder(run=run) is VIDEOTOOLBOX


def test_required_amf_unavailable_raises():
    with pytest.raises(EncoderUnavailableError, match="hevc_amf"):
        select_encoder(required="hevc_amf", run=_fake_run(failing=("hevc_amf",)))
