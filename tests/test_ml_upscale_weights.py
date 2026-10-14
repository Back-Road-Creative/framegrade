"""The ML-upscale weight pin is real and the export tool's arch matches the checkpoint layout."""

from __future__ import annotations

import re

import pytest

from gomoveshift.pipeline import quality


def test_weight_pin_is_real_sha256():
    sha = quality.ML_UPSCALE_WEIGHTS_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert sha != "0" * 64


def test_export_arch_loads_upstream_key_layout():
    torch = pytest.importorskip("torch")
    from tools import export_realesrgan_onnx as tool  # noqa: PLC0415

    # Tiny synthetic checkpoint built from the arch itself (1 block, no download).
    small = tool.build_model(num_block=1)
    ckpt = {tool.STATE_DICT_KEY: small.state_dict()}
    fresh = tool.build_model(num_block=1)
    fresh.load_state_dict(ckpt[tool.STATE_DICT_KEY], strict=True)
    keys = set(ckpt[tool.STATE_DICT_KEY])
    assert {"conv_first.weight", "conv_body.weight", "conv_last.bias"} <= keys
    assert "body.0.rdb3.conv5.weight" in keys
    assert tool.STATE_DICT_KEY == "params_ema"
    # Full-size arch has the 23 RRDB blocks upstream's x4plus checkpoint carries.
    assert len(tool.build_model().body) == 23
    out = fresh.eval()(torch.zeros(1, 3, 8, 8))
    assert tuple(out.shape) == (1, 3, 32, 32)
