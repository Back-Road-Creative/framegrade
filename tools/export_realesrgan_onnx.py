"""Export Real-ESRGAN x4plus (RRDBNet) to the ONNX file ``quality.ml_upscale_still`` loads.

Reproducible: the RRDBNet architecture is defined inline (no ``basicsr`` /
``realesrgan`` dependency) and the upstream ``RealESRGAN_x4plus.pth`` is loaded
with ``torch.load(weights_only=True)``. The service never downloads weights, so
fetch the .pth by hand (BSD-3-Clause, xinntao/Real-ESRGAN v0.1.0 release), then::

    .venv/bin/python tools/export_realesrgan_onnx.py RealESRGAN_x4plus.pth out.onnx

and pin the printed sha256 in ``quality.ML_UPSCALE_WEIGHTS_SHA256``.

I/O contract (matches ``quality._run_model``): one input ``input`` float32
NCHW, RGB, 0..1, dynamic N/H/W; one output ``output`` float32 NCHW, RGB, 0..1,
4x spatial size. No tiling is done by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "output"
STATE_DICT_KEY = "params_ema"  # upstream checkpoint key


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """x4 RRDBNet; scale is fixed at 4 (no pixel-unshuffle path is needed)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


def build_model(num_block: int = 23) -> RRDBNet:
    return RRDBNet(3, 3, 64, num_block, 32)


def load_model(pth: Path) -> RRDBNet:
    ckpt = torch.load(str(pth), map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(ckpt[STATE_DICT_KEY], strict=True)
    return model.eval()


def export(pth: Path, out: Path) -> str:
    model = load_model(pth)
    dummy = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(out),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "n", 2: "h", 3: "w"},
                OUTPUT_NAME: {0: "n", 2: "oh", 3: "ow"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pth", type=Path, help="upstream RealESRGAN_x4plus.pth")
    ap.add_argument("out", type=Path, help="ONNX path to write")
    args = ap.parse_args(argv)
    print(export(args.pth, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
