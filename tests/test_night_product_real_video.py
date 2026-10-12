"""The typed timelapse contract accepts a real production-encoded MP4."""

import datetime as dt
from types import SimpleNamespace

import numpy as np

from gomoveshift.pipeline.photos import astro_timelapse, night_products
from gomoveshift.pipeline.photos.night_collection import (
    NightCollection,
    NightFrame,
    NightSequence,
    ProductRequest,
)
from gomoveshift.pipeline.photos.night_project import NightStore


def test_typed_timelapse_encodes_and_receipts_real_video(tmp_path, monkeypatch):
    card = tmp_path / "card"
    card.mkdir()
    frames = []
    for index in range(2):
        path = card / f"light-{index}.NEF"
        path.write_bytes(f"captured-{index}".encode())
        frames.append(
            NightFrame(
                path,
                dt.datetime(2026, 9, 14, 23, 59, index),
                3200,
                1,
            )
        )
    collection = NightCollection(
        root=card,
        lights=tuple(frames),
        likely_darks=(),
        unreadable=(),
        sequences=(NightSequence(tuple(frames)),),
        existing_linear=(),
        invalid_linear=(),
        refused=(),
    )
    store = NightStore(tmp_path / "projects")
    identifier = store.create_product_request(
        "brand",
        collection,
        [tmp_path],
        ProductRequest.create(["timelapse"]),
    )
    row = store.get("brand", identifier)
    task = row["tasks"][0]

    decoded = {
        frames[0].path: np.full((4, 6, 3), 0.20, dtype=np.float32),
        frames[1].path: np.full((4, 6, 3), 0.75, dtype=np.float32),
    }

    def decode(path, *_args):
        return SimpleNamespace(rgb=decoded[path].copy()), None

    def develop(rgb, sky_mask=None):
        return SimpleNamespace(rgb_uint16=np.rint(rgb * 65535).astype(np.uint16))

    def redevelop(rgb, _base):
        return np.rint(rgb * 65535).astype(np.uint16)

    monkeypatch.setattr(astro_timelapse, "decode_with_he_fallback", decode)
    monkeypatch.setattr(astro_timelapse, "develop_stack", develop)
    monkeypatch.setattr(astro_timelapse, "develop_stack_with_params", redevelop)
    engines = night_products.default_engines()
    engines.update(
        raw_megapixels=lambda _path: 0.000024,
        preflight_timelapse=lambda *_args, **_kwargs: None,
        timelapse_fps=2,
        timelapse_height=4,
    )

    receipt = night_products.render_product(
        row,
        task,
        tmp_path / "attempt",
        engines=engines,
    )

    video = next(output for output in receipt["outputs"] if output["role"] == "video")
    assert receipt["state"] == "complete"
    assert receipt["evidence"]["frames_used"] == 2
    assert video["media"]["width"] == 6
    assert video["media"]["height"] == 4
    assert video["media"]["duration_s"] > 0
    assert video["media"]["codec"]
    assert night_products.validate_product_receipt(row, task, tmp_path / "attempt") == receipt
