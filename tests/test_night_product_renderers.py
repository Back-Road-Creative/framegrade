"""Distinct night products produce receipt-bound, type-correct private artifacts."""

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.photos.night_collection import (
    NightCollection,
    NightFrame,
    NightSequence,
    ProductRequest,
)
from gomoveshift.pipeline.photos.night_project import NightStore
from gomoveshift.pipeline.photos import night_products as module


def _project(tmp_path, products):
    card = tmp_path / "card"
    card.mkdir()
    frames = []
    for index in range(5):
        path = card / f"light-{index}.NEF"
        path.write_bytes(f"raw-{index}".encode())
        frames.append(
            NightFrame(
                path, dt.datetime(2026, 9, 14, 23, 59) + dt.timedelta(seconds=20 * index), 3200, 15
            )
        )
    dark = card / "darks" / "dark.NEF"
    dark.parent.mkdir()
    dark.write_bytes(b"dark")
    linears = []
    for index in range(2):
        path = card / f"stack-{index}.linear"
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({"schema": 4, "payloads": {}}))
        linears.append(path)
    collection = NightCollection(
        root=card,
        lights=tuple(frames),
        likely_darks=(dark,),
        unreadable=(),
        sequences=(NightSequence(tuple(frames)),),
        existing_linear=tuple(linears),
        invalid_linear=(),
        refused=(),
    )
    store = NightStore(tmp_path / "projects")
    identifier = store.create_product_request(
        "brand", collection, [tmp_path], ProductRequest.create(products), "natural"
    )
    return store.get("brand", identifier)


def _write(attempt: Path, name: str, data=b"valid-media") -> Path:
    path = attempt / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _media_ok(_product, _path, _spec):
    return True


def test_best_stills_render_every_linear_and_publish_aggregate_receipt(tmp_path):
    row = _project(tmp_path, ["best_stills"])
    task = row["tasks"][0]
    calls = []

    def render_still(source, destination, recipe):
        calls.append((source, destination, recipe))
        _write(destination, "master.tif")
        _write(destination, "preview.jpg")
        receipt = {"state": "complete", "source": str(source)}
        (destination / "receipt.json").write_text(json.dumps(receipt))
        return receipt

    engines = module.default_engines()
    engines["render_still"] = render_still
    attempt = tmp_path / "attempt-best"
    progress = []

    receipt = module.render_product(
        row,
        task,
        attempt,
        engines=engines,
        media_validator=_media_ok,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert receipt["state"] == "complete"
    assert len(calls) == 2
    assert [item["role"] for item in receipt["outputs"]].count("master") == 2
    assert all(call[2] == "natural" for call in calls)
    assert progress == [(1, 2), (2, 2)]
    assert module.validate_product_receipt(row, task, attempt, media_validator=_media_ok) == receipt


def test_timelapse_receipt_is_partial_when_real_frames_drop_and_validates_video(tmp_path):
    row = _project(tmp_path, ["timelapse"])
    task = row["tasks"][0]
    seen = {}
    engines = module.default_engines()
    engines["raw_megapixels"] = lambda _path: 20.9
    engines["preflight_timelapse"] = lambda out, **kw: seen.update(preflight=(out, kw))

    def render(lights, output, darks, **kwargs):
        seen.update(lights=lights, darks=darks, kwargs=kwargs)
        kwargs["progress"](4, len(lights))
        _write(output.parent, output.name, b"mp4")
        return output, 4, ((Path(lights[-1]), "decode failed"),)

    engines["render_timelapse"] = render
    probes = []

    def validate_media(product, path, spec):
        probes.append((product.value, path, spec["duration_s"]))
        return True

    attempt = tmp_path / "attempt-video"
    progress = []
    receipt = module.render_product(
        row,
        task,
        attempt,
        engines=engines,
        media_validator=validate_media,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert receipt["state"] == "partial"
    assert receipt["evidence"]["frames_requested"] == 5
    assert receipt["evidence"]["frames_used"] == 4
    assert receipt["evidence"]["missing_intervals"][0]["path"].endswith("light-4.NEF")
    assert receipt["evidence"]["missing_intervals"][0]["captured_at"].startswith("2026-09-15")
    assert seen["preflight"][1] == {"megapixels": 20.9, "dark_count": 1}
    assert probes and probes[0][0] == "timelapse"
    assert progress == [(4, 5)]


@pytest.mark.parametrize(
    ("product", "filename", "engine_name"),
    [("star_trails", "trails.tif", "render_trail"), ("deep_still", "deep.tif", "render_deep")],
)
def test_still_product_types_use_their_existing_engines(tmp_path, product, filename, engine_name):
    row = _project(tmp_path, [product])
    task = row["tasks"][0]
    engines = module.default_engines()
    engines["raw_megapixels"] = lambda _path: 20.9
    engines["preflight_trail"] = lambda *_a, **_kw: None
    engines["render_preview"] = lambda _master, output: _write(output.parent, output.name, b"jpeg")

    if product == "star_trails":

        def render(lights, output, darks, **kwargs):
            _write(output.parent, output.name)
            kwargs["progress"](len(lights), len(lights))
            return output, len(lights), ()

        engines[engine_name] = render
    else:

        def render(linears, output):
            image = _write(output.parent, filename)
            coverage = _write(output.parent, "deep.coverage.npy")
            sidecar = _write(output.parent, "deep.deep.json")
            return SimpleNamespace(image=image, coverage=coverage, sidecar=sidecar)

        engines[engine_name] = render

    progress = []
    receipt = module.render_product(
        row,
        task,
        tmp_path / f"attempt-{product}",
        engines=engines,
        media_validator=_media_ok,
        progress=lambda completed, total: progress.append((completed, total)),
    )
    assert receipt["state"] == "complete"
    assert any(item["primary"] and item["path"].endswith(filename) for item in receipt["outputs"])
    assert any(
        item["role"] == "preview" and item["path"] == "preview.jpg" for item in receipt["outputs"]
    )
    assert progress[-1] == (
        5 if product == "star_trails" else 1,
        5 if product == "star_trails" else 1,
    )


def test_browser_preview_is_profiled_and_derived_from_real_master_pixels(tmp_path):
    import numpy as np
    from PIL import Image

    from gomoveshift.pipeline.photos.develop import export_tiff16

    master = export_tiff16(
        np.array([[[0, 257, 65535], [32768, 10000, 5000]]], dtype=np.uint16),
        tmp_path / "master.tif",
    )

    preview = module.render_master_preview(master, tmp_path / "preview.jpg")

    with Image.open(preview) as image:
        image.load()
        assert image.format == "JPEG"
        assert image.size == (2, 1)
        assert image.mode == "RGB"
        assert image.info.get("icc_profile")


def test_wrong_media_type_and_changed_input_fail_before_a_reusable_receipt(tmp_path):
    row = _project(tmp_path, ["timelapse"])
    task = row["tasks"][0]
    engines = module.default_engines()

    def wrong(_row, attempt):
        return module.RenderedProduct(
            (module.ProductOutput(_write(attempt, "still.tif"), "video", True),)
        )

    with pytest.raises(ValueError, match="MP4"):
        module.render_product(
            row,
            task,
            tmp_path / "wrong",
            renderers={"timelapse": wrong},
            engines=engines,
            media_validator=_media_ok,
        )

    Path(row["inputs"]["lights"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        module.render_product(
            row,
            task,
            tmp_path / "changed",
            renderers={"timelapse": wrong},
            engines=engines,
            media_validator=_media_ok,
        )


def test_receipt_reuse_refuses_tampered_bytes(tmp_path):
    row = _project(tmp_path, ["timelapse"])
    task = row["tasks"][0]

    def render(_row, attempt):
        return module.RenderedProduct(
            (module.ProductOutput(_write(attempt, "night.mp4"), "video", True),),
            evidence={"frames_used": 5, "fps": 24, "duration_s": 5 / 24},
        )

    attempt = tmp_path / "attempt"
    module.render_product(
        row,
        task,
        attempt,
        renderers={"timelapse": render},
        media_validator=_media_ok,
    )
    (attempt / "night.mp4").write_bytes(b"tampered")
    assert module.validate_product_receipt(row, task, attempt, media_validator=_media_ok) is None


def test_default_still_validation_records_real_tiff_shape_and_dtype(tmp_path):
    import numpy as np
    import tifffile

    row = _project(tmp_path, ["best_stills"])
    task = row["tasks"][0]

    def render(_row, attempt):
        path = attempt / "master.tif"
        tifffile.imwrite(path, np.zeros((4, 6, 3), dtype=np.uint16))
        return module.RenderedProduct((module.ProductOutput(path, "master", True),))

    receipt = module.render_product(
        row, task, tmp_path / "real-tiff", renderers={"best_stills": render}
    )
    assert receipt["outputs"][0]["media"] == {"shape": [4, 6, 3], "dtype": "uint16"}
