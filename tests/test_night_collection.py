"""Read-only night collection discovery and typed requested products."""

import datetime as dt
import json
from pathlib import Path

import pytest

from gomoveshift.pipeline.photos import night_collection as module


class Meta:
    def __init__(self, iso=3200, exposure_s=15, capture_utc=None):
        self.iso = iso
        self.exposure_s = exposure_s
        self.capture_utc = capture_utc


def _raw(root: Path, name: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"raw-header")
    return path


def test_scan_groups_cross_midnight_lights_and_explicit_dark_folder(tmp_path, monkeypatch):
    lights = [_raw(tmp_path, f"card/DSC_{index}.NEF") for index in range(5)]
    dark = _raw(tmp_path, "card/darks/DARK_1.NEF")
    stamps = {
        path: dt.datetime(2026, 9, 14, 23, 59, 20) + dt.timedelta(seconds=20 * index)
        for index, path in enumerate(lights)
    }
    stamps[dark] = dt.datetime(2026, 9, 15, 0, 2)
    monkeypatch.setattr(module, "extract_exif_capture_timestamp", lambda path: stamps[path])
    monkeypatch.setattr(module, "extract_exif_capture_metadata", lambda _path: Meta())

    found = module.scan_collection(tmp_path / "card")

    assert found.light_count == 5
    assert found.likely_dark_count == 1
    assert found.unreadable == ()
    assert len(found.sequences) == 1
    assert found.sequences[0].frame_count == 5
    assert found.sequences[0].started_at.date() != found.sequences[0].ended_at.date()
    assert found.capture_span_s == 80


def test_scan_reports_unknown_metadata_and_validated_existing_linear(tmp_path, monkeypatch):
    raw = _raw(tmp_path, "card/unknown.nef")
    linear = tmp_path / "card" / "existing.linear"
    linear.mkdir()
    (linear / "manifest.json").write_text(json.dumps({"schema": 4, "payloads": {}}))
    broken = tmp_path / "card" / "broken.linear"
    broken.mkdir()
    (broken / "manifest.json").write_text("not json")
    monkeypatch.setattr(module, "extract_exif_capture_timestamp", lambda _path: None)
    monkeypatch.setattr(module, "extract_exif_capture_metadata", lambda _path: Meta(None, None))

    found = module.scan_collection(tmp_path / "card")

    assert found.light_count == 0
    assert found.unreadable == (raw,)
    assert found.existing_linear == (linear,)
    assert found.invalid_linear == (broken,)


def test_scan_explains_timing_confidence_and_sequence_gap(tmp_path, monkeypatch):
    lights = [_raw(tmp_path, f"card/DSC_{index:02d}.NEF") for index in range(10)]
    first = dt.datetime(2026, 9, 14, 23, 55)
    stamps = {
        path: first
        + (
            dt.timedelta(seconds=20 * index)
            if index < 5
            else dt.timedelta(minutes=5, seconds=20 * index)
        )
        for index, path in enumerate(lights)
    }
    monkeypatch.setattr(module, "extract_exif_capture_timestamp", lambda path: stamps[path])
    monkeypatch.setattr(
        module,
        "extract_exif_capture_metadata",
        lambda path: Meta(capture_utc=stamps[path].replace(tzinfo=dt.timezone.utc)),
    )

    summary = module.scan_collection(tmp_path / "card").as_dict()

    assert len(summary["sequences"]) == 2
    assert summary["sequences"][0]["timezone_confidence"] == "recorded_offset"
    assert summary["sequences"][1]["break_before"]["kind"] == "capture_gap"
    assert summary["sequences"][1]["break_before"]["gap_s"] > 60


def test_scan_splits_known_frame_geometry_and_discloses_unknown_geometry(tmp_path, monkeypatch):
    lights = [_raw(tmp_path, f"card/DSC_{index:02d}.NEF") for index in range(10)]
    first = dt.datetime(2026, 9, 14, 22, 0)
    stamps = {path: first + dt.timedelta(seconds=20 * index) for index, path in enumerate(lights)}
    dimensions = {
        path: ((6000, 4000) if index < 5 else (4000, 3000)) for index, path in enumerate(lights)
    }
    monkeypatch.setattr(module, "extract_exif_capture_timestamp", lambda path: stamps[path])
    monkeypatch.setattr(module, "extract_exif_capture_metadata", lambda _path: Meta())
    monkeypatch.setattr(
        module, "extract_exif_dimensions", lambda path: dimensions[path], raising=False
    )

    summary = module.scan_collection(tmp_path / "card").as_dict()

    assert [sequence["geometry"] for sequence in summary["sequences"]] == [
        {"width": 6000, "height": 4000, "confidence": "recorded"},
        {"width": 4000, "height": 3000, "confidence": "recorded"},
    ]
    assert summary["sequences"][1]["break_before"] == {
        "kind": "geometry_change",
        "from": [6000, 4000],
        "to": [4000, 3000],
    }

    monkeypatch.setattr(module, "extract_exif_dimensions", lambda _path: None, raising=False)
    unknown = module.scan_collection(tmp_path / "card").as_dict()
    assert len(unknown["sequences"]) == 1
    assert unknown["sequences"][0]["geometry"] == {
        "width": None,
        "height": None,
        "confidence": "unknown",
    }


def test_scan_reports_short_incompatible_run_instead_of_silently_omitting_it(tmp_path, monkeypatch):
    lights = [_raw(tmp_path, f"card/DSC_{index:02d}.NEF") for index in range(6)]
    first = dt.datetime(2026, 9, 14, 22, 0)
    stamps = {path: first + dt.timedelta(seconds=20 * index) for index, path in enumerate(lights)}
    monkeypatch.setattr(module, "extract_exif_capture_timestamp", lambda path: stamps[path])
    monkeypatch.setattr(
        module,
        "extract_exif_capture_metadata",
        lambda path: Meta(iso=100 if path == lights[0] else 3200),
    )
    monkeypatch.setattr(module, "extract_exif_dimensions", lambda _path: (6000, 4000))

    summary = module.scan_collection(tmp_path / "card").as_dict()

    assert [sequence["frame_count"] for sequence in summary["sequences"]] == [5]
    assert summary["unsequenced"] == [
        {
            "frames": [str(lights[0])],
            "frame_count": 1,
            "reason": "fewer_than_minimum_compatible_frames",
            "boundary_before": None,
            "boundary_after": {"kind": "iso_change", "from": 100.0, "to": 3200.0},
        }
    ]


def test_scan_explains_the_boundary_before_a_short_trailing_run(tmp_path, monkeypatch):
    lights = [_raw(tmp_path, f"card/DSC_{index:02d}.NEF") for index in range(6)]
    first = dt.datetime(2026, 9, 14, 22, 0)
    stamps = {path: first + dt.timedelta(seconds=20 * index) for index, path in enumerate(lights)}
    monkeypatch.setattr(module, "extract_exif_capture_timestamp", lambda path: stamps[path])
    monkeypatch.setattr(
        module,
        "extract_exif_capture_metadata",
        lambda path: Meta(iso=100 if path == lights[-1] else 3200),
    )
    monkeypatch.setattr(module, "extract_exif_dimensions", lambda _path: (6000, 4000))

    summary = module.scan_collection(tmp_path / "card").as_dict()

    assert [sequence["frame_count"] for sequence in summary["sequences"]] == [5]
    assert summary["unsequenced"] == [
        {
            "frames": [str(lights[-1])],
            "frame_count": 1,
            "reason": "fewer_than_minimum_compatible_frames",
            "boundary_before": {"kind": "iso_change", "from": 3200.0, "to": 100.0},
            "boundary_after": None,
        }
    ]


def test_scan_refuses_symlink_escape_and_non_directory(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    _raw(outside, "escape.nef")
    card = tmp_path / "card"
    card.mkdir()
    (card / "linked").symlink_to(outside, target_is_directory=True)

    found = module.scan_collection(card)
    assert found.light_count == 0
    assert found.refused == (card / "linked",)
    with pytest.raises(ValueError, match="directory"):
        module.scan_collection(outside / "escape.nef")


def test_requested_products_are_unique_typed_and_require_matching_results():
    request = module.ProductRequest.create(["best_stills", "timelapse", "timelapse"])
    assert request.products == (
        module.NightProduct.BEST_STILLS,
        module.NightProduct.TIMELAPSE,
    )
    still = module.ProductResult(
        product=module.NightProduct.BEST_STILLS,
        path=Path("keeper.tif"),
        receipt=Path("keeper.receipt.json"),
    )
    assert request.missing([still]) == (module.NightProduct.TIMELAPSE,)
    assert not request.complete([still])
    video = module.ProductResult(
        product=module.NightProduct.TIMELAPSE,
        path=Path("night.mp4"),
        receipt=Path("night.receipt.json"),
    )
    assert request.complete([still, video])
    with pytest.raises(ValueError, match="supported"):
        module.ProductRequest.create(["video"])
    with pytest.raises(ValueError, match="at least one"):
        module.ProductRequest.create([])


def test_product_result_rejects_wrong_media_extension():
    with pytest.raises(ValueError, match="timelapse"):
        module.ProductResult(
            product=module.NightProduct.TIMELAPSE,
            path=Path("not-a-video.tif"),
            receipt=Path("receipt.json"),
        )
    with pytest.raises(ValueError, match="still"):
        module.ProductResult(
            product=module.NightProduct.STAR_TRAILS,
            path=Path("not-a-still.mp4"),
            receipt=Path("receipt.json"),
        )
