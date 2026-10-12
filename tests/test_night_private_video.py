"""Night timelapse keepers reuse the verified video privacy pipeline."""

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gomoveshift.pipeline.day_master_encoder import REQUIRED_PRIVACY_CLASSES
from gomoveshift.pipeline.photos import night_private_video as module


def _privacy(state="verified"):
    return {name: {"state": state, "reason": "test"} for name in REQUIRED_PRIVACY_CLASSES}


def _row(tmp_path):
    project = "a" * 64
    attempt = tmp_path / project / "0-timelapse-attempt"
    attempt.mkdir(parents=True)
    source = attempt / "timelapse.mp4"
    source.write_bytes(b"source-video")
    row = {
        "schema": 2,
        "kind": "night_products",
        "id": project,
        "brand": "brand",
        "build": {"night_products.py": "build-sha"},
        "runtime": {"python": "3.12"},
        "tasks": [
            {
                "product": "timelapse",
                "status": "complete",
                "output": str(attempt),
            }
        ],
    }
    receipt = {
        "outputs": [
            {
                "path": source.name,
                "role": "video",
                "size": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ]
    }
    return row, source, receipt


def test_private_video_blurs_copy_strips_location_and_reuses_receipt(tmp_path, monkeypatch):
    row, source, receipt = _row(tmp_path)
    calls = []

    def render(actual_source, output, mask):
        calls.append((actual_source, mask))
        output.write_bytes(b"blurred-video")
        return {
            "masked_frames": 2,
            "decoded_frames": 10,
            "cabin_coverage_verified": True,
            "cabin_vehicle_frames": 3,
        }

    monkeypatch.setattr(
        module,
        "_validated_video",
        lambda path, expected_sha, expected_size: (
            path.stat().st_size == expected_size
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
        ),
    )
    kwargs = {
        "receipt_validator": lambda *_args: receipt,
        "privacy_reporter": lambda **_kwargs: _privacy(),
        "renderer": render,
    }
    first = module.prepare_private_video(row, 0, tmp_path, **kwargs)
    second = module.prepare_private_video(row, 0, tmp_path, **kwargs)

    assert first == second
    assert len(calls) == 1 and calls[0][0] == source
    assert first["media_type"] == "video"
    assert first["masked_frames"] == 2
    assert first["cabin_coverage_verified"] is True
    assert first["location_metadata_absent"] is True
    assert first["publishable"] is False
    assert "night_private_video.py" in first["export_build"]
    assert "redactcam" in first["export_runtime"]["packages"]
    output = tmp_path / row["id"] / first["path"]
    assert module.validate_private_video(row, first, tmp_path) == output
    output.write_bytes(b"tampered")
    assert module.validate_private_video(row, first, tmp_path) is None


def test_private_video_refuses_unattested_class_before_render(tmp_path, monkeypatch):
    row, _source, receipt = _row(tmp_path)
    privacy = _privacy()
    privacy["vehicle_occupants"] = {"state": "inconclusive", "reason": "model disabled"}
    render = Mock()
    monkeypatch.setattr(module, "_validated_video", lambda *_args: True)

    with pytest.raises(ValueError, match="vehicle_occupants"):
        module.prepare_private_video(
            row,
            0,
            tmp_path,
            receipt_validator=lambda *_args: receipt,
            privacy_reporter=lambda **_kwargs: privacy,
            renderer=render,
        )

    render.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"format": {"tags": {}}, "streams": [{"tags": {}}]}, True),
        ({"format": {"tags": {"location": "+1-2/"}}}, False),
        ({"streams": [{"tags": {"com.apple.quicktime.location.ISO6709": "+1-2/"}}]}, False),
        ({"format": {"tags": {"GPSCoordinates": "1,2"}}}, False),
    ],
)
def test_location_metadata_probe_fails_closed(monkeypatch, tmp_path, payload, expected):
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=module.json.dumps(payload)),
    )

    assert module._location_metadata_absent(tmp_path / "keeper.mp4") is expected


def test_cabin_coverage_refuses_an_exposed_vehicle_occupant(monkeypatch, tmp_path):
    drive = SimpleNamespace(
        blur_vehicle_cabin_frac=0.6,
        blur_vehicle_conf=0.5,
        blur_vehicle_min_height_frac=0.07,
        blur_vehicle_ignore_bottom_frac=0.12,
        blur_plate_max_area_frac=0.006,
    )
    leak = SimpleNamespace(frame=5, coverage=0.0)
    report = SimpleNamespace(ok=False, leaks=[leak], vehicle_frames=3)
    monkeypatch.setattr(
        "gomoveshift.analysis.verify_cabin_coverage", lambda *_args, **_kwargs: report
    )

    with pytest.raises(ValueError, match="1 of 3"):
        module._verify_cabin_mask(
            tmp_path / "source.mp4", tmp_path / "mask.mkv", drive, tmp_path / "vehicle", None
        )


@pytest.mark.parametrize(
    ("change", "validated", "message"),
    [
        (lambda row: row["tasks"].clear(), {}, "unknown night product task"),
        (lambda row: row["tasks"][0].update(product="best_stills"), {}, "timelapse product"),
        (lambda _row: None, None, "validated night product receipt"),
        (
            lambda _row: None,
            {"outputs": [{"role": "video", "path": "bad.mp4", "size": 0}]},
            "validated timelapse size",
        ),
    ],
)
def test_private_video_estimate_fails_closed(tmp_path, change, validated, message):
    row, _source, _receipt = _row(tmp_path)
    change(row)
    with pytest.raises(ValueError, match=message):
        module.estimate_private_video_bytes(row, 0, receipt_validator=lambda *_args: validated)


@pytest.mark.parametrize(
    ("change", "validated", "message"),
    [
        (lambda row: row.update(schema=1), {}, "typed night product project"),
        (lambda row: row["tasks"].clear(), {}, "unknown night product task"),
        (lambda row: row["tasks"][0].update(status="failed"), {}, "completed timelapse"),
        (lambda _row: None, None, "validated night product receipt"),
        (lambda _row: None, {"outputs": []}, "validated timelapse video"),
    ],
)
def test_private_video_export_rejects_invalid_request_or_receipt(
    tmp_path, change, validated, message
):
    row, _source, _receipt = _row(tmp_path)
    change(row)
    with pytest.raises(ValueError, match=message):
        module.prepare_private_video(row, 0, tmp_path, receipt_validator=lambda *_args: validated)
