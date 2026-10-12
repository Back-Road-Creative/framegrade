"""Distribution-safe copy of a film_receipt -- see
``docs/pipeline/150-route-story-route-privacy.md`` and
``docs/pipeline/16-route-story-film.md`` ("Route trim").
"""

from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story.film import MapLabel, film_receipt
from gomoveshift.pipeline.route_story.public_receipt import (
    PublicReceiptRefusedError,
    public_receipt,
)
from gomoveshift.pipeline.route_story.route_privacy import RouteTrim, receipt_fragment, trim_route
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from route_story_fixtures import driving_points


def _raster(tmp_path: Path, name: str, credit: str) -> SatelliteRaster:
    path = tmp_path / name
    Image.new("RGB", (32, 16), "#446677").save(path)
    return SatelliteRaster(path, -180, -90, 180, 90, credit)


def _trimmed_receipt(tmp_path: Path, *, start_m: float = 25.0, end_m: float = 0.0, labels=()):
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    trim_result = trim_route([segment], RouteTrim(start_m=start_m, end_m=end_m))
    national = _raster(tmp_path, "national.jpg", "National credit")
    regional = _raster(tmp_path, "regional.jpg", "Regional credit")
    local = _raster(tmp_path, "local.jpg", "Local credit")
    receipt = film_receipt(
        segments=trim_result.segments,
        national=national,
        local=local,
        regional=[regional],
        photos=[],
        labels=labels,
        route_trim=receipt_fragment(trim_result)["route_trim"],
    )
    return receipt, trim_result


def test_refuses_when_trim_result_is_none(tmp_path: Path):
    receipt, _ = _trimmed_receipt(tmp_path)

    with pytest.raises(PublicReceiptRefusedError):
        public_receipt(receipt, None)


def test_refuses_when_trim_is_zero(tmp_path: Path):
    receipt, trim_result = _trimmed_receipt(tmp_path, start_m=0.0, end_m=0.0)

    with pytest.raises(PublicReceiptRefusedError):
        public_receipt(receipt, trim_result)


def test_refuses_when_receipt_has_no_route_trim(tmp_path: Path):
    segment = driving_points(10, speed_mps=20.0, dt=1.0)
    trim_result = trim_route([segment], RouteTrim(start_m=25.0))
    national = _raster(tmp_path, "national.jpg", "National credit")
    local = _raster(tmp_path, "local.jpg", "Local credit")
    # route_trim intentionally omitted from the receipt.
    receipt = film_receipt(segments=trim_result.segments, national=national, local=local, photos=[])

    with pytest.raises(PublicReceiptRefusedError):
        public_receipt(receipt, trim_result)


def test_strips_imagery_to_credit_only(tmp_path: Path):
    receipt, trim_result = _trimmed_receipt(tmp_path)

    sanitized = public_receipt(receipt, trim_result)

    assert sanitized["national_imagery"] == {"credit": "National credit"}
    assert sanitized["local_imagery"] == {"credit": "Local credit"}
    assert sanitized["regional_imagery"] == [{"credit": "Regional credit"}]
    for block in (sanitized["national_imagery"], sanitized["local_imagery"]):
        assert "bounds" not in block
        assert "sha256" not in block
        assert "dimensions" not in block


def test_strips_map_labels_to_a_count(tmp_path: Path):
    labels = [
        MapLabel("Secret Trailhead", 35.0, -80.0),
        MapLabel("Operator Driveway", 35.0002, -80.0002),
    ]
    receipt, trim_result = _trimmed_receipt(tmp_path, labels=labels)
    assert receipt["map_labels"]["count"] == 2
    assert receipt["map_labels"]["names"] == ["Secret Trailhead", "Operator Driveway"]

    sanitized = public_receipt(receipt, trim_result)

    assert sanitized["map_labels"] == {"count": 2}
    assert "names" not in sanitized["map_labels"]


def test_strips_photos_to_a_count(tmp_path: Path):
    receipt, trim_result = _trimmed_receipt(tmp_path)
    receipt["photos"] = [{"media_id": "a.jpg"}, {"media_id": "b.jpg"}]

    sanitized = public_receipt(receipt, trim_result)

    assert sanitized["photos"] == {"count": 2}


def test_keeps_route_trim_route_summary_outputs_and_map_renderer_verbatim(tmp_path: Path):
    receipt, trim_result = _trimmed_receipt(tmp_path)

    sanitized = public_receipt(receipt, trim_result)

    assert sanitized["route_trim"] == receipt["route_trim"]
    assert sanitized["route_summary"] == receipt["route_summary"]
    assert sanitized["outputs"] == receipt["outputs"]
    assert sanitized["map_renderer"] == receipt["map_renderer"]
