"""Ingest resource preflight: gate disk/RAM before the first decode.

``photos ingest`` is the heaviest photo path — a full-resolution decode plus a
multi-GB float64 grade transient per frame — yet historically ran with no
resource gate (``preflight_photo_run`` fires only at publish, with a RAM floor
sized for resize/upload, not develop). ``preflight_photo_ingest`` reuses the
same drive preflight primitives with a develop-sized RAM floor and runs at the
top of ``ingest()``, so a shortfall raises — as the ``PhotoIngestError`` the
CLI handler surfaces — before any decode or gallery write.
"""

import textwrap
from collections import namedtuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoveshift.pipeline import preflight as drive_preflight
from gomoveshift.pipeline.photos import preflight as photo_preflight
from gomoveshift.pipeline.photos.develop import DevelopedPhoto
from gomoveshift.pipeline.photos.ingest import PhotoIngestError, ingest
from gomoveshift.pipeline.photos.preflight import preflight_photo_ingest

_GB = 1_073_741_824
_Usage = namedtuple("_Usage", "total used free")

JOURNEY_MD = textwrap.dedent("""\
    ---
    title: Journey A
    days:
    - day_number: 1
      date: '2025-01-01'
      photos:
      - location: Place A
        photos:
        - photo-001.jpg
    ---
    Body text
""")
DAY = dict(day_date="2025-01-01", location="Place A")


class _Mem:
    def __init__(self, available):
        self.available = available


def _probes(monkeypatch, disk_gb=400, ram_gb=16):
    monkeypatch.setattr(
        drive_preflight.shutil,
        "disk_usage",
        lambda p: _Usage(1000 * _GB, (1000 - disk_gb) * _GB, disk_gb * _GB),
    )
    monkeypatch.setattr(drive_preflight, "_virtual_memory", lambda: _Mem(int(ram_gb * _GB)))


def _site(tmp_path):
    journey = tmp_path / "content" / "journeys" / "journey-a.md"
    journey.parent.mkdir(parents=True)
    journey.write_text(JOURNEY_MD)
    (tmp_path / "static" / "images" / "journeys" / "journey-a").mkdir(parents=True)
    return journey


def _card(tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    (card / "DSC_0001.NEF").write_bytes(b"raw bytes")
    return card


def _developed(path):
    return DevelopedPhoto(
        rgb=np.random.default_rng(7).integers(0, 256, (4, 4, 3), dtype=np.uint8),
        path=path,
        params=None,
        cluster_index=0,
    )


def _never_develop(path):
    raise AssertionError(f"develop must not be called for {path}")


def test_low_ram_raises_before_any_decode(tmp_path, monkeypatch):
    # 1.5 GB clears the publish-path floor but not the develop-sized one.
    _probes(monkeypatch, ram_gb=1.5)
    journey = _site(tmp_path)
    with pytest.raises(PhotoIngestError, match=r"RAM: 1\.50 GB available, need 4\.0 GB"):
        ingest(_card(tmp_path), journey, develop_fn=_never_develop, **DAY)
    assert journey.read_text() == JOURNEY_MD


def test_low_disk_raises_before_any_decode_or_gallery_write(tmp_path, monkeypatch):
    _probes(monkeypatch, disk_gb=1)
    journey = _site(tmp_path)
    with pytest.raises(PhotoIngestError, match=r"disk space at .*: 1\.0 GB free, need 2 GB"):
        ingest(_card(tmp_path), journey, develop_fn=_never_develop, **DAY)
    gallery = tmp_path / "static" / "images" / "journeys" / "journey-a"
    assert list(gallery.iterdir()) == []
    assert journey.read_text() == JOURNEY_MD


def test_healthy_probes_proceed(tmp_path, monkeypatch):
    _probes(monkeypatch)
    journey = _site(tmp_path)
    report = ingest(_card(tmp_path), journey, develop_fn=_developed, **DAY)
    assert report.ingested == ["dsc_0001.jpg"]


def test_preflight_photo_ingest_checks_blur_provider_when_enabled(tmp_path, monkeypatch):
    _probes(monkeypatch)
    mock_check = MagicMock()
    monkeypatch.setattr(photo_preflight, "check_blur_inference_provider", mock_check)
    preflight_photo_ingest(tmp_path, blur_enabled=True)
    mock_check.assert_called_once_with()


def test_preflight_photo_ingest_skips_blur_provider_when_disabled(tmp_path, monkeypatch):
    _probes(monkeypatch)
    mock_check = MagicMock()
    monkeypatch.setattr(photo_preflight, "check_blur_inference_provider", mock_check)
    preflight_photo_ingest(tmp_path, blur_enabled=False)
    mock_check.assert_not_called()


def test_ingest_passes_skip_blur_through_to_preflight_provider_check(tmp_path, monkeypatch):
    # ingest() must gate the SAME onnxruntime-provider check when it will run
    # blur -- a clobbered venv silently blurring gallery exports on CPU is the
    # identical failure the drive pipeline's preflight already refuses.
    _probes(monkeypatch)
    journey = _site(tmp_path)
    mock_check = MagicMock(side_effect=RuntimeError("CUDAExecutionProvider is not available"))
    monkeypatch.setattr(photo_preflight, "check_blur_inference_provider", mock_check)
    with pytest.raises(PhotoIngestError, match="CUDAExecutionProvider"):
        ingest(_card(tmp_path), journey, develop_fn=_never_develop, **DAY)
    mock_check.assert_called_once_with()


def test_ingest_skip_blur_true_never_checks_provider(tmp_path, monkeypatch):
    _probes(monkeypatch)
    journey = _site(tmp_path)
    mock_check = MagicMock()
    monkeypatch.setattr(photo_preflight, "check_blur_inference_provider", mock_check)
    ingest(_card(tmp_path), journey, develop_fn=_developed, skip_blur=True, **DAY)
    mock_check.assert_not_called()


def test_ingest_ram_floor_is_develop_sized_not_the_publish_floor(tmp_path, monkeypatch):
    from gomoveshift.pipeline.photos.preflight import preflight_photo_ingest, preflight_photo_run

    _probes(monkeypatch, ram_gb=3)
    preflight_photo_run(tmp_path)  # resize/upload floor: 3 GB passes
    with pytest.raises(RuntimeError, match="RAM"):
        preflight_photo_ingest(tmp_path)  # develop floor: 3 GB is a shortfall
