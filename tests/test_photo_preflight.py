"""Photo-run resource preflight + non-silent ledger booking failure.

Two behaviours:

1. ``preflight_photo_run`` gates disk/RAM (reusing the drive preflight
   primitives) before the photo run does any irreversible work, and surfaces a
   leftover unbooked-ledger marker so a prior silent booking miss can't lead to
   a silent double-book.
2. ``record_confirmed_upload`` reports success/failure so the photo
   orchestrator can persist a durable marker when a confirmed upload could not
   be booked — without re-raising (which would release already-uploaded photos,
   the release-after-upload bug).
"""

import json
import logging
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import content_ledger as cl
from gomoveshift.pipeline import preflight as drive_preflight
from gomoveshift.pipeline.photos import face_filter, preflight as photo_preflight
from gomoveshift.pipeline.photos.caption_generator import CaptionBatch, PhotoCaption
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.pipeline.photos.preflight import (
    LEDGER_UNBOOKED_MARKER,
    estimate_astro_run_disk_gb,
    estimate_astro_stack_ram_gb,
    preflight_astro_stack,
    preflight_photo_run,
)
from gomoveshift.pipeline.photos.scheduler import PhotoSlot

_GB = 1_073_741_824
_Usage = namedtuple("_Usage", "total used free")


class _Mem:
    def __init__(self, available):
        self.available = available


def _plenty(monkeypatch, disk_gb=400, ram_gb=16):
    monkeypatch.setattr(
        drive_preflight.shutil,
        "disk_usage",
        lambda p: _Usage(1000 * _GB, (1000 - disk_gb) * _GB, disk_gb * _GB),
    )
    monkeypatch.setattr(drive_preflight, "_virtual_memory", lambda: _Mem(int(ram_gb * _GB)))


def _caption():
    return PhotoCaption(
        photo_index=0,
        main_caption="a quiet morning on the coast",
        tiktok_caption="coast",
        hashtags=["travel"],
        journey_slug="a-journey",
        filename="photo-001.jpg",
        location="Coast",
    )


def _slot():
    return PhotoSlot(date="2026-08-04", time="10:00:00", day_of_week="Tuesday")


def test_preflight_raises_on_low_disk(tmp_path, monkeypatch):
    _plenty(monkeypatch)
    monkeypatch.setattr(
        drive_preflight.shutil,
        "disk_usage",
        lambda p: _Usage(1000 * _GB, 999 * _GB, 1 * _GB),  # 1 GB free < 2 GB floor
    )
    with pytest.raises(RuntimeError, match="disk space"):
        preflight_photo_run(tmp_path)


def test_preflight_raises_on_low_ram(tmp_path, monkeypatch):
    _plenty(monkeypatch)
    monkeypatch.setattr(drive_preflight, "_virtual_memory", lambda: _Mem(int(0.2 * _GB)))
    with pytest.raises(RuntimeError, match="RAM"):
        preflight_photo_run(tmp_path)


def test_preflight_passes_with_ample_resources(tmp_path, monkeypatch):
    _plenty(monkeypatch)
    preflight_photo_run(tmp_path)  # must not raise


def test_preflight_photo_run_checks_blur_provider_when_enabled(tmp_path, monkeypatch):
    _plenty(monkeypatch)
    mock_check = MagicMock()
    monkeypatch.setattr(photo_preflight, "check_blur_inference_provider", mock_check)
    preflight_photo_run(tmp_path, blur_enabled=True)
    mock_check.assert_called_once_with()


def test_preflight_photo_run_skips_blur_provider_when_disabled(tmp_path, monkeypatch):
    _plenty(monkeypatch)
    mock_check = MagicMock()
    monkeypatch.setattr(photo_preflight, "check_blur_inference_provider", mock_check)
    preflight_photo_run(tmp_path, blur_enabled=False)
    mock_check.assert_not_called()


def test_preflight_photo_run_resolves_blur_enabled_from_face_filter_when_unset(
    tmp_path, monkeypatch
):
    _plenty(monkeypatch)
    mock_check = MagicMock()
    monkeypatch.setattr(photo_preflight, "check_blur_inference_provider", mock_check)
    monkeypatch.setattr(face_filter, "_blur_enabled", False)
    preflight_photo_run(tmp_path)  # blur_enabled not passed -> resolves via face_filter
    mock_check.assert_not_called()


def test_preflight_photo_run_raises_on_cpu_only_provider_before_disk_or_ram_check(
    tmp_path, monkeypatch
):
    disk_check = MagicMock(side_effect=AssertionError("disk check must not run"))
    ram_check = MagicMock(side_effect=AssertionError("ram check must not run"))
    monkeypatch.setattr(photo_preflight, "check_disk_space", disk_check)
    monkeypatch.setattr(photo_preflight, "check_ram", ram_check)
    monkeypatch.setattr(
        photo_preflight,
        "check_blur_inference_provider",
        MagicMock(side_effect=RuntimeError("CUDAExecutionProvider is not available")),
    )
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        preflight_photo_run(tmp_path, blur_enabled=True)
    disk_check.assert_not_called()
    ram_check.assert_not_called()


def test_preflight_warns_on_leftover_unbooked_marker(tmp_path, monkeypatch, caplog):
    _plenty(monkeypatch)
    marker = tmp_path / LEDGER_UNBOOKED_MARKER
    marker.write_text(json.dumps({"slots": [{"date": "2026-08-04", "time": "10:00:00"}]}))
    with caplog.at_level(logging.WARNING):
        preflight_photo_run(tmp_path)
    assert any("LEDGER-UNBOOKED" in r.getMessage() for r in caplog.records)


def test_astro_ram_estimate_scales_with_frame_count():
    # The develop-chain/decode-overhead terms are frame-count-INDEPENDENT
    # (astro_develop.develop_stack runs once per stack, not once per input frame), so
    # quadrupling frame_count no longer quadruples the estimate the way the old
    # sequence-list-only model did -- growth is still monotonic, just sub-linear.
    small = estimate_astro_stack_ram_gb(frame_count=10, megapixels=24)
    large = estimate_astro_stack_ram_gb(frame_count=40, megapixels=24)
    assert large > small
    assert large < small * 4


def test_astro_run_disk_estimate_includes_every_linear_artifact_and_one_atomic_temporary():
    one = estimate_astro_run_disk_gb(
        chunk_count=1, megapixels=20.8768, keep_linear=True, write_tiffs=False
    )
    thirty_eight = estimate_astro_run_disk_gb(
        chunk_count=38, megapixels=20.8768, keep_linear=True, write_tiffs=False
    )

    # 26 bytes/pixel: two float32 RGB planes plus one uint16 count plane. The one-chunk
    # case holds TWO artifacts at peak because atomic publication keeps the temporary
    # beside any already-published set; 38 chunks hold 39 artifact-equivalents.
    payload_gib = 20.8768e6 * 26 / _GB
    assert one == pytest.approx(payload_gib * 2 * 1.10)
    assert thirty_eight == pytest.approx(payload_gib * 39 * 1.10)


def test_astro_run_disk_estimate_adds_tiffs_only_when_requested():
    linear_only = estimate_astro_run_disk_gb(
        chunk_count=4, megapixels=24, keep_linear=True, write_tiffs=False
    )
    both = estimate_astro_run_disk_gb(
        chunk_count=4, megapixels=24, keep_linear=True, write_tiffs=True
    )

    assert both > linear_only


def test_astro_ram_estimate_small_chunk_does_not_collapse():
    # The develop-chain/decode-overhead terms dominate a small chunk: a 3-frame estimate
    # must not be a third of a 9-frame one (that's exactly the undercount -- those
    # terms don't shrink just because fewer frames are being stacked).
    three = estimate_astro_stack_ram_gb(frame_count=3, megapixels=24)
    nine = estimate_astro_stack_ram_gb(frame_count=9, megapixels=24)
    assert three > nine / 3


def test_astro_ram_estimate_charges_each_concurrent_decode():
    """Parallel decode holds ``decode_workers`` frames in flight instead of one, and the
    gate must know it BEFORE the first decode -- an under-counting preflight is exactly
    what killed most chunks of a long astro run.

    Measured with a child-process peak-RSS benchmark over 8 real 20.9 MP frames (each
    worker count in a FRESH child, so no measurement inherits the previous run's retained
    heap): peak RSS 2.315 GiB serial -> 2.648 at 2 workers -> 3.332 at 4, a marginal
    ~0.34 GiB per extra worker. One decoded float32 RGB frame at that size is 0.234 GiB,
    so the marginal cost is ~1.45 frame-equivalents -- the frame itself plus rawpy's
    demosaic buffers. The model charges 2 to stay above it.

    Charged as frame-equivalents rather than a fitted GiB constant so it scales with
    megapixels on its own. ``_ASTRO_RAM_DECODE_OVERHEAD_GIB_PER_MP`` is deliberately NOT
    multiplied by the worker count: it is a documented residual that also absorbs the
    sigma-clip promotion and baseline interpreter overhead, neither of which is per-worker,
    and multiplying it would charge 5.85 GiB per extra thread at 20.9 MP and refuse runs
    that fit comfortably. A gate that refuses a fitting run is as broken as one that waves
    an OOM through.
    """
    serial = estimate_astro_stack_ram_gb(frame_count=10, megapixels=20.8768)
    four = estimate_astro_stack_ram_gb(frame_count=10, megapixels=20.8768, decode_workers=4)

    assert four > serial
    # 3 extra workers x 2 frame-equivalents x 0.2334 GiB = ~1.40 GiB
    assert four - serial == pytest.approx(1.40, abs=0.05)
    # Above the 0.34 GiB/worker actually measured, and nowhere near the 5.85 GiB/worker
    # that multiplying the residual would have charged.
    assert (four - serial) / 3 > 0.34
    assert (four - serial) / 3 < 1.0


def test_astro_ram_estimate_default_is_unchanged_by_the_worker_knob():
    """``decode_workers`` defaults to 1, so every existing caller's estimate is untouched
    -- the new term can only ever be added by a caller that actually asked for concurrency."""
    assert estimate_astro_stack_ram_gb(
        frame_count=10, megapixels=24
    ) == estimate_astro_stack_ram_gb(frame_count=10, megapixels=24, decode_workers=1)


def test_astro_ram_estimate_scales_with_pixel_count():
    small = estimate_astro_stack_ram_gb(frame_count=20, megapixels=12)
    large = estimate_astro_stack_ram_gb(frame_count=20, megapixels=48)
    assert large == pytest.approx(small * 4)
    assert large > small


def test_astro_ram_estimate_scales_with_dark_count():
    without_darks = estimate_astro_stack_ram_gb(frame_count=5, megapixels=20.9)
    with_darks = estimate_astro_stack_ram_gb(frame_count=5, megapixels=20.9, dark_count=5)
    assert with_darks > without_darks


def test_astro_ram_estimate_matches_reference_point():
    # A 20-frame chunk of 24 MP frames is ~5.4 GiB per float32 sequence-equivalent;
    # astro_stack.stack() holds only the caller's TWO frame lists live at once (banding
    # streams straight off each list, no whole-array copy), plus headroom -- see
    # preflight.py's derivation comment. Widened from the old (10, 18) now that the
    # develop-chain/decode-overhead terms add their own MP-scaling GiB on top.
    estimate = estimate_astro_stack_ram_gb(frame_count=20, megapixels=24)
    assert 20 < estimate < 30


def test_astro_ram_estimate_matches_measured_peak():
    # Acceptance gate: the estimate must never be LOWER than a measured peak.
    # 11.9 GB is the measured peak of a 5x15s ISO-6400 astro run at 5600x3728 (~20.9 MP) --
    # see the preflight.py derivation comments; the estimate was 2.9 GiB before the
    # develop-chain terms were added (astro_benchmark.py's child-process RSS
    # measurement is how a future run validates this model against reality).
    estimate = estimate_astro_stack_ram_gb(frame_count=5, megapixels=20.9)
    assert estimate >= 11.9


def test_preflight_astro_stack_raises_when_estimate_exceeds_available_ram(tmp_path, monkeypatch):
    _plenty(monkeypatch, ram_gb=16)
    with pytest.raises(RuntimeError, match="RAM"):
        preflight_astro_stack(tmp_path, frame_count=100, megapixels=24)


def test_preflight_astro_stack_passes_when_run_fits(tmp_path, monkeypatch):
    _plenty(monkeypatch, ram_gb=16)
    preflight_astro_stack(tmp_path, frame_count=5, megapixels=1)  # must not raise


def test_preflight_astro_stack_raises_on_low_disk(tmp_path, monkeypatch):
    _plenty(monkeypatch, ram_gb=16)
    monkeypatch.setattr(
        drive_preflight.shutil,
        "disk_usage",
        lambda p: _Usage(1000 * _GB, 999 * _GB, 1 * _GB),  # 1 GB free < 2 GB floor
    )
    with pytest.raises(RuntimeError, match="disk space"):
        preflight_astro_stack(tmp_path, frame_count=5, megapixels=1)


def _photo_entry(date="2026-08-04"):
    return cl.ContentEntry(
        date=date,
        time="10:00:00",
        content_type="photo",
        title="t",
        platform="all",
        status="scheduled",
        source="photo_pipeline",
    )


def test_record_confirmed_upload_returns_true_on_success(tmp_path):
    ledger_path = tmp_path / ".content_ledger.json"
    assert cl.record_confirmed_upload([_photo_entry()], ledger_path) is True


def test_record_confirmed_upload_returns_false_on_failure(tmp_path, monkeypatch, caplog):
    ledger_path = tmp_path / ".content_ledger.json"

    def _boom(self, path=None):
        raise OSError("disk full during ledger save")

    monkeypatch.setattr(cl.ContentLedger, "save", _boom)
    with caplog.at_level(logging.ERROR):
        result = cl.record_confirmed_upload([_photo_entry()], ledger_path)
    assert result is False
    assert any("LEDGER-BOOKING-FAILED" in r.getMessage() for r in caplog.records)


def test_book_slots_writes_marker_when_booking_fails(tmp_path, monkeypatch):
    orch = PhotoOrchestrator(output_dir=tmp_path, ledger_booker=lambda rows: False)
    orch._book_slots([_slot()], [_caption()], tmp_path / "week.csv")
    marker = tmp_path / LEDGER_UNBOOKED_MARKER
    assert marker.exists()
    data = json.loads(marker.read_text())
    assert data["slots"][0]["date"] == "2026-08-04"
    assert data["slots"][0]["time"] == "10:00:00"


def test_book_slots_no_marker_on_success(tmp_path, monkeypatch):
    orch = PhotoOrchestrator(output_dir=tmp_path, ledger_booker=lambda rows: True)
    orch._book_slots([_slot()], [_caption()], tmp_path / "week.csv")
    assert not (tmp_path / LEDGER_UNBOOKED_MARKER).exists()


def test_publish_runs_preflight_before_upload(tmp_path, monkeypatch):
    batch = CaptionBatch(
        week_start="2026-02-23",
        voice_style="voice_first_person",
        captions=[_caption()],
        reviewed=True,
    )
    yaml_path = tmp_path / "captions.yaml"
    batch.save(yaml_path)

    orch = PhotoOrchestrator(site_root=tmp_path / "site", output_dir=tmp_path)

    def _boom(output_dir, **kw):
        raise RuntimeError("Insufficient disk space at photo output")

    monkeypatch.setattr("gomoveshift.pipeline.photos.orchestrator.preflight_photo_run", _boom)

    with patch("gomoveshift.pipeline.photos.orchestrator.PipelineUploader") as up:
        with pytest.raises(RuntimeError, match="disk space"):
            orch.publish_batch(str(yaml_path))
        up.assert_not_called()


def test_photo_preflight_warns_advisory_when_ml_upscaler_unusable(tmp_path, monkeypatch, caplog):
    _plenty(monkeypatch)
    monkeypatch.setattr(
        photo_preflight,
        "check_ml_upscale",
        lambda profile_upscale: drive_preflight.MlUpscaleCheck(False, True, "weights: missing"),
    )
    with caplog.at_level(logging.WARNING):
        preflight_photo_run(tmp_path, upscale="ml", blur_enabled=False)
    assert "weights: missing" in caplog.text
    assert "resampled frame" in caplog.text


def test_photo_preflight_silent_when_upscale_off(tmp_path, monkeypatch, caplog):
    _plenty(monkeypatch)
    with caplog.at_level(logging.WARNING):
        preflight_photo_run(tmp_path, upscale="off", blur_enabled=False)
    assert "upscal" not in caplog.text.lower()
