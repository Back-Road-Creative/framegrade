"""Tests for ``route_story.verify``: offline re-checking of a Route Story
receipt (a bundle manifest or a proof-clip receipt).

A fake ``video_prober`` stands in for a real ffprobe call so these tests
stay fast and deterministic; ``_default_video_prober``'s wiring to
``FFprobeClient`` is covered separately (it returns ``None`` on a missing
file, exercising the exact path ``verify_receipt`` falls back on).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story import bundle, verify

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _write_minimal_bundle(root: Path, *, zoom: int = 6) -> None:
    (root / "labels.geojson").write_text('{"type": "FeatureCollection", "features": []}\n')
    (root / "route.geojson").write_text('{"type": "FeatureCollection", "features": []}\n')
    for kind, suffix in (("imagery", "jpg"), ("terrain", "png")):
        tile_dir = root / kind / str(zoom) / "0"
        tile_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (bundle.TILE_SIZE, bundle.TILE_SIZE), (12, 34, 56)).save(
            tile_dir / f"0.{suffix}"
        )


def _fake_probe(*, width: int = 1920, height: int = 1080, duration: float = 2.0):
    def probe(path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": width, "height": height}],
            "format": {"duration": str(duration)},
        }

    return probe


def _write_proof_receipt(
    tmp_path: Path, *, bundle_root: Path, gpx_path: Path, output_path: Path
) -> Path:
    receipt = {
        "schema": verify.PROOF_RECEIPT_SCHEMA,
        "output": str(output_path),
        "gpx": str(gpx_path),
        "bundle": str(bundle_root),
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "frame_count": 60,
        "duration_seconds": 2.0,
    }
    receipt_path = tmp_path / "out.mp4.receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt_path


def _prepared_proof_receipt(tmp_path: Path) -> Path:
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    _write_minimal_bundle(bundle_root)
    bundle.write_manifest(bundle_root, {})

    gpx_path = tmp_path / "trip.gpx"
    gpx_path.write_text("<gpx></gpx>")

    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"not a real video, just needs to exist")

    return _write_proof_receipt(
        tmp_path, bundle_root=bundle_root, gpx_path=gpx_path, output_path=output_path
    )


# ---------------------------------------------------------------------------
# proof receipt: happy path + tampering
# ---------------------------------------------------------------------------


def test_verify_receipt_proof_fixture_passes(tmp_path: Path) -> None:
    receipt_path = _prepared_proof_receipt(tmp_path)

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe())

    assert report.kind == "proof"
    assert report.ok, report.as_dict()


def test_verify_receipt_proof_flags_tampered_bundle_checksum(tmp_path: Path) -> None:
    receipt_path = _prepared_proof_receipt(tmp_path)
    bundle_root = tmp_path / "bundle"
    (bundle_root / "labels.geojson").write_text("tampered")

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe())

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("checksum mismatch" in d and "labels.geojson" in d for d in failures)


def test_verify_receipt_proof_flags_missing_referenced_file(tmp_path: Path) -> None:
    receipt_path = _prepared_proof_receipt(tmp_path)
    bundle_root = tmp_path / "bundle"
    (bundle_root / "labels.geojson").unlink()

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe())

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("missing file" in d and "labels.geojson" in d for d in failures)


def test_verify_receipt_proof_flags_missing_output(tmp_path: Path) -> None:
    receipt_path = _prepared_proof_receipt(tmp_path)
    (tmp_path / "out.mp4").unlink()

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe())

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("output missing" in d for d in failures)


def test_verify_receipt_proof_flags_dimension_mismatch(tmp_path: Path) -> None:
    receipt_path = _prepared_proof_receipt(tmp_path)

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe(width=1280, height=720))

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("expected 1920x1080" in d for d in failures)


def test_verify_receipt_proof_flags_duration_mismatch(tmp_path: Path) -> None:
    receipt_path = _prepared_proof_receipt(tmp_path)

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe(duration=99.0))

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("expected ~2.000s" in d for d in failures)


def test_verify_receipt_proof_flags_missing_gpx_source(tmp_path: Path) -> None:
    receipt_path = _prepared_proof_receipt(tmp_path)
    (tmp_path / "trip.gpx").unlink()

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe())

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("gpx source missing" in d for d in failures)


def test_verify_receipt_proof_missing_required_key(tmp_path: Path) -> None:
    receipt_path = tmp_path / "out.mp4.receipt.json"
    receipt_path.write_text(
        json.dumps({"schema": verify.PROOF_RECEIPT_SCHEMA, "output": "out.mp4"})
    )

    report = verify.verify_receipt(receipt_path, video_prober=_fake_probe())

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("missing keys" in d for d in failures)


# ---------------------------------------------------------------------------
# bundle manifest receipt
# ---------------------------------------------------------------------------


def test_verify_receipt_bundle_manifest_fixture_passes(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    manifest_path = bundle.write_manifest(tmp_path, {})

    report = verify.verify_receipt(manifest_path)

    assert report.kind == "bundle_manifest"
    assert report.ok, report.as_dict()


def test_verify_receipt_bundle_manifest_flags_tampered_checksum(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    manifest_path = bundle.write_manifest(tmp_path, {})
    (tmp_path / "route.geojson").write_text("tampered")

    report = verify.verify_receipt(manifest_path)

    assert not report.ok
    failures = [c.detail for c in report.checks if not c.ok]
    assert any("checksum mismatch" in d and "route.geojson" in d for d in failures)


# ---------------------------------------------------------------------------
# unknown / malformed receipts
# ---------------------------------------------------------------------------


def test_verify_receipt_unknown_schema_fails_loudly(tmp_path: Path) -> None:
    receipt_path = tmp_path / "mystery.json"
    receipt_path.write_text(json.dumps({"schema": "something-else/1"}))

    report = verify.verify_receipt(receipt_path)

    assert report.kind == "unknown"
    assert not report.ok


def test_verify_receipt_no_schema_key_fails_loudly(tmp_path: Path) -> None:
    receipt_path = tmp_path / "mystery.json"
    receipt_path.write_text(json.dumps({"some": "thing"}))

    report = verify.verify_receipt(receipt_path)

    assert report.kind == "unknown"
    assert not report.ok


def test_verify_receipt_missing_file_fails_loudly(tmp_path: Path) -> None:
    report = verify.verify_receipt(tmp_path / "does-not-exist.json")

    assert report.kind == "unknown"
    assert not report.ok


def test_verify_report_ok_is_false_with_no_checks() -> None:
    assert verify.VerifyReport(kind="empty").ok is False


# ---------------------------------------------------------------------------
# default video prober wiring
# ---------------------------------------------------------------------------


def test_default_video_prober_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert verify._default_video_prober(tmp_path / "nope.mp4") is None


# ---------------------------------------------------------------------------
# CLI parser + handler
# ---------------------------------------------------------------------------


def test_route_story_verify_parser_accepts_receipt_path() -> None:
    from gomoveshift.cli.parser import _build_parser

    args = _build_parser().parse_args(["route-story-verify", "out.mp4.receipt.json"])

    assert args.receipt == Path("out.mp4.receipt.json")


def test_cmd_route_story_verify_exits_zero_on_pass(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from gomoveshift.cli.handlers.route_story_verify import cmd_route_story_verify

    receipt_path = _prepared_proof_receipt(tmp_path)
    monkeypatch.setattr(
        "gomoveshift.pipeline.route_story.verify._default_video_prober", _fake_probe()
    )

    exit_code = cmd_route_story_verify(SimpleNamespace(receipt=receipt_path), config=None)

    assert exit_code == 0


def test_cmd_route_story_verify_exits_nonzero_on_tamper(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from gomoveshift.cli.handlers.route_story_verify import cmd_route_story_verify

    receipt_path = _prepared_proof_receipt(tmp_path)
    (tmp_path / "bundle" / "labels.geojson").write_text("tampered")
    monkeypatch.setattr(
        "gomoveshift.pipeline.route_story.verify._default_video_prober", _fake_probe()
    )

    exit_code = cmd_route_story_verify(SimpleNamespace(receipt=receipt_path), config=None)

    assert exit_code == 1
