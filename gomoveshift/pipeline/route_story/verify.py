"""Offline re-verification of a Route Story receipt.

A receipt (a ``route-story-proof-receipt/2`` from ``proof.render_proof_clip``
or a bundle ``manifest.json`` written by ``bundle.write_manifest``) records
facts about an already-completed offline step. ``verify_receipt`` re-checks
every one of those facts without any network call: every file it references
still exists, a referenced bundle's sha256 manifest still matches the bytes
on disk, and -- when the receipt recorded an output video -- that video's
own dimensions and duration still match what the receipt says was rendered.

This module never fetches anything or trusts the receipt's own claims; it
independently recomputes what it can and reports pass/fail with a reason,
naming the specific file on a mismatch.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..validation.ffprobe_client import FFprobeClient
from . import bundle as bundle_module

VideoProber = Callable[[Path], Optional[dict]]

PROOF_RECEIPT_SCHEMA = "route-story-proof-receipt/2"
_REQUIRED_PROOF_KEYS = ("output", "gpx", "bundle", "width", "height", "duration_seconds")


@dataclass(frozen=True)
class Check:
    """One offline fact re-checked against a receipt."""

    name: str
    ok: bool
    detail: str = ""


@dataclass
class VerifyReport:
    """Every check run against one receipt, and whether they all passed."""

    kind: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name=name, ok=ok, detail=detail))

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "ok": self.ok,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
        }


def _default_video_prober(path: Path) -> Optional[dict]:
    client = FFprobeClient(subprocess)
    return client.probe_or_none(path, show_streams=True, show_format=True)


def verify_receipt(path: Path, *, video_prober: VideoProber | None = None) -> VerifyReport:
    """Re-check every referenced fact in the receipt at ``path``.

    Detects the receipt's kind from its top-level ``schema`` field: a
    bundle ``manifest.json`` (``bundle.MANIFEST_SCHEMA``) or a
    ``route-story-proof-receipt/2``. Any other/missing schema is reported
    as an ``unknown`` kind with a single failing check -- never silently
    treated as passing.
    """
    path = Path(path)
    if not path.is_file():
        report = VerifyReport(kind="unknown")
        report.add("receipt_file", False, f"receipt not found: {path}")
        return report

    data = json.loads(path.read_text(encoding="utf-8"))
    schema = data.get("schema")

    if schema in bundle_module.MANIFEST_SCHEMAS:
        return _verify_bundle_manifest(path.parent)
    if schema == PROOF_RECEIPT_SCHEMA:
        return _verify_proof_receipt(data, video_prober=video_prober or _default_video_prober)

    report = VerifyReport(kind="unknown")
    report.add("schema", False, f"unknown receipt kind: schema={schema!r}")
    return report


def _verify_bundle_manifest(bundle_root: Path) -> VerifyReport:
    report = VerifyReport(kind="bundle_manifest")
    problems = bundle_module.verify_bundle(bundle_root)
    if not problems:
        report.add("bundle_files", True, "all files present and checksums match")
        return report
    for problem in problems:
        report.add("bundle_files", False, problem)
    return report


def _verify_proof_receipt(data: dict, *, video_prober: VideoProber) -> VerifyReport:
    report = VerifyReport(kind="proof")

    missing = [key for key in _REQUIRED_PROOF_KEYS if key not in data]
    if missing:
        report.add("required_keys", False, f"missing keys: {', '.join(sorted(missing))}")
        return report
    report.add("required_keys", True)

    gpx_path = Path(data["gpx"])
    if gpx_path.is_file():
        report.add("gpx_source", True, str(gpx_path))
    else:
        report.add("gpx_source", False, f"gpx source missing: {gpx_path}")

    bundle_root = Path(data["bundle"])
    if not (bundle_root / "manifest.json").is_file():
        report.add("bundle", False, f"bundle manifest missing: {bundle_root / 'manifest.json'}")
    else:
        problems = bundle_module.verify_bundle(bundle_root)
        if not problems:
            report.add("bundle", True, "bundle files and checksums match")
        for problem in problems:
            report.add("bundle", False, problem)

    output_path = Path(data["output"])
    if not output_path.is_file():
        report.add("output_file", False, f"output missing: {output_path}")
        return report
    report.add("output_file", True, str(output_path))

    probed = video_prober(output_path)
    if not probed:
        report.add("output_media", False, f"ffprobe could not read output: {output_path}")
        return report

    streams = probed.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        report.add("output_media", False, f"no video stream in output: {output_path}")
        return report

    width = video_stream.get("width")
    height = video_stream.get("height")
    if width == data["width"] and height == data["height"]:
        report.add("output_dimensions", True, f"{width}x{height}")
    else:
        report.add(
            "output_dimensions",
            False,
            f"expected {data['width']}x{data['height']}, got {width}x{height}",
        )

    duration = float(probed.get("format", {}).get("duration", 0.0))
    expected_duration = float(data["duration_seconds"])
    fps = float(data.get("fps") or 30.0) or 30.0
    tolerance = max(2.0 / fps, 0.25)
    if abs(duration - expected_duration) <= tolerance:
        report.add("output_duration", True, f"{duration:.3f}s")
    else:
        report.add(
            "output_duration",
            False,
            f"expected ~{expected_duration:.3f}s, got {duration:.3f}s",
        )

    return report
