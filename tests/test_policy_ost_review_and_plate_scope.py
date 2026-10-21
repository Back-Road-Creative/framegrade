"""Acceptance tests for the OST-review and plate-scope decisions.

Policy home: ``docs/pipeline/158-production-video-policy.md`` sections 8 and 9.

* OST: routine on-screen text moves to a Waypost-owned automated review; until
  that path is built and qualified, the existing confirmation gate stays and no
  automation may pass ``--ost-audit-confirmed``.
* Plates: in scope regardless of apparent readability. No size floor on the
  plate path, so a ~15 px plate the detector finds is blurred like any other.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.config import DriveConfig

SERVICE = Path(__file__).resolve().parents[1]
POLICY_DOC = SERVICE / "docs" / "pipeline" / "158-production-video-policy.md"

# Everything that launches or schedules a drive. Definitions of the flag
# (parser, gate, flag registry) live under gomoveshift/pipeline and cli and are
# not launchers, so they are not scanned.
_LAUNCHER_ROOTS = [
    SERVICE / "bin",
    SERVICE / "gomoveshift" / "console",
    SERVICE / "gomoveshift" / "health",
]
_LAUNCHER_SUFFIXES = {".sh", ".cron", ".py", ".yaml", ".yml", ".toml", ".service", ".timer"}
_AUTO_CONFIRM = re.compile(r"--ost-audit-confirmed|ost_audit_confirmed\s*=\s*True")


def _policy_text() -> str:
    return POLICY_DOC.read_text(encoding="utf-8")


def test_policy_names_the_ost_reviewer_decision():
    text = _policy_text()
    assert "Who confirms in production: unverified" not in text
    assert "Waypost-owned" in text
    assert "existing confirmation gate stays" in text


def test_policy_records_no_unreadable_plate_exemption():
    text = _policy_text()
    assert "## 9. Plate scope" in text
    assert "no unreadable-plate exemption" in text


def test_no_launcher_auto_confirms_the_ost_gate():
    hits = []
    for root in _LAUNCHER_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in _LAUNCHER_SUFFIXES:
                continue
            if ".venv" in path.parts or "tests" in path.parts:
                continue
            try:
                body = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if _AUTO_CONFIRM.search(body):
                hits.append(str(path.relative_to(SERVICE)))
    assert hits == [], f"automation passes the OST confirmation: {hits}"


def test_drive_config_has_no_plate_size_floor():
    floors = [
        name
        for name in DriveConfig.model_fields
        if name.startswith("blur_plate")
        and re.search(r"min_(px|width|height|area|size|side|w|h)\b", name)
    ]
    assert floors == []


class _PaintedPlate:
    """Stub plate detector: reports the painted (non-zero) rectangle in whatever
    frame or tile crop it is handed, in that crop's own coordinates."""

    def detect(self, frame, conf):
        ys, xs = np.nonzero(frame[:, :, 0])
        if xs.size == 0:
            return []
        x0, y0 = int(xs.min()), int(ys.min())
        return [(x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)]


@pytest.mark.parametrize(
    "frame_w,frame_h,box",
    [
        (1920, 1080, (900, 600, 15, 6)),  # ~15 px plate at 1080p
        (3840, 2160, (1800, 1200, 15, 6)),  # same plate at 4K, tiled
        (1080, 1920, (500, 1400, 15, 6)),  # ~15 px plate in a 1080x1920 Short
    ],
)
def test_small_plate_survives_the_plate_filters(frame_w, frame_h, box):
    from redactcam.detect import detect_plates, frame_tiles

    drive = DriveConfig()
    _, _, bw, bh = box
    # The detector's aspect band (redactcam YoloDetector.detect) uses these bounds.
    assert bw >= drive.blur_plate_min_aspect * bh
    assert bw <= drive.blur_plate_max_aspect * bh

    x, y = box[0], box[1]
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frame[y : y + bh, x : x + bw] = 255
    tiles = frame_tiles(frame_w, frame_h, drive.blur_tile_px, drive.blur_tile_overlap)
    kept = detect_plates(frame, _PaintedPlate(), 0.5, tiles, drive.blur_plate_max_area_frac)
    assert (x, y, bw, bh) in [tuple(b) for b in kept], kept
