"""Verify the occupant CABIN blur actually covers drivers in a rendered output.

Independent of HOW the blur was produced: it detects vehicles in the ORIGINAL with
the same COCO detector the blur uses, then checks that the cabin region (the top
``cabin_frac`` of each vehicle box) was actually blurred in the rendered output —
measured as the fraction of cabin pixels the blur CHANGED (``|blurred - original|``)
plus the residual sharpness there. A cabin left unchanged/sharp is an exposed driver
= a leak. This gate measures every detected vehicle rather than eyeballing a few
frames.

CLI:
    python tools/verify_blur_coverage.py ORIGINAL.mp4 BLURRED.mp4 --vehicle-model PATH \
        [--cabin-frac 0.6] [--conf 0.35] [--sample-fps 8] [--min-coverage 0.5]
Exit 0 = every detected cabin adequately blurred; exit 1 = at least one exposed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from redactcam import (  # noqa: E402
    build_plate_detector,
    build_vehicle_detector,
    cabin_region,
    plate_inside,
)
from redactcam.detect import detect_plates, filter_vehicles, frame_tiles  # noqa: E402


@dataclass
class Leak:
    frame: int
    coverage: float
    sharpness: float
    box: tuple[int, int, int, int]


@dataclass
class CoverageReport:
    vehicle_frames: int = 0
    leaks: list[Leak] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.leaks


def verify_cabin_coverage(
    original: str | Path,
    blurred: str | Path,
    vehicle_model: str | Path,
    *,
    plate_model: str | Path | None = None,
    cabin_frac: float = 0.6,
    conf: float = 0.6,  # MUST match blur_vehicle_conf — else the verifier flags
    sample_fps: float = 8.0,  # non-occupant objects (signs, parked caravans) the blur skips
    diff_thresh: int = 12,
    min_coverage: float = 0.5,
    sharp_thresh: float = 60.0,
    min_height_frac: float = 0.07,
    ignore_bottom_frac: float = 0.12,
    plate_conf: float = 0.4,
    plate_max_area_frac: float = 0.006,
    infer_size: int = 640,
) -> CoverageReport:
    """Per-frame check that every detected vehicle's cabin region is blurred in the
    rendered output. A cabin is a LEAK when fewer than ``min_coverage`` of its pixels
    were changed by the blur AND its residual Laplacian variance exceeds ``sharp_thresh``
    (still detailed → an exposed driver, not a flat panel). ``min_coverage`` is 0.5, not
    1.0, because the blur is a CONTIGUOUS rectangle over the cabin while the verifier
    re-detects the vehicle independently — a few-percent box offset leaves the driver
    (cabin center) covered but trims edge pixels, so a real exposure reads near 0%
    coverage while a covered-but-offset car reads 60-80% (validated: an exposed
    windshield measured 0-20%, a covered one 63-76%)."""
    det = build_vehicle_detector(str(vehicle_model), infer_size, 0.45, 0)
    if det is None:
        raise SystemExit(f"vehicle model not resolvable: {vehicle_model}")
    # Plate detector for corroboration (only check vehicles that have a plate inside —
    # the same real-car test the blur uses). Without a plate model, fall back to
    # checking every detected vehicle (the old behaviour).
    plate_det = build_plate_detector(plate_model, infer_size, 0.45, 2.0, 6.0)
    cap_o = cv2.VideoCapture(str(original))
    cap_b = cv2.VideoCapture(str(blurred))
    if not (cap_o.isOpened() and cap_b.isOpened()):
        raise SystemExit("could not open one of the videos")
    fps = cap_o.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, round(fps / sample_fps)) if sample_fps > 0 else 1
    tiles = None
    rep = CoverageReport()
    idx = 0
    # Read both videos in lockstep (frame-aligned, no seeking — the blurred output is
    # the original blurred, so the indices match) and check only the sample frames.
    while True:
        ok_o, orig = cap_o.read()
        ok_b, blur = cap_b.read()
        if not (ok_o and ok_b):
            break
        if idx % step == 0:
            h, w = orig.shape[:2]
            if tiles is None:
                tiles = frame_tiles(w, h, 1280, 0.15)
            plates = (
                detect_plates(orig, plate_det, plate_conf, tiles, plate_max_area_frac)
                if plate_det is not None
                else None
            )
            # Same worth-blurring filter the blur uses (drop the ego hood + tiny cars),
            # so the gate checks the same vehicles the blur is responsible for.
            for bx in filter_vehicles(
                det.detect(orig, conf), h, min_height_frac, ignore_bottom_frac
            ):
                # Plate-corroborate: skip a vehicle with no plate inside it (a roadside
                # billboard/sign/caravan the blur also skips) — matches the blur's logic
                # so the gate doesn't flag what the blur intentionally leaves readable.
                if plates is not None and not plate_inside(bx, plates):
                    continue
                rep.vehicle_frames += 1
                x0, y0, x1, y1 = cabin_region(bx, cabin_frac, w, h)
                if x1 <= x0 or y1 <= y0:
                    continue
                co = orig[y0:y1, x0:x1]
                cb = blur[y0:y1, x0:x1]
                cov = float((cv2.absdiff(cb, co).max(axis=2) > diff_thresh).mean())
                sharp = cv2.Laplacian(cv2.cvtColor(cb, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if cov < min_coverage and sharp > sharp_thresh:
                    rep.leaks.append(Leak(idx, cov, float(sharp), bx))
        idx += 1
    cap_o.release()
    cap_b.release()
    return rep


def _main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify occupant cabin blur coverage.")
    ap.add_argument("original")
    ap.add_argument("blurred")
    ap.add_argument("--vehicle-model", required=True)
    ap.add_argument("--plate-model", default=None, help="plate ONNX for corroboration")
    ap.add_argument("--cabin-frac", type=float, default=0.6)
    ap.add_argument("--conf", type=float, default=0.6)
    ap.add_argument("--sample-fps", type=float, default=8.0)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--sharp-thresh", type=float, default=60.0)
    ap.add_argument("--min-height-frac", type=float, default=0.07)
    ap.add_argument("--ignore-bottom-frac", type=float, default=0.12)
    a = ap.parse_args(argv)
    rep = verify_cabin_coverage(
        a.original,
        a.blurred,
        a.vehicle_model,
        plate_model=a.plate_model,
        cabin_frac=a.cabin_frac,
        conf=a.conf,
        sample_fps=a.sample_fps,
        min_coverage=a.min_coverage,
        sharp_thresh=a.sharp_thresh,
        min_height_frac=a.min_height_frac,
        ignore_bottom_frac=a.ignore_bottom_frac,
    )
    print(f"checked {rep.vehicle_frames} vehicle-frame(s); {len(rep.leaks)} exposed cabin(s)")
    for lk in sorted(rep.leaks, key=lambda v: v.coverage)[:15]:
        print(
            f"  LEAK f{lk.frame}: coverage={lk.coverage:.0%} sharpness={lk.sharpness:.0f} box={lk.box}"
        )
    print("RESULT:", "PASS" if rep.ok else "FAIL")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
