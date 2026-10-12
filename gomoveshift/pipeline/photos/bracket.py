"""Exposure-bracket detection: which files on a card are one HDR set.

Pure and deterministic — reads EXIF, returns groups of paths, mutates nothing.
The merge itself (OpenCV ``MergeMertens``, before the grade) and the user
flag that turns it on are a separate stage that consumes these groups.

The rule: same capture date, neighbouring frames inside ``MAX_GAP_S``, ISO held
constant, ``exposure_s`` stepping in ONE direction by at least
``MIN_STEP_RATIO``, the run inside ``MAX_SPAN_S``, at least
``MIN_BRACKET_FRAMES`` frames, and — once the run is closed — a total exposure
that actually MOVED by ``MIN_STEP_RATIO`` across it. A frame repeating its
neighbour's exposure is a duplicate: it neither extends the run nor breaks it,
and — like every other file the rule doesn't claim — it is left alone as a
normal single photo.

One direction is the load-bearing test: it separates a deliberate sweep from
auto-exposure drift across unrelated frames, which the loose window below would
otherwise swallow. The closing test is what a bracket is FOR: a depth-of-field
sweep steps the shutter in one direction like a bracket and cancels every step
with the aperture, so the frames differ in focus depth and not in exposure, and
merging them blends three depths of field to buy no range at all. One known
non-check remains, for the merge stage: a metered-frame-first AEB burst
(0, −, +) does not group — it wants a tight window.

These thresholds live here rather than in ``develop.py``, yet they still feed
``DEVELOP_ALGO_VERSION``: ``_develop_algo_inputs`` sweeps this module's ALL_CAPS
constants through ``_tunable_constants(vars(bracket))`` and hashes the rule's
source beside its own. Nothing here writes a pixel, but with the merge stage
wired these numbers decide WHICH frames become one — so a retune has to
invalidate a developed photo exactly as a grade change does.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path

from .exif_gps import (
    extract_exif_aperture,
    extract_exif_capture_metadata,
    extract_exif_capture_timestamp,
)

# Every threshold below is tuned against a hand-stepped bracket — a handful of
# frames over about a minute at fixed ISO, shutter dialled up by hand with a
# repeated exposure in the middle — not a textbook AEB burst.

# Seconds between neighbouring frames. A hand-stepped sequence has gaps of
# seconds to tens of seconds — a photographer turning the dial by hand, so the
# sub-2-second window an AEB detector uses finds nothing in it at all. 45 s
# clears the worst such gap with headroom, well below the minute-plus a walk
# to a new composition takes.
MAX_GAP_S = 45.0

# Total length of one run. Without it a golden-hour session whose light rises
# steadily chains dozens of unrelated frames into one "bracket", every pair
# inside MAX_GAP_S. Three minutes is longer than a hand-stepped
# sequence spans and short enough that the photographer never left the tripod.
MAX_SPAN_S = 180.0

# How far two exposures must differ to be a step rather than the same exposure
# shot twice. The tightest real step is a third of a stop (1.25x as shutter speeds
# 1/320 → 1/400, 1.26x as EXIF's rationals round it); a repeated shutter speed
# is exactly 1.0. 1.15
# sits under the tightest step a body shoots, far above equal-exposure float noise.
MIN_STEP_RATIO = 1.15

# A pair is not evidence. MAX_GAP_S must be loose enough for hand stepping, and
# at 45 s any two consecutive frames whose auto-exposure drifted a third of a
# stop would read as a bracket — the commonest accident on a card. Three
# same-direction steps is a sweep, and a two-frame merge buys the least range.
MIN_BRACKET_FRAMES = 3


@dataclass(frozen=True)
class _Frame:
    """One candidate frame: everything the grouping rule reads, read once."""

    path: Path
    captured_at: datetime.datetime
    exposure_s: float
    iso: float
    f_number: float | None


def _read_frame(photo_path):
    """A ``_Frame``, or ``None`` when this file can't take part in a bracket.

    Every incomplete case reads ``None`` — no capture time, no ISO, no or
    nonsensical exposure, unreadable file — because the rule compares all three
    and an unknown can't be held constant. Aperture is the exception: it is read
    here rather than through ``PhotoExif`` (see ``extract_exif_aperture``), and
    a missing one leaves the frame a candidate, so a card that writes no
    ``FNumber`` still brackets on the other three. Never raises, per the
    readers' contract.
    """
    captured_at = extract_exif_capture_timestamp(photo_path)
    meta = extract_exif_capture_metadata(photo_path)
    if captured_at is None or meta.iso is None or not meta.exposure_s or meta.exposure_s <= 0:
        return None
    return _Frame(
        Path(photo_path),
        captured_at,
        float(meta.exposure_s),
        float(meta.iso),
        extract_exif_aperture(photo_path),
    )


def _step(previous: _Frame, frame: _Frame):
    """How ``frame`` follows ``previous``: ``None`` = not the same bracket,
    ``0`` = duplicate exposure, ``-1``/``+1`` = a step down/up.

    Aperture is deliberately NOT compared here. A sequence may open up by
    aperture across its frames, so a held-aperture test would refuse a
    genuine bracket;
    what a depth-of-field sweep actually lacks is total-exposure range, and
    ``detect_brackets`` measures that once the run is closed.
    """
    if frame.captured_at.date() != previous.captured_at.date():
        return None  # a bracket never straddles the day a card registers under
    if (frame.captured_at - previous.captured_at).total_seconds() > MAX_GAP_S:
        return None
    if frame.iso != previous.iso:
        return None
    pair = (frame.exposure_s, previous.exposure_s)
    if max(pair) / min(pair) < MIN_STEP_RATIO:
        return 0
    return 1 if frame.exposure_s > previous.exposure_s else -1


def detect_brackets(photo_paths) -> list[tuple[Path, ...]]:
    """Group ``photo_paths`` into exposure brackets; return them in capture order.

    Each bracket is a tuple of paths in capture order, the brackets ordered by
    their first frame — stable and input-order independent, since frames sort by
    ``(capture time, path)`` first. A file in no bracket simply doesn't appear.

    A closed run must also have gathered measurably different LIGHT, which is
    shutter over aperture squared and not the shutter alone: a photographer
    opening up as the shutter shortens is sweeping depth of field, not
    bracketing, and every frame meters the same. The test needs ``FNumber`` on
    every frame of the run, so a card that writes none groups exactly as it did
    before the tag was read.
    """
    frames = sorted(
        (f for f in (_read_frame(p) for p in photo_paths or []) if f is not None),
        key=lambda f: (f.captured_at, str(f.path)),
    )
    brackets: list[tuple[Path, ...]] = []
    run: list[_Frame] = []
    previous, direction = None, 0

    def _close():
        if len(run) < MIN_BRACKET_FRAMES:
            return
        if all(f.f_number for f in run):
            light = [f.exposure_s / f.f_number**2 for f in run]
            if max(light) / min(light) < MIN_STEP_RATIO:
                return  # the aperture cancelled the sweep: no range to merge
        brackets.append(tuple(f.path for f in run))

    for frame in frames:
        step = _step(previous, frame) if previous is not None else None
        overlong = (
            bool(run) and (frame.captured_at - run[0].captured_at).total_seconds() > MAX_SPAN_S
        )
        if step is None or overlong or (step and direction and step != direction):
            _close()
            run, direction = [frame], 0
        elif step == 0:
            pass  # the same rung shot twice: not a step, not a break, ungrouped
        else:
            run.append(frame)
            direction = step
        previous = frame
    _close()
    return brackets
