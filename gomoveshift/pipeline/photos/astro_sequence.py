"""Astro intervalometer-run detection: which files on a card are one night stack.

Pure and deterministic — reads EXIF, returns groups of paths, mutates nothing.
Same shape as ``bracket.py``, a different rule: a bracket wants exposure
STEPPING; an intervalometer astro run wants exposure and ISO HELD CONSTANT
across many frames fired at a steady cadence, which is what a sigma-clipped
stack of "the same sky, many looks" needs. ``detect_sequences`` finds the
runs; ``chunk_sequence`` then splits one run into stack-sized chunks, because
a run left running for an hour would blur in a stack from field rotation
alone and would not fit comfortably in RAM either — a separate function
because the two rules retune against a real card independently.

Every threshold below is PROVISIONAL, against no reference night card yet —
only the shape of ``bracket.py``'s own tuned constants. Retune with the same
measured-against-the-real-card discipline ``bracket.py`` documents for its
own numbers. These thresholds are part of ``DEVELOP_ALGO_VERSION``:
``_develop_algo_inputs`` in ``develop.py`` sweeps this module's ALL_CAPS constants
through ``_tunable_constants(vars(astro_sequence))``, exactly as it does for
``bracket.py``, so retuning one changes the algorithm version.
"""

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path

from .exif_gps import extract_exif_capture_metadata, extract_exif_capture_timestamp

logger = logging.getLogger(__name__)

# Seconds between neighbouring frames a steady intervalometer run may leave: a
# ~10-25s exposure plus a short intervalometer gap, with headroom, well under
# what a real break (repositioning, clouds) would take. Provisional.
MAX_FRAME_GAP_S = 60.0

# How far two exposures may differ and still read as HELD rather than
# stepped — the reciprocal of bracket.py's MIN_STEP_RATIO:
# that module treats >= 1.15 as a deliberate step, so here the same 1.15 is
# instead the CEILING for "the same exposure" (EXIF-rational rounding noise,
# not a step). Provisional.
MAX_HOLD_RATIO = 1.15

# Below this many frames a stack buys too little sqrt(N) noise reduction to be
# worth the alignment cost: sqrt(5) ~= 2.2x vs sqrt(20) ~= 4.5x. Also the floor
# a chunk must clear to stand alone (see chunk_sequence).
MIN_SEQUENCE_FRAMES = 5

# Default frames per stack chunk — a field-rotation cap for a fixed
# stack length, "~15-20 frames" (no per-declination math); 20 is its loose
# end. Provisional pending the reference card + a real lens FOV.
STACK_LEN_DEFAULT = 20

# Default wall-clock span a chunk may cover: 20 frames at ~15s exposure plus
# interval run roughly this long end to end. Caps the span, not just the
# count, so unusually long gaps can't drift past the rotation budget the
# frame cap protects. Provisional.
MAX_CHUNK_SPAN_S = 330.0


@dataclass(frozen=True)
class SequenceGates:
    """The frame-gap ceiling and minimum sequence length one ``detect_sequences``
    call used — public and typed so a later caller (e.g. a CLI provenance record)
    can capture the chosen values without re-deriving whether they were this
    module's steady-cadence defaults or an explicit sparse-night override."""

    max_frame_gap_s: float
    min_sequence_frames: int

    @property
    def is_sparse(self) -> bool:
        """Whether either value departs from the steady-cadence default."""
        return (
            self.max_frame_gap_s != MAX_FRAME_GAP_S
            or self.min_sequence_frames != MIN_SEQUENCE_FRAMES
        )


@dataclass(frozen=True)
class _Frame:
    """One candidate frame: everything the grouping rule reads, read once."""

    path: Path
    captured_at: datetime.datetime
    exposure_s: float
    iso: float


def _read_frame(photo_path):
    """A ``_Frame``, or ``None`` when this file can't take part in a run.

    Every incomplete case reads ``None`` — no capture time, no ISO, no or
    nonsensical exposure, unreadable file — because the rule holds all three
    constant and an unknown can't be held. Never raises, per the readers'
    contract.
    """
    captured_at = extract_exif_capture_timestamp(photo_path)
    meta = extract_exif_capture_metadata(photo_path)
    if captured_at is None or meta.iso is None or not meta.exposure_s or meta.exposure_s <= 0:
        return None
    return _Frame(Path(photo_path), captured_at, float(meta.exposure_s), float(meta.iso))


def _holds(previous: _Frame, frame: _Frame, max_frame_gap_s: float) -> bool:
    """Whether ``frame`` continues ``previous``'s run: inside ``max_frame_gap_s`` of
    ``previous``, ISO unchanged, exposure within ``MAX_HOLD_RATIO`` of equal.

    Deliberately NOT "same calendar day". The elapsed-gap test below already separates
    two shoots on either side of midnight, so a date comparison can only ever fire on a
    pair that is *inside* the window -- which happens exactly once a night, at 00:00, on
    the astro runs this module exists to group. A card shot across
    midnight at a steady cadence would otherwise come back as two runs, split
    between the last frame before 00:00 and the first after it.
    """
    if (frame.captured_at - previous.captured_at).total_seconds() > max_frame_gap_s:
        return False
    if frame.iso != previous.iso:
        return False
    pair = (frame.exposure_s, previous.exposure_s)
    return max(pair) / min(pair) < MAX_HOLD_RATIO


def detect_sequences(
    photo_paths,
    *,
    max_frame_gap_s: float = MAX_FRAME_GAP_S,
    min_sequence_frames: int = MIN_SEQUENCE_FRAMES,
) -> list[tuple[Path, ...]]:
    """Group ``photo_paths`` into intervalometer runs, in capture order.

    Each run is a tuple of paths in capture order, the runs ordered by their
    first frame — stable and input-order independent, since frames sort by
    ``(capture time, path)`` first, exactly as ``bracket.py`` does. A file in
    no run of at least ``min_sequence_frames`` simply doesn't appear; unreadable
    or EXIF-incomplete frames are excluded the same way, never raising.

    ``max_frame_gap_s``/``min_sequence_frames`` default to this module's
    steady-cadence constants. A caller may widen either for a sparse-cadence
    night (e.g. a long intervalometer interval); doing so logs one WARNING —
    at sparse cadence, ``astro_triage.py``'s persistence-based streak
    classification is blind and registration/projection error is higher than
    a steady-cadence run — never logged on the defaults. Wrap the values in
    ``SequenceGates`` to hand them to a later caller for provenance.
    """
    gates = SequenceGates(max_frame_gap_s, min_sequence_frames)
    if gates.is_sparse:
        logger.warning(
            "sparse-cadence sequence gates in use (max_frame_gap_s=%s, "
            "min_sequence_frames=%s): persistence-based streak classification "
            "is blind at this cadence and registration/projection error is "
            "higher than a steady-cadence run",
            max_frame_gap_s,
            min_sequence_frames,
        )
    frames = sorted(
        (f for f in (_read_frame(p) for p in photo_paths or []) if f is not None),
        key=lambda f: (f.captured_at, str(f.path)),
    )
    sequences: list[tuple[Path, ...]] = []
    run: list[_Frame] = []
    previous = None

    def _close():
        if len(run) >= min_sequence_frames:
            sequences.append(tuple(f.path for f in run))

    for frame in frames:
        if previous is not None and _holds(previous, frame, max_frame_gap_s):
            run.append(frame)
        else:
            _close()
            run = [frame]
        previous = frame
    _close()
    return sequences


def chunk_sequence(
    run: tuple[Path, ...],
    stack_len: int = STACK_LEN_DEFAULT,
    max_span_s: float = MAX_CHUNK_SPAN_S,
) -> list[tuple[Path, ...]]:
    """Split one ``detect_sequences`` run into contiguous stack-sized chunks.

    Takes paths, not ``(path, captured_at)`` pairs, and re-reads capture time
    per frame: composes directly with ``detect_sequences``'s own output with
    nothing extra for a caller to carry, and a run is small enough (tens of
    frames, not thousands) that the extra EXIF read costs nothing.

    A chunk closes once the next frame would push it past ``stack_len`` frames
    or ``max_span_s`` of wall clock from the chunk's own first frame. A
    resulting trailing chunk shorter than ``MIN_SEQUENCE_FRAMES`` merges into
    its predecessor when that stays within ``max_span_s`` end to end (the
    merged chunk may then exceed ``stack_len`` — a few extra frames cost less
    than discarding them); otherwise — no predecessor, or a merge that would
    blow the span — it is dropped. A caller must not assume the returned
    chunks' frames sum to ``len(run)``: the drop is the contract.
    """
    if not run:
        return []
    frames = list(run)
    timestamps = [extract_exif_capture_timestamp(p) for p in frames]

    chunks: list[list[Path]] = [[frames[0]]]
    starts: list[datetime.datetime] = [timestamps[0]]
    for path, ts in zip(frames[1:], timestamps[1:]):
        start = starts[-1]
        span = (ts - start).total_seconds() if start is not None and ts is not None else 0.0
        if len(chunks[-1]) >= stack_len or span > max_span_s:
            chunks.append([path])
            starts.append(ts)
        else:
            chunks[-1].append(path)

    if len(chunks[-1]) < MIN_SEQUENCE_FRAMES:
        trailing = chunks.pop()
        starts.pop()  # keep starts aligned with chunks now that trailing is gone
        if chunks:
            pred_start, end_ts = starts[-1], timestamps[-1]
            merged_span = (
                (end_ts - pred_start).total_seconds()
                if pred_start is not None and end_ts is not None
                else 0.0
            )
            if merged_span <= max_span_s:
                chunks[-1].extend(trailing)
            # else: the merge would blow the span cap — trailing is dropped
        # else: no predecessor to absorb an undersized solo chunk — dropped

    return [tuple(c) for c in chunks]
