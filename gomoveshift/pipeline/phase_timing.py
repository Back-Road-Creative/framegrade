"""Per-phase wall-time recorder for the drive pipeline (plan unit A5).

The orchestrator wraps each phase it runs in :meth:`PhaseTimingRecorder.phase`.
The record lands in ``<stages_dir>/phase_timings.json`` under one key,
``phase_timings``::

    {"phase_timings": {"phase_a": {"start": ISO, "end": ISO, "wall_s": 12.5,
                                    "units": None, "unit": None,
                                    "frames": 5000, "frames_per_s": 400.0}}}

``frames`` / ``frames_per_s`` are present only when a frame count is known;
``units`` is the work count the phase covered (days, for the day phase), so a
rate (seconds per unit) can be derived later. Only phases that finish are
recorded: a phase that raises leaves no entry. A resumed run merges into the
existing file, the newest entry for a phase winning.

Preflight reads these files back through :func:`recorded_rates`: the median
seconds-per-unit per phase over every record found, so the estimate is
deterministic for a given set of records.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ._atomic_write import atomic_write_json

logger = logging.getLogger(__name__)

RECORD_NAME = "phase_timings.json"
RECORD_KEY = "phase_timings"

#: Fewest recorded runs of a phase before preflight trusts the recorded rate
#: over the built-in constant. One real measurement beats a guess that every
#: drive so far ran about 5x over.
MIN_RECORDED_SAMPLES = 1


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def build_entry(
    start: float,
    end: float,
    *,
    units: Optional[float] = None,
    unit: Optional[str] = None,
    frames: Optional[int] = None,
) -> Dict[str, Any]:
    """One phase's record. ``frames_per_s`` only when ``frames`` is known."""
    wall = max(0.0, end - start)
    entry: Dict[str, Any] = {
        "start": _iso(start),
        "end": _iso(end),
        "wall_s": round(wall, 3),
        "units": units,
        "unit": unit,
    }
    if frames is not None:
        entry["frames"] = int(frames)
        entry["frames_per_s"] = round(frames / wall, 3) if wall > 0 else None
    return entry


class PhaseTimingRecorder:
    """Accumulates phase entries and persists them to ``path`` after each one."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: Dict[str, Dict[str, Any]] = {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            prior = raw.get(RECORD_KEY) if isinstance(raw, dict) else None
            if isinstance(prior, dict):
                self.entries = {k: v for k, v in prior.items() if isinstance(v, dict)}
        except (OSError, ValueError):
            pass

    def add(
        self,
        name: str,
        start: float,
        end: float,
        *,
        units: Optional[float] = None,
        unit: Optional[str] = None,
        frames: Optional[int] = None,
    ) -> None:
        self.entries[name] = build_entry(start, end, units=units, unit=unit, frames=frames)
        self.save()

    @contextmanager
    def phase(
        self, name: str, *, units: Optional[float] = None, unit: Optional[str] = None
    ) -> Iterator[Dict[str, Any]]:
        """Time a block. The yielded dict may get ``frames`` set before exit.

        Recording is best-effort: a write failure is logged, never raised into
        the pipeline. An exception in the block records nothing.
        """
        extra: Dict[str, Any] = {}
        start = time.time()
        yield extra
        try:
            self.add(name, start, time.time(), units=units, unit=unit, frames=extra.get("frames"))
        except OSError as exc:  # pragma: no cover - disk full / permissions
            logger.warning("phase timing not recorded for %s: %s", name, exc)

    def save(self) -> None:
        atomic_write_json(self.path, {RECORD_KEY: self.entries})


def discover_record_paths(stages_dir: Path, *, siblings: bool = False) -> List[Path]:
    """This drive's record; with ``siblings``, also other drives' under the
    same output root (``<root>/*/_stages``). Sorted, existing files only."""
    stages_dir = Path(stages_dir)
    found = {stages_dir / RECORD_NAME}
    if siblings:
        try:
            found.update(stages_dir.parent.parent.glob(f"*/_stages/{RECORD_NAME}"))
        except OSError:
            pass
    return sorted(p for p in found if p.is_file())


def recorded_rates(paths: Sequence[Path]) -> Dict[str, Dict[str, float]]:
    """Median seconds-per-unit per phase over ``paths``.

    Returns ``{phase: {"sec_per_unit": median, "samples": n}}`` for phases with
    at least :data:`MIN_RECORDED_SAMPLES` usable entries. A phase with no
    ``units`` counts one unit per run. Unreadable files and bad entries are
    skipped.
    """
    samples: Dict[str, List[float]] = {}
    for path in sorted(Path(p) for p in paths):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = raw.get(RECORD_KEY) if isinstance(raw, dict) else None
        if not isinstance(entries, dict):
            continue
        for name, e in entries.items():
            if not isinstance(e, dict):
                continue
            wall, units = e.get("wall_s"), e.get("units")
            if not isinstance(wall, (int, float)) or wall <= 0:
                continue
            if units is None:
                units = 1
            if not isinstance(units, (int, float)) or units <= 0:
                continue
            samples.setdefault(name, []).append(float(wall) / float(units))
    return {
        name: {"sec_per_unit": statistics.median(vals), "samples": len(vals)}
        for name, vals in samples.items()
        if len(vals) >= MIN_RECORDED_SAMPLES
    }
