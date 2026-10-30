"""Review-set scan: candidate review windows per lighting and motion condition.

Library only. Reads a per-second score array (``unified_scorer`` columns) and,
when present, the stop-mask motion states, GPS fixes and the master's
``.tonemap.json`` record, and proposes windows a human should look at for each
condition. Nothing here edits, cuts or renders anything.

Every condition is always present in the result with a ``status``:

* ``ok`` - the inputs it needs were given (``windows`` may be empty);
* ``no_input`` - an optional input was absent, so the condition was not scanned;
* ``unsupported`` - no detector exists (backlight, foliage, wet windshield);
  a human tags these, they are never guessed.

Each window records the rule that fired and the threshold values it used, so a
reviewer can see why it was proposed. Output is deterministic and
JSON-serialisable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from .stop_mask import COL_MOTION_ENERGY, MotionState, build_gps_gap_mask

SCHEMA_VERSION = 1

# Score columns (mirror ``analysis.unified_scorer``; kept local so the scan
# does not import the scorer's OpenCV stack).
COL_TIMESTAMP = 0
COL_COLOR_VIBRANCY = 4
COL_SKY_RATIO = 6
COL_GOLDEN_HOUR = 7
COL_BRIGHTNESS = 10
NUM_COLUMNS = 11

CONDITIONS: tuple[str, ...] = (
    "daylight",
    "overcast",
    "dusk_night",
    "golden_hour",
    "tunnel",
    "creeping",
    "parked_traffic",
    "hdr_gradients",
    "gps_gap",
    "backlight",
    "foliage",
    "wet_windshield",
)
UNSUPPORTED: tuple[str, ...] = ("backlight", "foliage", "wet_windshield")


@dataclass(frozen=True)
class ScanThresholds:
    """Rule thresholds. Defaults are review heuristics, not quality gates."""

    min_window_s: int = 3
    daylight_min_brightness: float = 0.45
    daylight_min_vibrancy: float = 0.35
    overcast_min_brightness: float = 0.30
    overcast_max_vibrancy: float = 0.35
    dusk_night_max_brightness: float = 0.25
    golden_hour_min: float = 0.6
    tunnel_max_sky_ratio: float = 0.2
    tunnel_min_brightness_drop: float = 0.15
    tunnel_min_window_s: int = 2
    parked_min_motion_energy: float = 0.3
    gps_max_gap_s: Optional[float] = None  # None = the stop mask's own default


def _runs(flags: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Inclusive ``(first, last)`` index runs of True at least ``min_len`` long."""
    out: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, f in enumerate(flags.tolist() + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= max(min_len, 1):
                out.append((start, i - 1))
            start = None
    return out


def _windows(
    scores: np.ndarray,
    flags: np.ndarray,
    rule: str,
    thresholds: dict[str, Any],
    min_len: int,
    inputs: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    ts = scores[:, COL_TIMESTAMP]
    out = []
    for a, b in _runs(flags, min_len):
        out.append(
            {
                "start_s": float(ts[a]),
                "end_s": float(ts[b]) + 1.0,
                "rule": rule,
                "thresholds": dict(thresholds),
                "inputs": {k: round(float(np.mean(v[a : b + 1])), 4) for k, v in inputs.items()},
            }
        )
    return out


def _entry(status: str, windows: list, *, heuristic: bool = False, note: str = "") -> dict:
    e: dict[str, Any] = {"status": status, "heuristic": heuristic, "windows": windows}
    if note:
        e["note"] = note
    return e


def scan_review_set(
    scores: np.ndarray,
    *,
    states: Optional[np.ndarray] = None,
    gps_points=None,
    tonemap: Optional[dict] = None,
    thresholds: Optional[ScanThresholds] = None,
) -> dict[str, Any]:
    """Return candidate review windows for every condition in ``CONDITIONS``.

    ``scores`` is the per-second score array; ``states`` the stop-mask int8
    ``MotionState`` series; ``gps_points`` the day's ``(t, lat, lon)`` fixes;
    ``tonemap`` the parsed ``.tonemap.json``. Each optional input left as
    ``None`` makes its conditions report ``no_input``.
    """
    th = thresholds or ScanThresholds()
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < NUM_COLUMNS:
        raise ValueError(f"scores must be (n, >={NUM_COLUMNS}); got shape {scores.shape}")
    n = scores.shape[0]
    bright = scores[:, COL_BRIGHTNESS]
    vib = scores[:, COL_COLOR_VIBRANCY]
    sky = scores[:, COL_SKY_RATIO]
    golden = scores[:, COL_GOLDEN_HOUR]
    motion = scores[:, COL_MOTION_ENERGY]
    res: dict[str, dict] = {}

    res["daylight"] = _entry(
        "ok",
        _windows(
            scores,
            (bright >= th.daylight_min_brightness) & (vib >= th.daylight_min_vibrancy),
            "brightness_and_vibrancy_high",
            {
                "min_brightness": th.daylight_min_brightness,
                "min_color_vibrancy": th.daylight_min_vibrancy,
            },
            th.min_window_s,
            {"brightness": bright, "color_vibrancy": vib},
        ),
    )
    res["overcast"] = _entry(
        "ok",
        _windows(
            scores,
            (bright >= th.overcast_min_brightness) & (vib < th.overcast_max_vibrancy),
            "bright_but_low_vibrancy",
            {
                "min_brightness": th.overcast_min_brightness,
                "max_color_vibrancy": th.overcast_max_vibrancy,
            },
            th.min_window_s,
            {"brightness": bright, "color_vibrancy": vib},
        ),
    )
    res["dusk_night"] = _entry(
        "ok",
        _windows(
            scores,
            bright < th.dusk_night_max_brightness,
            "brightness_low",
            {"max_brightness": th.dusk_night_max_brightness},
            th.min_window_s,
            {"brightness": bright},
        ),
    )
    res["golden_hour"] = _entry(
        "ok",
        _windows(
            scores,
            golden >= th.golden_hour_min,
            "golden_hour_score_high",
            {"min_golden_hour": th.golden_hour_min},
            th.min_window_s,
            {"golden_hour": golden},
        ),
    )
    baseline = float(np.median(bright)) if n else 0.0
    res["tunnel"] = _entry(
        "ok",
        _windows(
            scores,
            (sky <= th.tunnel_max_sky_ratio) & (bright <= baseline - th.tunnel_min_brightness_drop),
            "dark_sky_and_brightness_drop",
            {
                "max_sky_ratio": th.tunnel_max_sky_ratio,
                "min_brightness_drop": th.tunnel_min_brightness_drop,
                "baseline_brightness_median": round(baseline, 4),
            },
            th.tunnel_min_window_s,
            {"sky_ratio": sky, "brightness": bright},
        ),
        heuristic=True,
    )

    if states is None:
        for name in ("creeping", "parked_traffic"):
            res[name] = _entry("no_input", [], note="stop-mask states not provided")
    else:
        st = np.asarray(states)
        if st.shape[0] != n:
            raise ValueError(f"states length {st.shape[0]} != scores length {n}")
        res["creeping"] = _entry(
            "ok",
            _windows(
                scores,
                st == int(MotionState.CREEPING),
                "motion_state_creeping",
                {"state": "CREEPING"},
                th.min_window_s,
                {"motion_energy": motion},
            ),
        )
        res["parked_traffic"] = _entry(
            "ok",
            _windows(
                scores,
                (st == int(MotionState.STOPPED)) & (motion >= th.parked_min_motion_energy),
                "stopped_with_high_motion_energy",
                {"state": "STOPPED", "min_motion_energy": th.parked_min_motion_energy},
                th.min_window_s,
                {"motion_energy": motion},
            ),
            heuristic=True,
        )

    if tonemap is None:
        res["hdr_gradients"] = _entry("no_input", [], note="tonemap sidecar not provided")
    else:
        w = []
        if n:
            ts = scores[:, COL_TIMESTAMP]
            w = [
                {
                    "start_s": float(ts[0]),
                    "end_s": float(ts[-1]) + 1.0,
                    "rule": "tonemap_sidecar_present",
                    "thresholds": {},
                    "inputs": {
                        k: tonemap.get(k)
                        for k in ("tonemap", "source_color_trc", "source_color_primaries")
                    },
                }
            ]
        res["hdr_gradients"] = _entry("ok", w, note="whole master was tonemapped")

    from .gps_quality import usable_fix_times

    if gps_points is None or not usable_fix_times(gps_points):
        res["gps_gap"] = _entry("no_input", [], note="no usable GPS fixes")
    else:
        kw = {} if th.gps_max_gap_s is None else {"max_gap_s": th.gps_max_gap_s}
        gap = build_gps_gap_mask(n, gps_points, **kw)
        res["gps_gap"] = _entry(
            "ok",
            _windows(
                scores,
                gap,
                "gps_fix_gap_exceeds_max",
                {"max_gap_s": th.gps_max_gap_s},
                1,
                {},
            ),
        )

    for name in UNSUPPORTED:
        res[name] = _entry("unsupported", [], note="no detector; a human tags this")

    return {
        "schema_version": SCHEMA_VERSION,
        "duration_s": n,
        "thresholds": asdict(th),
        "conditions": {c: res[c] for c in CONDITIONS},
    }
