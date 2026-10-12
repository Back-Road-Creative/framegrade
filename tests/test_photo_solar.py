"""Parity guard: the sun-track package still gives this pipeline's numbers.

The solar maths now lives in the published ``sun-track`` package, so its own
suite owns the algorithm (naive-datetime rejection, timezone normalisation,
refraction, the band predicate). What CANNOT live upstream is the evidence
this pipeline was actually tuned against, so it stays here:

- The reference elevations are pinned values, cross-checked against an
  independent low-precision solar-position formula (agreement within about
  0.4 degrees), at round-number coordinates the package has never heard of.
  The rows use 35.0N, -83.0W (mid-latitude, northern hemisphere) and
  53.0N, -8.0W (high latitude). A package upgrade that moved these numbers
  would silently re-grade every photo the selector has ever scored.
- The golden band edges are what ``photos/styles.py`` gates the "golden" style
  on. They are pinned here rather than merely re-exported, so widening the
  band upstream fails this suite instead of quietly changing style selection.

Tolerance is +/-0.4 degrees: looser than float noise, far tighter than the
10-degree golden band the values gate.
"""

from datetime import datetime, timezone

import pytest
from sun_track import (
    GOLDEN_ELEVATION_MAX,
    GOLDEN_ELEVATION_MIN,
    is_golden_elevation,
    solar_elevation_deg,
)

TOLERANCE_DEG = 0.4

# (lat, lon, UTC datetime, expected apparent elevation in degrees)
REFERENCE_CASES = [
    pytest.param(
        35.0,
        -83.0,
        datetime(2026, 7, 1, 23, 5, tzinfo=timezone.utc),
        19.0,
        id="midlat-jul01-evening",
    ),
    pytest.param(
        35.0,
        -83.0,
        datetime(2026, 7, 2, 11, 40, tzinfo=timezone.utc),
        13.9,
        id="midlat-jul02-morning",
    ),
    pytest.param(
        53.0,
        -8.0,
        datetime(2026, 6, 21, 16, 0, tzinfo=timezone.utc),
        41.3,
        id="highlat-solstice-17-local",
    ),
    pytest.param(
        53.0,
        -8.0,
        datetime(2026, 6, 21, 20, 0, tzinfo=timezone.utc),
        6.8,
        id="highlat-solstice-21-local",
    ),
    pytest.param(
        53.0,
        -8.0,
        datetime(2026, 1, 1, 17, 0, tzinfo=timezone.utc),
        -5.2,
        id="highlat-newyear-dusk",
    ),
    pytest.param(
        35.0,
        -83.0,
        datetime(2026, 7, 2, 10, 40, tzinfo=timezone.utc),
        2.8,
        id="midlat-jul02-early",
    ),
]


@pytest.mark.parametrize("lat, lon, utc, expected", REFERENCE_CASES)
def test_reference_elevations(lat: float, lon: float, utc: datetime, expected: float) -> None:
    assert solar_elevation_deg(lat, lon, utc) == pytest.approx(expected, abs=TOLERANCE_DEG)


def test_golden_band_constants() -> None:
    """The band styles.py gates on, pinned against an upstream widening."""
    assert GOLDEN_ELEVATION_MIN == -4.0
    assert GOLDEN_ELEVATION_MAX == 6.0
    # Both edges inclusive — the selector's favored-scene test depends on it.
    assert is_golden_elevation(-4.0) is True
    assert is_golden_elevation(6.0) is True
    assert is_golden_elevation(-4.001) is False
    assert is_golden_elevation(6.001) is False
