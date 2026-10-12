"""Parity guard: the 90-minute GPX match gap survives the sun-track extraction.

GPX parsing and interpolation now live in the published ``sun-track`` package,
whose own suite owns that behaviour. One thing did NOT survive the move
unchanged, and this file exists for it:

Upstream, the match gap was ``GPX_MATCH_MAX_GAP_MIN = 90`` — a module constant
inside the in-repo gpx_position module, with no way to override it. In the
package it is the per-call ``max_gap_min`` argument of
:meth:`TrackIndex.position_at`, defaulting to 90.0. Identical behaviour today,
but the value is now somebody else's default to change. This pipeline runs
unattended and grades photos against it, so ``exif_gps`` pins its own copy and
passes it explicitly; these tests fail if that ever silently reverts to
inheriting the package default.
"""

from datetime import datetime, timedelta, timezone

from sun_track import TrackIndex

from gomoveshift.pipeline.photos import exif_gps

UTC = timezone.utc
TRACK_POINT_UTC = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)


def test_match_gap_is_pinned_to_ninety_minutes() -> None:
    """The pipeline owns this number; it is not read from the package."""
    assert exif_gps.GPX_MATCH_MAX_GAP_MIN == 90.0


def test_gap_is_passed_explicitly_rather_than_inherited(monkeypatch) -> None:
    """The one call site names ``max_gap_min``.

    If the explicit argument were ever dropped, the recorder below sees
    ``None`` instead of 90.0 and this fails — which is the whole point: a
    future change to sun-track's default must not reach this pipeline.
    """
    recorded: dict[str, float | None] = {}

    class RecordingIndex:
        def position_at(self, utc, max_gap_min=None):
            recorded["max_gap_min"] = max_gap_min
            return (53.0, -8.0)

    monkeypatch.setattr(exif_gps, "_configured_track_index", lambda: RecordingIndex())
    exif_gps._solar_elevation_at_capture(TRACK_POINT_UTC)
    assert recorded["max_gap_min"] == 90.0


def test_ninety_minutes_is_the_accept_reject_boundary(monkeypatch) -> None:
    """What the pinned number MEANS, through the real package index."""
    monkeypatch.setattr(
        exif_gps,
        "_configured_track_index",
        lambda: TrackIndex(((TRACK_POINT_UTC, 53.0, -8.0),)),
    )
    # 89 minutes past the only track point: inside the gap, so the capture
    # takes that point's position and gets an elevation.
    assert exif_gps._solar_elevation_at_capture(TRACK_POINT_UTC + timedelta(minutes=89)) is not None
    # 91 minutes past it: beyond the gap → no position, so no elevation, and
    # the style selector falls back to its wall-clock golden window.
    assert exif_gps._solar_elevation_at_capture(TRACK_POINT_UTC + timedelta(minutes=91)) is None
