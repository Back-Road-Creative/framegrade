"""Compass bearing scales the E-W component by cos(latitude).

``_compass_from_waypoints`` fed ``atan2(dlon, dlat)`` raw, treating one degree
of longitude as one degree of latitude. At mid-high latitudes a longitude
degree spans far less ground than a latitude degree, so the E-W component was
over-weighted and a mostly-northbound leg could ship as "NE". Scaling ``dlon``
by ``cos(mean latitude)`` restores true ground bearing.
"""

from gomoveshift.pipeline._drive_helpers import _compass_from_waypoints


def test_mid_latitude_mostly_north_stays_north():
    # ~52N: 1 deg north, 0.5 deg east. Unscaled atan2 -> 26.6 deg ("NE");
    # cos(52.5 deg) shrinks the east leg so the ground bearing is 16.9 deg ("N").
    assert _compass_from_waypoints(52.0, 0.0, 53.0, 0.5) == "N"


def test_equator_unchanged_no_overcorrection():
    # cos(~0) ~= 1, so the same 1-north/0.5-east leg stays "NE" -- guards against
    # applying the scale where it should be a no-op.
    assert _compass_from_waypoints(0.0, 0.0, 1.0, 0.5) == "NE"


def test_due_north_and_due_east_still_correct():
    assert _compass_from_waypoints(52.0, 0.0, 53.0, 0.0) == "N"
    assert _compass_from_waypoints(52.0, 0.0, 52.0, 1.0) == "E"


def test_zero_vector_returns_north():
    assert _compass_from_waypoints(52.0, 0.0, 52.0, 0.0) == "N"
