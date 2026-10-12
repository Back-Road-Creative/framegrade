"""The coastal band must survive a dead primary Overpass mirror.

Observed on a coastal drive that shipped with the coastal band silently disabled:

    Coastline query skipped (HTTP Error 500: Internal Server Error);
    coastal band disabled

``_query_overpass`` iterates ``OVERPASS_ENDPOINTS`` with retries, so the POI
pass failed over and succeeded. ``_coastline_points_for_bbox`` posted straight to
``OVERPASS_ENDPOINTS[0]`` with no failover at all, so it died on the same
mirror the POI pass had already routed around. Confirmed out-of-band: the primary timed out at
25s while the secondary answered HTTP 200 in 1.8s.

The failure is silent by design (``coastal band disabled`` is logged at INFO
and an empty list is a legal result), so nothing downstream notices that a
coastal drive lost its coastal chapters.
"""

import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.chapters import overpass_source as ops
from gomoveshift.pipeline.chapters.generator import _coastline_points_for_bbox

pytestmark = pytest.mark.real_overpass


@pytest.fixture(autouse=True)
def _reset_sticky_endpoint():
    """The sticky last-good endpoint (overpass_source) is process-lifetime
    state — never leak it between tests."""
    ops.reset_sticky_overpass_endpoint()
    yield
    ops.reset_sticky_overpass_endpoint()


WAYPOINTS = [
    SimpleNamespace(latitude=51.45, longitude=-9.65),
    SimpleNamespace(latitude=51.55, longitude=-9.45),
]

_COASTLINE = {
    "elements": [
        {
            "type": "way",
            "id": 7,
            "geometry": [{"lat": 51.50, "lon": -9.60}, {"lat": 51.51, "lon": -9.59}],
        }
    ]
}


class _FakeResponse:
    status = 200

    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_coastline_fails_over_to_secondary_endpoint():
    """A 500 from the primary mirror must not disable the coastal band."""
    seen: list[str] = []

    def fake_urlopen(req, timeout=None):
        url = req.full_url
        seen.append(url)
        if url == ops.OVERPASS_ENDPOINTS[0]:
            raise urllib.error.HTTPError(url, 500, "Internal Server Error", None, None)
        return _FakeResponse(_COASTLINE)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        points = _coastline_points_for_bbox(WAYPOINTS)

    assert seen[0] == ops.OVERPASS_ENDPOINTS[0], "primary must still be tried first"
    assert ops.OVERPASS_ENDPOINTS[1] in seen, "secondary endpoint was never tried"
    assert points == [(51.50, -9.60), (51.51, -9.59)]


def test_coastline_returns_empty_when_every_endpoint_fails():
    """All mirrors down is still a soft failure — empty band, no exception."""

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 504, "Gateway Timeout", None, None)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert _coastline_points_for_bbox(WAYPOINTS) == []
