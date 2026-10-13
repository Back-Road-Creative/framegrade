"""The place-shape helpers live engine-side, in ``pipeline.place_shape``.

``ingest/ost_formatter.py`` needs the county/township/region shape checks and the
mid-route vote; they used to sit in ``title_builder`` (a publish-side module the
media engine must not import). They moved to ``place_shape`` and ``title_builder``
re-exports them, so both import paths name the same objects.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline import place_shape, title_builder

_MOVED = (
    "_looks_like_county",
    "_looks_like_township",
    "_looks_like_region",
    "_REGION_TOKENS_RE",
    "_evenly_sample",
    "_endpoint_county_state",
    "_MID_VOTE_SAMPLES",
    "_central_window",
    "_vote_mid_region",
)


@pytest.mark.parametrize("name", _MOVED)
def test_title_builder_reexports_the_same_object(name):
    assert getattr(title_builder, name) is getattr(place_shape, name)


def test_place_shape_imports_nothing_from_title_builder():
    src = Path(place_shape.__file__).read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            assert "title_builder" not in (node.module or "")
            assert all("title_builder" not in a.name for a in node.names)
        elif isinstance(node, ast.Import):
            assert all("title_builder" not in a.name for a in node.names)


class TestShapeChecks:
    def test_county_and_township(self):
        assert place_shape._looks_like_county("Buncombe County")
        assert not place_shape._looks_like_county("Asheville")
        assert place_shape._looks_like_township("Union Township")
        assert not place_shape._looks_like_township("Asheville")


class TestSampling:
    def test_evenly_sample_spans_the_sequence(self):
        assert place_shape._evenly_sample([0, 1, 2, 3, 4], 3) == [0, 2, 4]
        assert place_shape._evenly_sample([1, 2], 5) == [1, 2]

    def test_central_window_is_centred(self):
        assert place_shape._central_window(list(range(11)), 5) == [3, 4, 5, 6, 7]
        assert place_shape._central_window([1, 2, 3], 5) == [1, 2, 3]


class TestVote:
    def _gc(self, regions):
        gc = MagicMock()
        gc.reverse_geocode_detailed.side_effect = [
            {"county": c, "state": s} if c else {} for c, s in regions
        ]
        return gc

    def test_plurality_wins(self):
        gc = self._gc([("A", "NC")] * 3 + [("B", "NC")] * 2)
        coords = [(35.0 + i / 100, -82.0) for i in range(5)]
        assert place_shape._vote_mid_region(gc, coords) == ("A", "NC")

    def test_exact_tie_is_inconclusive(self):
        gc = self._gc([("A", "NC"), ("B", "NC")])
        assert place_shape._vote_mid_region(gc, [(35.0, -82.0), (35.1, -82.0)]) == (None, None)

    def test_no_geocoder_or_coords(self):
        assert place_shape._vote_mid_region(None, [(1.0, 2.0)]) == (None, None)
        assert place_shape._vote_mid_region(MagicMock(), []) == (None, None)
