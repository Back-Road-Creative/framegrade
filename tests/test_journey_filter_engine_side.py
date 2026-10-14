"""The journey filter is an engine module; ``chapters`` only re-exports it.

``_drive_helpers`` (engine) needs the GPX journey loader/filter. It used to import
it from ``pipeline.chapters`` (a Phase-D package), which the media-boundary guard
forbids. The GPS library is a sibling package outside this service, so the filter
lives in ``gomoveshift.pipeline.journey_filter`` and ``chapters._journey_filter``
keeps re-exporting the same objects for its publish-side callers.
"""

from __future__ import annotations

import gomoveshift.pipeline.chapters._journey_filter as legacy
import gomoveshift.pipeline.journey_filter as engine
from gps_journey.core.models import local_day

_NAMES = (
    "load_and_filter_journey",
    "load_journey_and_origin",
    "_read_stamped_journey",
    "_filter_by_date",
    "apply_gps_range",
    "local_day",
)


def test_engine_module_defines_the_filter():
    for name in _NAMES:
        assert hasattr(engine, name), name
    assert engine.local_day is local_day


def test_chapters_reexport_is_the_same_object():
    for name in _NAMES:
        assert getattr(legacy, name) is getattr(engine, name), name


def test_engine_module_defines_functions_itself():
    assert engine.load_and_filter_journey.__module__ == "gomoveshift.pipeline.journey_filter"
    assert engine.apply_gps_range.__module__ == "gomoveshift.pipeline.journey_filter"
