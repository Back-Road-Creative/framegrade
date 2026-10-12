"""Tests for the route inventory: stating the queue's
routing table as data, and the per-platform breakdown of a blocked drain
pass, rather than leaving both implicit in ``build_handlers``'s dict keys
and a bare integer counter.
"""

import inspect

from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.handlers import build_handlers
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.route_inventory import (
    NOT_ROUTED,
    NOT_ROUTED_CAVEAT,
    ROUTE_STATES,
    ROUTED,
    KNOWN_PLATFORMS,
    route_states,
)

NOW = "2026-08-23T00:00:00+00:00"


#: ``build_handlers`` keyword parameters that name no platform.
_NON_PLATFORM_PARAMS = {
    "conn",
    "record_loader",
    "worker_id",
    "tiktok_rendition_path_loader",
}


def _platforms_build_handlers_accepts() -> set:
    """The platforms ``build_handlers`` can wire, read off its signature
    rather than restated. Each platform is one keyword parameter; a
    ``<platform>_photo`` parameter wires the photo/carousel format of a
    platform already named, so it collapses onto the same platform."""
    return {
        name[: -len("_photo")] if name.endswith("_photo") else name
        for name in inspect.signature(build_handlers).parameters
        if name not in _NON_PLATFORM_PARAMS
    }


def test_known_platforms_matches_what_build_handlers_can_register():
    # Derived from build_handlers's own signature, NOT restated as a literal:
    # a platform added there without being added to KNOWN_PLATFORMS makes this
    # fail, which is the whole point -- an inventory that silently undercounts
    # is the defect this guards against, not a smaller version of it.
    assert set(KNOWN_PLATFORMS) == _platforms_build_handlers_accepts()


def test_route_states_reports_routed_for_a_registered_handler():
    handlers = {"youtube": lambda row: None}
    states = route_states(handlers)
    assert states["youtube"] == ROUTED


def test_route_states_reports_not_routed_for_a_platform_missing_from_handlers():
    handlers = {"youtube": lambda row: None}
    states = route_states(handlers)
    assert states["instagram"] == NOT_ROUTED
    assert states["facebook"] == NOT_ROUTED
    assert states["tiktok"] == NOT_ROUTED


def test_route_states_covers_every_known_platform_even_when_handlers_is_empty():
    states = route_states({})
    assert set(states) == set(KNOWN_PLATFORMS)
    assert all(state == NOT_ROUTED for state in states.values())


def test_route_states_ignores_a_handler_mapping_key_outside_the_known_set():
    # An arbitrary enqueued platform string (e.g. "snapchat" in
    # test_publishing_dispatch.py) is not part of the route table this
    # module states -- it is neither routed nor not_routed, it simply isn't
    # inventoried.
    handlers = {"snapchat": lambda row: None}
    states = route_states(handlers)
    assert "snapchat" not in states


def test_only_two_route_states_exist():
    """Pins the state count. Someone adding a third state (e.g. a
    "disabled" state) must touch this constant AND this test -- forcing
    them to confront NOT_ROUTED_CAVEAT below rather than silently growing
    the enum."""
    assert ROUTE_STATES == (ROUTED, NOT_ROUTED)
    assert len(ROUTE_STATES) == 2


def test_not_routed_caveat_names_the_missing_intent_source():
    """NOT_ROUTED cannot mean "deliberately disabled" -- nothing in this repo
    records that intent at the routing layer (checked: no per-platform
    enabled/disabled field in config.py's models; queue.py's
    platform_pauses/is_platform_paused governs drain SELECTION, not routing,
    and a paused platform can still have a handler registered). This test
    pins that the caveat says so, rather than letting the ambiguity go
    undocumented."""
    caveat = NOT_ROUTED_CAVEAT.lower()
    assert "not yet wired" in caveat or "not wired" in caveat
    assert "disabled" in caveat or "switched off" in caveat
    assert "no" in caveat and "record" in caveat


def _conn(tmp_path):
    return open_queue_db(tmp_path / "seo")


def test_drain_result_names_which_platforms_were_skipped(tmp_path):
    conn = _conn(tmp_path)
    try:
        enqueue(conn, "proj", 0, "instagram", "hash-a")
        enqueue(conn, "proj", 1, "facebook", "hash-b")
        enqueue(conn, "proj", 2, "facebook", "hash-c")
        result = drain_queue(conn, {}, now=NOW)
        assert result.skipped_no_handler_by_platform == {"instagram": 1, "facebook": 2}
        # The whole must equal the sum of its named parts -- the point of the inventory.
        assert result.skipped_no_handler == sum(result.skipped_no_handler_by_platform.values())
    finally:
        conn.close()


def test_skipped_no_handler_by_platform_stays_consistent_with_registered_handlers(tmp_path):
    conn = _conn(tmp_path)
    try:
        enqueue(conn, "proj", 0, "instagram", "hash-a")
        enqueue(conn, "proj", 0, "youtube", "hash-b")
        result = drain_queue(conn, {"youtube": lambda row: None}, now=NOW)
        assert result.skipped_no_handler == 1
        assert result.skipped_no_handler_by_platform == {"instagram": 1}
        assert result.skipped_no_handler == sum(result.skipped_no_handler_by_platform.values())
    finally:
        conn.close()
