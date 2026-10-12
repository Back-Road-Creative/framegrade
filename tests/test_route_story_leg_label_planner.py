"""Direct unit tests for the never-evict-an-incumbent label rule.

``LegLabelPlanner``/``_advance_active_labels`` are exercised here without any
capture, renderer, or leg plumbing -- just the windowing rule itself, and the
fact that the same rule applied one ``step()`` at a time produces the exact
same active-label set per frame as feeding the identical candidate lists
through ``_advance_active_labels`` directly.
"""

from __future__ import annotations

from gomoveshift.pipeline.route_story.layers import LegLabelPlanner, _advance_active_labels


def _label(name: str, visible: bool = True, kind: str = "poi") -> dict:
    return {"name": name, "visible": visible, "kind": kind}


def test_a_shown_label_stays_active_while_it_remains_a_candidate():
    planner = LegLabelPlanner(max_labels=1)

    # Frame 0: "a" is the only candidate, takes the one slot.
    planner.step([_label("a")])
    # Frame 1: "b" outranks nothing explicitly (list order = rank), but "a"
    # is still a candidate, so "a" keeps its slot -- "b" never displaces it.
    planner.step([_label("b"), _label("a")])

    assert set(planner._active) == {"a"}


def test_a_label_that_stops_being_a_candidate_frees_its_slot():
    planner = LegLabelPlanner(max_labels=1)

    planner.step([_label("a")])
    planner.step([_label("b")])  # "a" no longer a candidate -- evicted, not bumped.

    assert set(planner._active) == {"b"}


def test_step_reorders_incumbents_first_without_dropping_any_label():
    planner = LegLabelPlanner(max_labels=2)

    planner.step([_label("a")])
    result = planner.step([_label("b"), _label("a")])

    # Reordering, not filtering: both labels are still in the returned list,
    # but the incumbent ("a") now sorts ahead of the newcomer ("b").
    assert [entry["name"] for entry in result] == ["a", "b"]


def test_step_leaves_non_candidate_entries_untouched_and_passed_through():
    planner = LegLabelPlanner(max_labels=1)

    marker = {"name": "marker", "visible": True, "kind": "marker"}
    invisible = _label("hidden", visible=False)
    result = planner.step([_label("a"), marker, invisible])

    assert marker in result
    assert invisible in result


def test_precomputed_per_frame_active_sets_equal_sequential_step_calls():
    """The property the parallel wiring depends on: replaying the same
    per-frame candidate lists through ``_advance_active_labels`` directly
    (as if precomputed) reaches the identical active-label set, frame by
    frame, as driving the same lists one at a time through ``step()``.
    """
    frame_candidates = [
        ["a", "b"],
        ["b", "c"],
        ["c"],
        ["a", "c"],
        ["d", "c"],
    ]

    planner = LegLabelPlanner(max_labels=1)
    step_snapshots = []
    for candidates in frame_candidates:
        planner.step([_label(name) for name in candidates])
        step_snapshots.append(set(planner._active))

    active: dict[str, int] = {}
    windows: list[tuple[str, int, int]] = []
    direct_snapshots = []
    for index, candidates in enumerate(frame_candidates):
        _advance_active_labels(active, windows, index, candidates, max_labels=1)
        direct_snapshots.append(set(active))

    assert step_snapshots == direct_snapshots
