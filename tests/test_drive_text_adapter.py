"""GMS drive facts -> Waypost ``DriveTextData`` -> reviewed description.

Synthetic places only ("Harrow", "Pellmere", "Westmark", "Ashby" are invented).
Every printed place must cite route-record or region evidence; anything else is
refused before upload.
"""

from __future__ import annotations

import pytest
from seo_manager.analytics.final_text_surfaces import (
    build_visible,
    compose_description,
    compose_tags,
)
from seo_manager.generation.drive_text import build_drive_text

from gomoveshift.pipeline.drive_text_adapter import (
    DriveFacts,
    DriveTextRefusedError,
    build_drive_data,
    cadence_pair,
    compose,
    drive_entries,
)

INDEX = {
    "harrow": ("ev-harrow", "Harrow"),
    "pellmere": ("ev-pellmere", "Pellmere"),
    "ostry": ("ev-ostry", "Ostry"),
    "westmark": ("ev-westmark", "Westmark"),
    "ashby county": ("ev-ashby", "Ashby County"),
    "a1": ("ev-a1", "A1"),
}
NAMES = ["Harrow", "Pellmere", "Ostry", "Westmark", "Ashby County", "A1"]


def _facts(**over):
    base = {"place": "Harrow", "final_seconds": 600.0}
    base.update(over)
    return DriveFacts(**base)


def _text_of(data):
    return compose_description(drive_entries(data))


def _text(facts):
    entries = drive_entries(build_drive_data(facts, INDEX))
    return compose_description(entries), compose_tags(entries)


def test_minimal_facts_compose_a_gms_drive_description():
    desc, tags = _text(_facts())
    assert "POV drive through Harrow (10m 0s)." in desc
    assert "Location: Harrow" in desc
    assert "POV driving" in tags


def test_entries_are_the_waypost_parts_for_the_data():
    data = build_drive_data(_facts(), INDEX)
    assert drive_entries(data) == build_visible(build_drive_text(data))
    assert compose(data) == (
        compose_description(drive_entries(data)),
        compose_tags(drive_entries(data)),
    )


def test_unbacked_place_is_refused_by_name():
    with pytest.raises(DriveTextRefusedError) as exc:
        build_drive_data(_facts(place="Nowhere"), INDEX)
    assert exc.value.reasons == ["name_unbacked:place:nowhere"]


def test_match_is_exact_name_not_substring():
    with pytest.raises(DriveTextRefusedError):
        build_drive_data(_facts(place="Harr"), INDEX)


def test_caller_item_i_playlist_without_site_is_refused_before_render():
    with pytest.raises(DriveTextRefusedError) as exc:
        build_drive_data(
            _facts(
                site=False,
                playlist=("Ireland", "https://www.youtube.com/playlist?list=PLsyntheticList01"),
            ),
            INDEX,
        )
    assert exc.value.reasons == ["site_line_required"]


def test_caller_item_i_cadence_without_site_is_refused_before_render():
    with pytest.raises(DriveTextRefusedError) as exc:
        build_drive_data(_facts(site=False, cadence=("USA", "Friday")), INDEX)
    assert exc.value.reasons == ["site_line_required"]


def test_playlist_and_cadence_render_with_the_site_line():
    desc, _ = _text(
        _facts(
            site=True,
            playlist=("Ireland", "https://www.youtube.com/playlist?list=PLsyntheticList01"),
            cadence=("Ireland", "Friday"),
        )
    )
    assert "Full Ireland playlist: https://www.youtube.com/playlist?list=PLsyntheticList01" in desc
    assert "New Ireland drive every Friday" in desc


def test_caller_item_ii_roads_features_passing_truncate_to_five():
    data = build_drive_data(
        _facts(
            journey_start="Harrow",
            journey_end="Pellmere",
            passing=NAMES,
            roads=NAMES,
            features=NAMES,
        ),
        INDEX,
    )
    assert len(data.passing) == len(data.roads) == len(data.features) == 5


def test_caller_item_ii_route_and_geo_truncate_like_gms():
    names = NAMES[:5]
    data = build_drive_data(_facts(route=names * 2, geo_hashtags=names), INDEX)
    assert len(data.route) == 8
    assert len(data.geo_hashtags) == 4


def test_caller_item_ii_more_than_twelve_towns_refuse_before_upload():
    with pytest.raises(DriveTextRefusedError) as exc:
        build_drive_data(_facts(towns=["Harrow"] * 13), INDEX)
    assert exc.value.reasons == ["too_many:towns"]


def test_caller_item_ii_more_than_twelve_states_refuse_before_upload():
    with pytest.raises(DriveTextRefusedError) as exc:
        build_drive_data(_facts(states=["Westmark"] * 13), INDEX)
    assert exc.value.reasons == ["too_many:states"]


@pytest.mark.parametrize(
    ("prose", "pair"),
    [
        ("New Ireland drive every Friday", ("Ireland", "Friday")),
        ("new usa drive every monday", ("USA", "Monday")),
    ],
)
def test_caller_item_iii_cadence_prose_maps_to_region_weekday(prose, pair):
    assert cadence_pair(prose) == pair


@pytest.mark.parametrize("prose", ["Weekly on Fridays", "New Ireland drive every Funday", ""])
def test_caller_item_iii_unmappable_cadence_is_refused(prose):
    with pytest.raises(DriveTextRefusedError) as exc:
        cadence_pair(prose)
    assert exc.value.reasons == ["cadence_unmappable"]


def test_caller_item_v_arc_with_same_start_and_end_falls_back_to_towns():
    data = build_drive_data(
        _facts(journey_start="Harrow", journey_end="Harrow", towns=["Harrow", "Pellmere"]),
        INDEX,
    )
    assert data.journey_start is None and data.journey_end is None
    assert [t.text for t in data.towns] == ["Harrow", "Pellmere"]
    assert "Driving through Harrow, Pellmere." in _text_of(data)


def test_caller_item_v_same_start_end_with_no_towns_drops_the_arc():
    data = build_drive_data(
        _facts(journey_start="Harrow", journey_end="Harrow", passing=["Ostry"]), INDEX
    )
    assert data.journey_start is None and not data.passing


def test_arc_with_distinct_ends_renders_and_cites_passing_places():
    data = build_drive_data(
        _facts(journey_start="Harrow", journey_end="Pellmere", passing=["Ostry"]), INDEX
    )
    assert "Started in Harrow, ended in Pellmere, passing through Ostry." in _text_of(data)


def test_caller_item_iv_no_seo_line_means_no_seo_blank_line():
    # GMS emits a blank line where the SEO line would be; Waypost's layout
    # emits no block at all. The difference is accepted: the PR states it.
    assert "4K drive" not in _text(_facts())[0]


def test_tags_cite_places_and_carry_the_recording_date():
    data = build_drive_data(
        _facts(place_tags=["Harrow"], place_drive_tags=["Pellmere"], year=2026, month=3), INDEX
    )
    tags = compose_tags(drive_entries(data))
    assert {"Harrow", "Pellmere drive", "2026", "March", "spring"} <= set(tags)


def test_playlist_url_must_be_a_youtube_playlist():
    with pytest.raises(DriveTextRefusedError) as exc:
        build_drive_data(_facts(playlist=("Ireland", "https://example.test/pl")), INDEX)
    assert exc.value.reasons == ["playlist_url_invalid"]


# --- chapter binding: real generated titles are decorated ------------------

_REAL_TITLES = [
    "Start - Beaufort County, South Carolina",
    "Entering Beaufort County",
    "Entering Sea Island Parkway, Beaufort County — Coastal",
    "Entering State Road S-7-406, Beaufort County",
    "Entering Russ Point, Beaufort County",
    "Finish - Beaufort County, South Carolina",
]
_ROUTE = {
    "record": {
        "places": [
            {"name": n, "evidence_id": f"ev-{i}"}
            for i, n in enumerate(["Longwood", "Fort Fremont", "Lands End"])
        ]
    },
    "region_claims": [
        {"level": "county", "name": "Beaufort County"},
        {"level": "state", "name": "South Carolina"},
    ],
}


def _chapters(*pairs):
    return [{"timestamp": t, "title": n} for t, n in pairs]


def _attach(chapters, label="Longwood"):
    from gomoveshift.pipeline.drive_text_adapter import attach_drive_text

    return attach_drive_text(_ROUTE, {}, label, final_seconds=600.0, chapters=chapters)


def test_real_decorated_titles_do_not_refuse():
    chapters = [{"timestamp": f"{i}:00", "title": t} for i, t in enumerate(_REAL_TITLES)]
    meta = _attach(chapters)
    assert "drive_text_refusal" not in meta
    assert meta["final_text_description"]


def test_decorated_titles_bind_to_the_evidenced_place():
    meta = _attach(
        _chapters(
            ("0:00", "Start - Longwood, Beaufort County"),
            ("2:00", "Entering Fort Fremont, Beaufort County"),
            ("4:00", "Finish - Lands End"),
        )
    )
    assert "drive_text_refusal" not in meta
    text = compose_description(meta["final_text_description"])
    assert "Fort Fremont" in text and "Entering" not in text


def test_chapter_window_cites_route_places_not_regions():
    # Waypost reads a chapter's segment_places as retained-route places; a
    # region id there refuses the LF (route_place_unknown).
    meta = _attach(
        _chapters(
            ("0:00", "Start - Longwood, Beaufort County, South Carolina"),
            ("2:00", "Entering Fort Fremont, Beaufort County"),
            ("4:00", "Finish - Lands End, South Carolina"),
        )
    )
    route_ids = {p["evidence_id"] for p in _ROUTE["record"]["places"]}
    windows = [
        e["segment_places"] for e in meta["final_text_description"] if e.get("kind") == "chapter"
    ]
    assert len(windows) == 3
    assert all(w and set(w) <= route_ids for w in windows), windows


def test_unevidenced_chapter_is_dropped_not_refused():
    from gomoveshift.pipeline.drive_text_adapter import evidence_index, lf_drive_facts

    facts = lf_drive_facts(
        "Longwood",
        evidence_index(_ROUTE),
        final_seconds=600.0,
        chapters=_chapters(
            ("0:00", "Longwood"),
            ("1:00", "Entering Nowhere Road"),
            ("2:00", "Fort Fremont"),
            ("3:00", "Lands End"),
        ),
    )
    assert [p for _, p, _ in facts.chapters] == ["Longwood", "Fort Fremont", "Lands End"]


def test_consecutive_same_place_chapters_collapse_to_the_earliest():
    from gomoveshift.pipeline.drive_text_adapter import evidence_index, lf_drive_facts

    facts = lf_drive_facts(
        "Longwood",
        evidence_index(_ROUTE),
        final_seconds=600.0,
        chapters=_chapters(
            ("0:00", "Longwood"),
            ("1:00", "Entering Longwood, Beaufort County"),
            ("2:00", "Fort Fremont"),
            ("3:00", "Longwood"),
        ),
    )
    assert [(s, p) for s, p, _ in facts.chapters] == [
        (0, "Longwood"),
        (120, "Fort Fremont"),
        (180, "Longwood"),
    ]


def test_too_few_distinct_chapters_send_none():
    from gomoveshift.pipeline.drive_text_adapter import evidence_index, lf_drive_facts

    facts = lf_drive_facts(
        "Longwood",
        evidence_index(_ROUTE),
        final_seconds=600.0,
        chapters=_chapters(("0:00", "Longwood"), ("2:00", "Fort Fremont")),
    )
    assert facts.chapters == []


def test_reattach_with_real_titles_does_not_refuse():
    from gomoveshift.pipeline.drive_text_adapter import reattach_with_chapters

    from gomoveshift.pipeline.drive_text_adapter import attach_drive_text

    meta = attach_drive_text(_ROUTE, {}, "Longwood", final_seconds=600.0)
    chapters = [{"timestamp": f"{i}:00", "title": t} for i, t in enumerate(_REAL_TITLES)]
    reattach_with_chapters(_ROUTE, meta, chapters)
    assert "drive_text_refusal" not in meta
    assert meta["final_text_description"]


# --- a town named inside a region's name is not that town -------------------

_NESTED = {
    "record": {
        "places": [
            {"name": n, "evidence_id": f"ev-{i}"}
            for i, n in enumerate(["Beaufort", "Port Royal", "Lobeco"])
        ]
    },
    "region_claims": [
        {"level": "county", "name": "Beaufort County"},
        {"level": "state", "name": "South Carolina"},
    ],
}


def test_town_inside_a_region_name_is_not_named():
    from gomoveshift.pipeline.drive_text_adapter import _named_in, evidence_index

    named = _named_in("Start - Beaufort County, South Carolina", evidence_index(_NESTED))
    assert named == ["Beaufort County", "South Carolina"]


def test_region_only_chapter_is_dropped_not_bound_to_a_nested_town():
    from gomoveshift.pipeline.drive_text_adapter import evidence_index, lf_drive_facts

    facts = lf_drive_facts(
        "Port Royal",
        evidence_index(_NESTED),
        final_seconds=600.0,
        chapters=_chapters(
            ("0:00", "Start - Port Royal, Beaufort County"),
            ("1:00", "Entering Beaufort County"),
            ("2:00", "Entering Beaufort"),
            ("3:00", "Entering Lobeco, Beaufort County"),
            ("4:00", "Finish - Beaufort County, South Carolina"),
        ),
    )
    assert [(s, p) for s, p, _ in facts.chapters] == [
        (0, "Port Royal"),
        (120, "Beaufort"),
        (180, "Lobeco"),
    ]


def test_chapter_windows_hold_route_places_only():
    """Waypost refuses a chapter window naming a region
    (``route_place_unknown:region-…``), which refused the whole LF."""
    meta = attach_drive_text_nested(
        _chapters(
            ("0:00", "Start - Port Royal, Beaufort County, South Carolina"),
            ("2:00", "Entering Beaufort, Beaufort County"),
            ("4:00", "Finish - Lobeco, South Carolina"),
        )
    )
    route_ids = {p["evidence_id"] for p in _NESTED["record"]["places"]}
    chapters = [e for e in meta["final_text_description"] if e["kind"] == "chapter"]
    assert len(chapters) == 3
    for entry in chapters:
        assert set(entry["segment_places"]) <= route_ids, entry


def attach_drive_text_nested(chapters):
    from gomoveshift.pipeline.drive_text_adapter import attach_drive_text

    return attach_drive_text(_NESTED, {}, "Port Royal", final_seconds=600.0, chapters=chapters)


def test_reattach_replaces_chapters_already_attached():
    from gomoveshift.pipeline.drive_text_adapter import attach_drive_text, reattach_with_chapters

    meta = attach_drive_text(_ROUTE, {}, "Longwood", final_seconds=600.0)
    chapters = [{"timestamp": f"{i}:00", "title": t} for i, t in enumerate(_REAL_TITLES)]
    reattach_with_chapters(_ROUTE, meta, chapters)
    reattach_with_chapters(_ROUTE, meta, chapters[:3])
    assert meta["drive_text_inputs"]["chapters"] == chapters[:3]
