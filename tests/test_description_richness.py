"""Description-richness tests for ``_enrich_with_route_profile``.

Locks in the contract that the long-form YouTube description body surfaces
GPS-derived data that ``RouteProfile`` already computes — counties, states,
POIs (features), journey-arc (start_town → end_town), and up to 5 roads.

Strictly consumes existing ``RouteProfile`` fields; no new GPS work.
"""

from __future__ import annotations

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


def _base_result(location: str = "Somewhere, SC") -> dict:
    """A minimal pre-enrichment result dict (mimics _build_gms_description output)."""
    return {
        "title": f"Driving {location}",
        "description": (
            f"POV drive through {location}.\n"
            f"\n"
            f"📍 Location: {location}\n"
            f"🎬 Final cut: 10m 0s (edited from 30m 0s of footage)\n"
            f"📅 Filmed: 2025-09-14\n"
        ),
        "tags": ["scenic drive"],
        "location": location,
    }


# ── 1. Single-state, single-town: behaviour close to today ──────────────


def test_single_town_single_state_fallback_form():
    """One town, no journey arc — uses legacy 'Driving through X on Y.' form."""
    result = _base_result("Beaufort, SC")
    profile = {
        "towns_traversed": ["Beaufort"],
        "roads_traversed": ["US-21"],
        "features": [],
        "states_traversed": ["South Carolina"],
        "start_town": None,
        "end_town": None,
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    assert "Driving through Beaufort on US-21." in desc
    # Single state — no spans line
    assert "Spans" not in desc
    # No features — no highlights line
    assert "Highlights along the route" not in desc


# ── 2. Multi-state: spans-N-states line appears ─────────────────────────


def test_multi_state_drive_emits_spans_line():
    """When states_traversed has >1 entry, a 'Spans N states: ...' line appears."""
    result = _base_result("Lake Tahoe, CA")
    profile = {
        "towns_traversed": ["South Lake Tahoe", "Stateline"],
        "roads_traversed": ["US-50"],
        "features": [],
        "states_traversed": ["California", "Nevada"],
        "start_town": "South Lake Tahoe",
        "end_town": "Stateline",
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    assert "Spans 2 states: California, Nevada." in desc


# ── 3. Journey-arc: start_town + end_town distinct ──────────────────────


def test_journey_arc_sentence_appears():
    """Start/end town differ → 'Started in X, ended in Y, passing through ...'."""
    result = _base_result("Highway 1, CA")
    profile = {
        "towns_traversed": ["Monterey", "Big Sur", "Cambria", "Morro Bay"],
        "roads_traversed": ["CA-1"],
        "features": [],
        "states_traversed": ["California"],
        "start_town": "Monterey",
        "end_town": "Morro Bay",
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    assert "Started in Monterey, ended in Morro Bay" in desc
    # Intermediate towns (not the endpoints) appear in "passing through ..."
    assert "Big Sur" in desc
    assert "Cambria" in desc
    # Endpoints not duplicated in intermediates
    assert "passing through Monterey" not in desc
    # Roads listed on a separate line
    assert "Roads traveled: CA-1." in desc


def test_journey_arc_names_a_revisited_town_once():
    """A loop that re-enters a town must not read 'passing through X, Y, Y'.

    A Gettysburg loop: towns_traversed carried the battlefield loop's
    real sequence (… Cumberland Township, Gettysburg, Cumberland Township,
    Gettysburg, …) and the arc sentence shipped 'passing through Freedom
    Township, Gettysburg, Gettysburg'. The 🛤️ Route line keeps the true
    sequence; the prose names each place once, first-visit order.
    """
    result = _base_result("Gettysburg Battlefield, PA")
    profile = {
        "towns_traversed": [
            "Liberty Township",
            "Freedom Township",
            "Cumberland Township",
            "Gettysburg",
            "Cumberland Township",
            "Gettysburg",
            "Cumberland Township",
        ],
        "roads_traversed": [],
        "features": [],
        "states_traversed": ["Pennsylvania"],
        "start_town": "Liberty Township",
        "end_town": "Cumberland Township",
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    assert "passing through Freedom Township, Gettysburg." in desc
    assert "Gettysburg, Gettysburg" not in desc


def test_journey_arc_skipped_when_endpoints_match():
    """start_town == end_town → falls back to legacy form (no arc)."""
    result = _base_result("Beaufort, SC")
    profile = {
        "towns_traversed": ["Beaufort"],
        "roads_traversed": ["US-21"],
        "features": [],
        "states_traversed": ["South Carolina"],
        "start_town": "Beaufort",
        "end_town": "Beaufort",
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    assert "Started in" not in desc
    assert "Driving through Beaufort" in desc


# ── 4. POIs (features) appear in body AND tags ──────────────────────────


def test_features_appear_in_description_body_and_tags():
    """Features populate a 'Highlights along the route' line AND tags."""
    result = _base_result("Beaufort, SC")
    profile = {
        "towns_traversed": ["Beaufort"],
        "roads_traversed": [],
        "features": ["Hunting Island State Park", "Penn Center"],
        "states_traversed": ["South Carolina"],
        "start_town": None,
        "end_town": None,
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    assert "Highlights along the route: Hunting Island State Park, Penn Center." in desc
    tags = enriched["tags"]
    assert "Hunting Island State Park" in tags
    assert "Penn Center" in tags


def test_features_capped_at_five_in_body():
    """Only the first 5 features appear in the body line."""
    result = _base_result("Somewhere, SC")
    profile = {
        "towns_traversed": ["A"],
        "roads_traversed": [],
        "features": [f"POI {i}" for i in range(8)],
        "states_traversed": ["South Carolina"],
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    # First 5 in, 6th out of the body
    assert "POI 4" in desc
    # POI 5 is the 6th feature (index 5) — must NOT be in the highlights line
    highlights_line = next(
        (line for line in desc.splitlines() if line.startswith("Highlights along")),
        "",
    )
    assert "POI 5" not in highlights_line


# ── 5. Roads count > 3: up to 5 listed in body ──────────────────────────


def test_roads_listed_up_to_five_in_body():
    """Roads cap raised from 3 → 5."""
    result = _base_result("Somewhere, SC")
    profile = {
        "towns_traversed": ["Town A"],
        "roads_traversed": ["R1", "R2", "R3", "R4", "R5", "R6"],
        "features": [],
        "states_traversed": ["South Carolina"],
        "start_town": None,
        "end_town": None,
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    desc = enriched["description"]
    assert "R5" in desc
    assert "R6" not in desc


# ── 6. Empty / None route_profile fields: graceful no-op ────────────────


def test_empty_route_profile_no_body_change():
    """All fields empty → description unchanged; no spurious body line."""
    result = _base_result("Somewhere, SC")
    original_desc = result["description"]
    profile = {
        "towns_traversed": [],
        "roads_traversed": [],
        "features": [],
        "states_traversed": [],
        "start_town": None,
        "end_town": None,
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    assert enriched["description"] == original_desc


def test_none_fields_handled_gracefully():
    """``None`` for list fields must not raise — coerced to empty."""
    result = _base_result("Somewhere, SC")
    profile = {
        "towns_traversed": None,
        "roads_traversed": None,
        "features": None,
        "states_traversed": None,
        "start_town": None,
        "end_town": None,
    }
    # Should not raise
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    assert "description" in enriched
    # No body added
    assert "Driving through" not in enriched["description"]
    assert "Spans" not in enriched["description"]
    assert "Highlights along" not in enriched["description"]


# ── 7. Redundancy-gate compatibility: chapter_towns threaded back ───────


def test_chapter_towns_threaded_back_for_gate():
    """Enrichment must populate ``chapter_towns`` so the description-redundancy
    gate at Phase D recognises the body's new nouns."""
    result = _base_result("Highway 1, CA")
    profile = {
        "towns_traversed": ["Monterey", "Big Sur"],
        "roads_traversed": ["CA-1"],
        "features": ["Bixby Bridge"],
        "states_traversed": ["California"],
        "start_town": "Monterey",
        "end_town": "Big Sur",
    }
    enriched = LongFormMetadataGenerator._enrich_with_route_profile(result, profile)
    nouns = enriched.get("chapter_towns") or []
    # Endpoints, intermediates, features all available as nouns
    assert "Monterey" in nouns
    assert "Big Sur" in nouns
    assert "Bixby Bridge" in nouns
    assert enriched.get("named_routes") == ["CA-1"]
