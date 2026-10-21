"""Synthetic Waypost-built drive-description entries for the final-text tests.

Built with Waypost's own part builders and registered ``gms-drive-1`` layout;
only invented place names reach them."""

from __future__ import annotations

from seo_manager.analytics.final_text_surfaces import (
    ParamBlock,
    TemplateBlock,
    build_visible,
)


def drive_entries(review_inputs: dict, *, chapters: bool = False) -> list:
    """Visible entries for a full ``gms-drive-1`` description of one drive.

    ``review_inputs``: ``{"places": {id: name}, "regions": {id: name}}``."""
    places, regions = review_inputs["places"], review_inputs["regions"]
    ids = list(places)[:2]
    region = next(iter(regions))
    named = {**places, **regions}
    parts = [
        ParamBlock(
            "description.hook",
            "description.hook.pov",
            {"place": region, "duration": "10m 0s"},
            {region: named[region]},
        ),
        ParamBlock(
            "description.location",
            "description.location.road" if len(ids) > 1 else "description.location.place",
            {"place": ids[0], "road": ids[1]} if len(ids) > 1 else {"place": ids[0]},
            {i: named[i] for i in ids},
        ),
        ParamBlock(
            "description.duration",
            "description.duration.plain",
            {"final": "10m 0s"},
            {},
        ),
        TemplateBlock("description.rule", "description.rule"),
        TemplateBlock("description.cta", "description.cta"),
        TemplateBlock("description.close", "description.close"),
        ParamBlock("description.hashtags", "description.hashtags.0", {}, {}),
    ]
    return build_visible(parts)


def attach_drive_description(state, meta: dict, lf_idx: int = 1) -> dict:
    """``meta`` with the drive-description entries for LF ``lf_idx``'s route."""
    from gomoveshift.pipeline.final_text_record import _region_id

    entry = (state.per_lf_route_record or {}).get(lf_idx) or {}
    places = {p["evidence_id"]: p["name"] for p in entry["record"]["places"]}
    regions = {
        _region_id(c["level"], c["name"]): c["name"] for c in entry.get("region_claims") or []
    }
    meta["final_text_description"] = drive_entries({"places": places, "regions": regions})
    return meta
