"""Place evidence a Short's own GPS measured, as the final-text review reads it.

Pure functions over a ``ShortPublishRecord``: no platform client, no network. The YouTube
adapter's review and the Shorts title generator (``pipeline/social_generator.py``) both read
place evidence from here, so a title is only built from places the review will accept, and
the generator never imports a platform client (``tests/test_platform_client_boundary.py``).
"""

from __future__ import annotations

from ..pipeline.location_evidence import _evidence_id
from .models import ShortPublishRecord

_PLACE_SOURCE = "clip_gps_vote"


def clip_places(record: ShortPublishRecord, segment_id: str) -> list[dict]:
    """The clip's own GPS region votes (``"County, State"``) as place evidence."""
    places = []
    for label in record.sampled_regions:
        county, _, state = (part.strip() for part in label.partition(","))
        if county:
            places.append(
                {
                    "evidence_id": _evidence_id(segment_id, county, county, state, _PLACE_SOURCE),
                    "kind": "location",
                    "value": county,
                    "source": _PLACE_SOURCE,
                }
            )
    return places


def clip_towns(record: ShortPublishRecord) -> list[dict]:
    """The towns the clip's own route record measured for every retained
    second, as place evidence under the route record's own evidence id.

    A town held for only part of the clip does not name the clip's place."""
    route = record.route_record or {}
    retained = float(route.get("retained_seconds") or 0.0)
    if retained <= 0:
        return []
    return [
        {
            "evidence_id": p["evidence_id"],
            "kind": "location",
            "value": p["name"],
            "source": route.get("containment_basis") or "retained_gps",
        }
        for p in route.get("places") or []
        if p.get("name")
        and p.get("evidence_id")
        and p.get("required")
        and float(p.get("retained_seconds") or 0.0) >= retained
    ]
