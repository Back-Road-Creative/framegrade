"""Standalone-asset enqueue bundle -- the door through which a route-story
film, a photo export, or any Waypost product NOT produced by the drive
pipeline's Phase D enters the delivery queue. See
``docs/social-publishing-queue.md`` ("Standalone assets").

Phase D's door (``enqueue-plan``) gates on Gate C/D's confirmation hash, an
interactive review recorded on the drive run's ``pipeline_state.json``. A
standalone asset has no such run, so :func:`review_hash` is this door's own
equivalent, covering the four reviewed fields (title, description, tags,
file sha256) so a post-review change is detectable and refused.

Privacy is fail-closed: this module never lets an operator self-attest
``verified``. A route-story receipt's ``privacy`` block is accepted as
evidence; anything else marks every required class ``inconclusive``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .dispatch import PRIVACY_INCONCLUSIVE, PRIVACY_REQUIRED_CLASSES
from .media_fingerprint import fingerprint_media
from .models import KNOWN_PLATFORMS

#: Matches ``pipeline.day_master_encoder.PRIVACY_ATTESTATION_SCHEMA`` (not
#: imported, to avoid pulling ffmpeg into this door for one string).
PRIVACY_ATTESTATION_SCHEMA = "gms.privacy_attestation/v1"

#: Matches ``plan_builder.build_publish_plan``'s default (most restrictive).
DEFAULT_TIKTOK_PRIVACY_LEVEL = "SELF_ONLY"

#: Matches ``dispatch.PRIVACY_ATTESTATION_GLOB`` so the existing drain serves it.
ASSET_ATTESTATION_FILENAME = "day_asset_privacy_attestation.json"


def review_hash(title: str, description: str, tags: Sequence[str], sha256: str) -> str:
    """sha256 over a canonical JSON of the four fields a human reviews.
    This IS the Gate C/D equivalent: what ``--confirm`` matches is exactly
    what would ship, byte for byte."""
    canonical = json.dumps(
        {"title": title, "description": description, "tags": list(tags), "sha256": sha256},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _inconclusive_attestation() -> dict:
    """Fail-closed default: every required class ``inconclusive``, never
    ``verified`` -- this door offers no self-attestation."""
    return {
        "schema": PRIVACY_ATTESTATION_SCHEMA,
        "required_classes": list(PRIVACY_REQUIRED_CLASSES),
        "classes": {
            name: {"state": PRIVACY_INCONCLUSIVE, "reason": "no route-story receipt supplied"}
            for name in PRIVACY_REQUIRED_CLASSES
        },
    }


def _normalize_attestation(privacy_attestation: Optional[Mapping]) -> dict:
    """Wrap a route-story receipt's ``privacy.classes`` (already in
    dispatch's vocabulary) in the same envelope the day-master encoder
    writes, so ``PrivacyAttestationLoader`` reads it identically."""
    if not privacy_attestation or not privacy_attestation.get("classes"):
        return _inconclusive_attestation()
    return {
        "schema": PRIVACY_ATTESTATION_SCHEMA,
        "required_classes": list(PRIVACY_REQUIRED_CLASSES),
        "classes": dict(privacy_attestation["classes"]),
    }


def _youtube_payload(title: str, description: str, tags: Sequence[str]) -> dict:
    return {"title": title, "description": description, "tags": list(tags), "category": ""}


def _instagram_payload(title: str, description: str) -> dict:
    caption = f"{title}\n\n{description}" if description else title
    return {"caption": caption}


def _facebook_payload(description: str) -> dict:
    return {"description": description}


def _tiktok_payload(title: str, privacy_level: str) -> dict:
    return {
        "title": title,
        "caption": "",
        "privacy_level": privacy_level,
        "disclose_branded_content": False,
        "disclose_your_brand": False,
    }


def build_asset_sidecar(
    video: Path,
    *,
    title: str,
    description: str,
    tags: Sequence[str],
    recording_date: str,
    platforms: Sequence[str],
    scheduled_at_utc: str,
    display_timezone: str = "UTC",
    privacy_attestation: Optional[Mapping] = None,
    project_name: Optional[str] = None,
    format: Optional[str] = None,
    tiktok_privacy_level: str = DEFAULT_TIKTOK_PRIVACY_LEVEL,
) -> dict:
    """Build a ``{"shorts": [record]}`` document ``load_publish_plan``
    accepts for a single standalone asset. Raises ``ValueError`` on an
    unknown platform and whatever :func:`fingerprint_media` raises for a
    missing/unreadable/no-video file, both before anything is written.

    ``format`` defaults to ``None`` ("not stated"), not a new ``"asset"``
    string: ``models.KNOWN_FORMATS`` is a closed set this door doesn't own,
    and ``plan_io`` treats ``None`` as a distinct valid state. Pass an
    explicit ``models.FORMAT_PHOTO``/``FORMAT_LONG_FORM`` when known."""
    unknown = sorted(set(platforms) - KNOWN_PLATFORMS)
    if unknown:
        raise ValueError(f"build_asset_sidecar: unknown platform(s): {unknown}")
    if not platforms:
        raise ValueError("build_asset_sidecar: at least one platform is required")

    media = fingerprint_media(Path(video))
    if project_name is None:
        project_name = f"asset:{media.sha256[:8]}"

    payloads: dict = {"youtube": None, "instagram": None, "facebook": None, "tiktok": None}
    if "youtube" in platforms:
        payloads["youtube"] = _youtube_payload(title, description, tags)
    if "instagram" in platforms:
        payloads["instagram"] = _instagram_payload(title, description)
    if "facebook" in platforms:
        payloads["facebook"] = _facebook_payload(description)
    if "tiktok" in platforms:
        payloads["tiktok"] = _tiktok_payload(title, tiktok_privacy_level)

    rhash = review_hash(title, description, tags, media.sha256)

    record = {
        "project_name": project_name,
        "lf_index": 0,
        "short_index": 0,
        "media": {
            "path": media.path,
            "size_bytes": media.size_bytes,
            "duration_seconds": media.duration_seconds,
            "width": media.width,
            "height": media.height,
            "codec": media.codec,
            "sha256": media.sha256,
        },
        "schedule": {
            "scheduled_at_utc": scheduled_at_utc,
            "display_timezone": display_timezone,
        },
        "recording_date": recording_date,
        "sampled_regions": [],
        "parent_lf_video_id": "",
        "parent_lf_publish_at": scheduled_at_utc,
        "payloads": payloads,
        "consent_hash": rhash,
        "reviewed_plan_hash": rhash,
        "routes": {"selected": list(platforms)},
        "format": format,
    }
    return {"shorts": [record]}


def write_asset_bundle(
    directory: Path, sidecar_doc: Mapping, privacy_attestation: Optional[Mapping]
) -> Tuple[Path, Path]:
    """Write ``<project>_publish_plan.json`` and the attestation file into
    ``directory``, returning ``(sidecar_path, attestation_path)``. Never
    writes a fake ``verified`` attestation -- see the module docstring."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    project_name = sidecar_doc["shorts"][0]["project_name"]
    sidecar_path = directory / f"{project_name}_publish_plan.json"
    sidecar_path.write_text(json.dumps(sidecar_doc, sort_keys=True, indent=2), encoding="utf-8")

    attestation_path = directory / ASSET_ATTESTATION_FILENAME
    attestation_doc = _normalize_attestation(privacy_attestation)
    attestation_path.write_text(json.dumps(attestation_doc, indent=2), encoding="utf-8")
    return sidecar_path, attestation_path
