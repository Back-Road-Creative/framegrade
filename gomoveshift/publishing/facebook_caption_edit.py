"""Rewrite a published Facebook Reel's description from what the Short's record can back.

The caption is BUILT, never typed, whenever the Short's own sidecar record holds one measured
county (``sampled_regions``): it names that county, the recording month and nothing else, so
it cannot carry an unbacked place and cannot fall back to a placeless label either. Only a
Short with no measured county takes operator text, and that text must claim nothing
(``content_gates.placeless_label_claims_nothing``). ``apply`` edits the live post, verifies
the read-back, then records the text in the sidecar. Never changes the queue row, never
deletes anything.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

from ..pipeline.validation.content_gates import (
    _county_names,
    placeless_label_claimed_words,
    placeless_label_claims_nothing,
    placeless_label_problem,
)
from .plan_io import load_publish_plan
from .queue import open_queue_db

# One of social_generator.FACEBOOK_CAPTION_TEMPLATES (pinned by a test).
MEASURED_TEMPLATE = "Passing through {location}."


class CaptionEditRefusedError(Exception):
    """The edit cannot be made; nothing was written."""


def _adapter(credentials_dir):
    from .facebook_adapter import FacebookReelsAdapter
    from .meta_auth import load_meta_credentials

    return FacebookReelsAdapter(load_meta_credentials(Path(credentials_dir)))


def measured_counties(short: dict) -> list:
    """The Short's own county-form region labels (``"<Name> County, <State>"``)."""
    return [
        r.strip()
        for r in short.get("sampled_regions") or []
        if isinstance(r, str) and _county_names(r)
    ]


def measured_caption(region: str, recording_date: str) -> str:
    """The caption for a Short whose record measured ``region``: the place, the month, tags."""
    text = MEASURED_TEMPLATE.format(location=region)
    try:
        text += f" Filmed {dt.date.fromisoformat(recording_date or ''):%B %Y}."
    except ValueError:
        pass  # no parseable date on the record: say nothing about when
    tags = [re.sub(r"[^A-Za-z0-9]", "", part) for part in region.split(",")]
    return text + "\n\n" + " ".join(f"#{t}" for t in tags if t)


def _placeless_problem(text: str):
    if placeless_label_claims_nothing(text):
        return None
    words = ", ".join(repr(w) for w in placeless_label_claimed_words(text))
    return "; ".join(x for x in (placeless_label_problem(text), f"claim words: {words}") if x)


def resolve_caption(short: dict, text) -> tuple:
    """``(caption, basis)`` for ``short``, or raise :class:`CaptionEditRefusedError`."""
    counties = measured_counties(short)
    if len(counties) > 1:
        raise CaptionEditRefusedError(
            f"the Short's record measured {len(counties)} counties ({'; '.join(counties)}); "
            "one caption cannot name them"
        )
    if counties:
        if text:
            raise CaptionEditRefusedError(
                f"the Short's record measured {counties[0]!r}; the caption is built from it, "
                "so drop --text (a typed caption is only for a Short with no measured county)"
            )
        return measured_caption(counties[0], short.get("recording_date") or ""), "measured county"
    if not text:
        raise CaptionEditRefusedError(
            "the Short's record holds no measured county; pass --text with claim-free words"
        )
    problem = _placeless_problem(text)
    if problem:
        raise CaptionEditRefusedError(f"new text is not claim-free ({problem})")
    return text, "claim-free text (no measured county on the record)"


def _write_sidecar(sidecar: Path, raw: dict) -> None:
    tmp = sidecar.with_name(sidecar.name + ".fbcaption-tmp")
    tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        load_publish_plan(tmp)  # must still parse before it replaces the old
    except Exception:
        tmp.unlink()
        raise
    os.replace(tmp, sidecar)


def edit_facebook_caption(storage_path, credentials_dir, delivery_id, text, *, apply) -> list:
    """Dry-run or apply the edit; returns the lines to print. Raises ``CaptionEditRefusedError``."""
    conn = open_queue_db(storage_path)
    try:
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise CaptionEditRefusedError(f"no delivery {delivery_id}")
    if row["platform"] != "facebook":
        raise CaptionEditRefusedError(
            f"delivery {delivery_id} is platform {row['platform']!r}, not facebook"
        )
    if row["status"] != "published":
        raise CaptionEditRefusedError(f"delivery {delivery_id} is {row['status']!r}, not published")
    upload_id = row["platform_upload_id"]
    if not upload_id:
        raise CaptionEditRefusedError(f"delivery {delivery_id} has no platform upload id")
    sidecar = Path(row["sidecar_path"] or "")
    if not sidecar.is_file():
        raise CaptionEditRefusedError(f"publish-plan sidecar not readable: {sidecar}")
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    short = next(
        (
            s
            for s in raw.get("shorts") or []
            if s.get("project_name") == row["project_name"]
            and s.get("short_index") == row["short_index"]
        ),
        None,
    )
    fb = ((short or {}).get("payloads") or {}).get("facebook")
    if not fb:
        raise CaptionEditRefusedError(f"{sidecar}: no facebook payload for the short")
    try:
        new, basis = resolve_caption(short, text)
    except CaptionEditRefusedError as exc:
        raise CaptionEditRefusedError(f"delivery {delivery_id}: {exc}") from None
    lines = [
        f"delivery {delivery_id} upload {upload_id}",
        f"current caption: {fb.get('description')!r}",
        f"new text: {new!r}",
        f"basis: {basis}",
    ]
    if not apply:
        return [*lines, "dry run: nothing written; re-run with --apply to edit the live post"]
    live = _adapter(credentials_dir).update_description(upload_id, new)
    if live.get("description") != new:
        raise CaptionEditRefusedError(
            f"read-back of {upload_id} is {live.get('description')!r}, not the new text; "
            "sidecar left unchanged"
        )
    fb["description"] = new
    _write_sidecar(sidecar, raw)
    return [*lines, f"wrote: delivery {delivery_id} live caption and sidecar now read {new!r}"]
