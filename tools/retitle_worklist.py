#!/usr/bin/env python3
"""Back-catalog retitle worklist CSV generator.

Reads ``videos.yaml`` + an optional saved RSS/Atom file (no network fetch)
and emits one row per ``(video, reason)``: ``duplicate-title``,
``highlights-from``, ``county-only-title`` (bare "X County[, ST]", no
settlement/route -- standalone regex fallback since this script has no
resolved-location object for the pipeline's own county gate), and
``preposition-led-label`` (``To X``, ``through To``, ``#To<Cap>``).
``suggested_title`` comes ONLY from structured fields on the video record
(never parsed out of the existing title, to avoid inventing a place).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

# RSS/Atom feed is external input; defusedxml blocks entity-expansion attacks
# (same rationale as gomoveshift/analysis/gps_parser.py's GPX parsing).
import defusedxml.ElementTree as ET  # noqa: N817  drop-in for stdlib ET, keep name
import yaml

# Reuse retitle_apply's own gates at compose time -- a composed title
# that would be REFUSED by retitle_apply must never be handed out as a
# suggestion in the first place (single source of truth for "does this pass").
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# The shorts' own place-hashtag builder, so a rewritten description carries the
# same per-component tags a fresh upload would (longform_metadata reuses it too).
from gomoveshift.pipeline.social_generator import (  # noqa: E402
    TERRAIN_HASHTAGS,
    _location_hashtags,
)
from tools.retitle_apply import evaluate_row as _evaluate_row  # noqa: E402

FIELDNAMES = ["video_id", "current_title", "reason", "suggested_title", "platform"]

_HIGHLIGHTS_RE = re.compile(r"^Highlights from ", re.IGNORECASE)

# Fallback county-only regex (module docstring): strip prefix/prep/suffix,
# then nothing must remain but "<Name> County[, ST]" -- no settlement/route.
_COUNTY_ONLY_CORE_RE = re.compile(r"^[A-Za-z][A-Za-z .'\-]* County(?:,\s*[A-Za-z]{2,})?$")
_COUNTY_ONLY_STRIP_RES = [
    re.compile(r"^.*\|\s*"),
    re.compile(r"^(?:To|Through|Up around|The road to|Around)\s+", re.IGNORECASE),
    re.compile(r"\s*in 4K\s*$", re.IGNORECASE),
    re.compile(r"\s*Scenic Drive\s*$", re.IGNORECASE),
]

_PREPOSITION_LABEL_PATTERNS = [
    re.compile(r"through To ", re.IGNORECASE),
    re.compile(r"Drive Through To", re.IGNORECASE),
    re.compile(r"#To[A-Z]"),
    re.compile(r"^To [A-Z]"),
]


def is_county_only_title(title: str) -> bool:
    """True when the title's location is a bare county, no settlement named."""
    core = title
    for pattern in _COUNTY_ONLY_STRIP_RES:
        core = pattern.sub("", core).strip()
    if core.lower().startswith("county "):
        return False  # "County Cork" etc. is the region's own name, not the defect.
    return bool(_COUNTY_ONLY_CORE_RE.match(core))


def is_highlights_from(title: str) -> bool:
    return bool(_HIGHLIGHTS_RE.search(title))


def has_preposition_led_label(text: str) -> bool:
    return any(p.search(text) for p in _PREPOSITION_LABEL_PATTERNS)


def build_suggested_title(video: dict[str, Any]) -> str:
    """Place-dense template — only from structured fields, never the title."""
    origin, dest, region, place = (
        str(video.get(k) or "").strip() for k in ("origin", "destination", "region", "place")
    )
    if origin and dest and region:
        return f"{origin} to {dest}, {region}"
    if place and region:
        return f"{place}, {region}"
    return "needs-human"


# Composition -- when no structured field is present, the ONLY safe
# source of place data is the place text already sitting inside the current
# (defective) title. These strip a known junk shape and return exactly the
# substring that survives; they never add a place name that wasn't already
# there.
_PREPOSITION_STRIP_RES = [
    re.compile(r"^.*\bThrough To\s+", re.IGNORECASE),
    re.compile(r"^#To(?=[A-Z])"),
    re.compile(r"^To\s+", re.IGNORECASE),
]


def _extract_place_from_title(title: str) -> Optional[str]:
    """Recover the real place text embedded in a template/prefix-led title.

    Handles the three shapes ``retitle_worklist`` classifies: the
    ``"{Region} Scenic Drive in 4K | {place}"`` template (duplicate-title's
    usual shape), a ``"Highlights from {place}"`` label, and a preposition-
    led label (``"To {place}"`` / ``"#To{Place}"`` / ``"... Through To
    {place}"``). Returns ``None`` when nothing recognisable survives the
    strip -- the caller must fall back to needs-human rather than guess.
    """
    candidate = None
    if "|" in title:
        candidate = title.rpartition("|")[2].strip()
    elif is_highlights_from(title):
        candidate = _HIGHLIGHTS_RE.sub("", title, count=1).strip()
    elif has_preposition_led_label(title):
        for pat in _PREPOSITION_STRIP_RES:
            if pat.search(title):
                candidate = pat.sub("", title, count=1).strip()
                break
    if not candidate:
        return None
    candidate = re.sub(r"\s*in 4K\s*$", "", candidate, flags=re.IGNORECASE).strip()
    if not candidate or "scenic drive" in candidate.lower() or candidate == title.strip():
        return None
    return candidate


def compose_suggested_title(
    video: dict[str, Any], *, dup_index: Optional[int] = None, dup_total: Optional[int] = None
) -> str:
    """Best available title for ``video``, gate-checked before it's handed out.

    Prefers the structured-field cascade (:func:`build_suggested_title`);
    falls back to the place text recovered from the current title
    (:func:`_extract_place_from_title`). A ``duplicate-title`` collision is
    disambiguated with the same ``"(Part N of M)"`` shape
    ``title_builder.ensure_unique_titles`` uses for sibling LFs. The result
    is run through ``retitle_apply``'s own gates -- a candidate that would be
    REFUSED is never surfaced; the row stays ``needs-human`` instead.
    """
    candidate = build_suggested_title(video)
    if candidate == "needs-human":
        extracted = _extract_place_from_title(str(video.get("title") or ""))
        if extracted:
            candidate = extracted
    if candidate == "needs-human":
        return candidate
    if dup_total and dup_total > 1:
        candidate = f"{candidate} (Part {dup_index} of {dup_total})"
    if _evaluate_row({"suggested_title": candidate}):
        return "needs-human"
    return candidate


# Hashtags that name no place: the shorts' fixed tags and their terrain tags.
_PLACELESS_TAGS = {t.lower() for t in ("#POVDriving", "#ScenicDrive", "#RoadTrip", "#Shorts")} | {
    t.lower() for tags in TERRAIN_HASHTAGS.values() for t in tags
}
_HASHTAG_RE = re.compile(r"#\w+")


def rewrite_description_place(description: str, old_place: str, new_place: str) -> Optional[str]:
    """``description`` naming ``new_place`` wherever it named ``old_place``.

    The text swap is whole-word (never inside a hashtag or a longer name).
    Every place hashtag is dropped and ``new_place``'s own
    :func:`_location_hashtags` go in after the line's last placeless tag, so
    no tag for the old place, its state, or a smashed route survives.
    ``None`` when the text never names ``old_place``, or when a hashtag the
    token pass can't isolate (``#VA.``) would survive: such a description
    stays as it is rather than half-fixed.
    """
    word = re.compile(rf"(?<![#\w]){re.escape(old_place)}(?!\w)")
    if not word.search(description):
        return None
    lines = word.sub(lambda _m: new_place, description).split("\n")
    new_tags = _location_hashtags(new_place)
    target = None  # (line index, token index) where the new tags go
    for i, line in enumerate(lines):
        tokens, kept, cut, last_tag = line.split(" "), [], False, None
        for token in tokens:
            if _HASHTAG_RE.fullmatch(token) and token.lower() not in _PLACELESS_TAGS:
                cut = True
                continue
            kept.append(token)
            if _HASHTAG_RE.fullmatch(token):
                last_tag = len(kept)
        if cut or last_tag is not None:
            lines[i] = " ".join(kept)
            if target is None or (cut and not target[2]):
                target = (i, last_tag if last_tag is not None else len(kept), cut)
    if target is None:
        lines += ["", " ".join(new_tags)]
    else:
        i, at, _ = target
        tokens = lines[i].split(" ") if lines[i] else []
        have = {t.lower() for t in tokens}
        tokens[at:at] = [t for t in new_tags if t.lower() not in have]
        lines[i] = " ".join(tokens)
    result = "\n".join(lines)
    allowed = _PLACELESS_TAGS | {t.lower() for t in new_tags}
    if any(t.lower() not in allowed for t in _HASHTAG_RE.findall(result)):
        return None
    return result


def load_videos(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text()) or {}
    videos = data.get("videos") or []
    return [v for v in videos if isinstance(v, dict) and v.get("id") and v.get("title")]


_YT_ID_RE = re.compile(r"[?&]v=([\w-]{6,})")


def load_rss_entries(path: Path) -> list[dict[str, Any]]:
    """Parse a saved channel RSS/Atom XML file. No network access."""
    root = ET.parse(path).getroot()
    entries: list[dict[str, Any]] = []
    for node in root.iter():  # RSS 2.0 <item> and Atom <entry>, namespace-agnostic.
        if node.tag.rsplit("}", 1)[-1] not in ("item", "entry"):
            continue
        found = {"title": "", "videoId": "", "link": ""}
        for child in node.iter():
            ctag = child.tag.rsplit("}", 1)[-1]
            key = "link" if ctag == "guid" else ctag
            if key in found and not found[key]:
                found[key] = child.get("href") or (child.text or "").strip()
        video_id = found["videoId"]
        if not video_id and found["link"]:
            m = _YT_ID_RE.search(found["link"])
            video_id = m.group(1) if m else ""
        if found["title"] and video_id:
            entries.append({"id": video_id, "title": found["title"]})
    return entries


def build_rows(
    videos: Iterable[dict[str, Any]],
    rss_entries: Iterable[dict[str, Any]],
    *,
    compose: bool = True,
) -> list[dict[str, str]]:
    """``compose=True`` (default): fill ``suggested_title`` via
    :func:`compose_suggested_title`. ``compose=False`` reproduces the
    plain (non-compose) behaviour byte-for-byte (:func:`build_suggested_title`
    only, always ``needs-human`` without a structured field)."""
    videos = list(videos)
    seen_ids = {v["id"] for v in videos}
    combined = videos + [r for r in rss_entries if r["id"] not in seen_ids]
    title_counts = Counter(v["title"] for v in combined)
    dup_seen: Counter = Counter()  # occurrence index within a duplicate-title group
    rows: list[dict[str, str]] = []
    for v in combined:
        title = v["title"]
        is_dup = title_counts.get(title, 0) > 1
        reasons = [
            reason
            for reason, hit in (
                ("duplicate-title", is_dup),
                ("highlights-from", is_highlights_from(title)),
                ("county-only-title", is_county_only_title(title)),
                ("preposition-led-label", has_preposition_led_label(title)),
            )
            if hit
        ]
        if compose:
            dup_index = None
            if is_dup:
                dup_seen[title] += 1
                dup_index = dup_seen[title]
            suggested = compose_suggested_title(
                v, dup_index=dup_index, dup_total=title_counts.get(title) if is_dup else None
            )
        else:
            suggested = build_suggested_title(v)
        rows.extend(
            {
                "video_id": v["id"],
                "current_title": title,
                "reason": reason,
                "suggested_title": suggested,
                "platform": "youtube",
            }
            for reason in reasons
        )
    return rows


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-yaml", required=True, type=Path)
    parser.add_argument("--rss", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--no-compose",
        dest="compose",
        action="store_false",
        help="Skip title composition; only structured video fields fill suggested_title.",
    )
    parser.set_defaults(compose=True)
    args = parser.parse_args(argv)

    videos = load_videos(args.videos_yaml)
    rss_entries = load_rss_entries(args.rss) if args.rss else []
    rows = build_rows(videos, rss_entries, compose=args.compose)
    write_csv(rows, args.out)
    counts = Counter(row["reason"] for row in rows)
    print(f"wrote {len(rows)} row(s) to {args.out}")
    for reason, count in sorted(counts.items()):
        print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
