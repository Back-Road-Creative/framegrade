"""Regenerate a queued bracket-route Short's text in its publish plan.

A GPS-less drive's Shorts planned before ``SocialGenerator(bracket_route=...)`` are
titled with the label's destination alone ("Pennsylvania" for "Virginia to
Pennsylvania"), carry no ``bracket_route`` and no route record, and the Short review
refuses each as ``description_no_clip_place``. ``sync_bracket_short_text`` rebuilds such a
Short's text with the generator path Phase D uses (``SocialGenerator(...,
bracket_route=record).generate_all`` over the parent LF, then the plan builder's own
payload assembly) and stores the record on the Short, so the review judges the label
against it. Nothing here writes wording of its own: the text is the generator's.

A Short is touched only when ALL hold: no ``route_record``; its parent LF sidecar
(``<project>_lf{NN}_youtube.json`` beside the plan) holds a valid
``drive_text_inputs.bracket_route``; its ``sampled_regions`` is exactly ``[label]``; its
old title carries the plan's own ``"<date>, Nth stretch"`` stamp. Anything else is
refused and left as it was. A rewritten Short must then pass ``short_review_refusal``, or
it is refused too. YouTube title, description and tags are regenerated; another
platform's text is replaced only when it names the collapsed destination and not the
label. Dry run unless ``apply``; the new plan is parsed by ``load_publish_plan`` before
it replaces the old one, and ``reviewed_plan_hash`` stays a stored field. No queue
database, no network.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .plan_io import load_publish_plan

#: ``(payload, field)`` of each non-YouTube text the generator also composes.
_OTHER_TEXT = (
    ("instagram", "caption"),
    ("facebook", "description"),
    ("tiktok", "title"),
    ("tiktok", "caption"),
)
_ORDINAL_RE = re.compile(r"(\d+)(?:st|nd|rd|th) stretch")


@dataclass(frozen=True)
class BracketShortChange:
    project_name: str
    short_index: int
    status: str  # "changed" | "unchanged" | "refused"
    old_title: str
    new_title: str
    reason: str = ""
    fields: Tuple[str, ...] = ()
    written: bool = False


def _lf_record(plan_dir: Path, project: str, lf_index: int) -> Tuple[Optional[dict], str, str]:
    """``(valid bracket record | None, LF location, why not)`` from the LF sidecar."""
    from gomoveshift.pipeline.bracket_route import bracket_record_valid

    path = plan_dir / f"{project}_lf{lf_index:02d}_youtube.json"
    try:
        side = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, "", f"LF sidecar {path.name} unreadable: {exc}"
    record = (side.get("drive_text_inputs") or {}).get("bracket_route")
    if not bracket_record_valid(record):
        return None, "", f"LF sidecar {path.name} has no valid bracket_route"
    return record, str(side.get("location") or ""), ""


def _refusal(short: dict, record: Optional[dict], why: str) -> str:
    """Why this Short may not be rewritten ("" when it may)."""
    if short.get("route_record") and (short["route_record"] or {}).get("bins"):
        return "has a route record"
    if why:
        return why
    label = str(record["label"]).strip()
    regions = [str(r).strip() for r in short.get("sampled_regions") or []]
    if regions != [label]:
        return f"sampled_regions {regions!r} is not exactly [{label!r}]"
    return ""


def _ordinal(short: dict) -> Optional[int]:
    """The ``Nth stretch`` position the plan's own title stamp records, or ``None``."""
    from gomoveshift.pipeline.social_generator import title_stamp

    title = short["payloads"]["youtube"]["title"]
    m = _ORDINAL_RE.search(title)
    if not m:
        return None
    try:
        stamp = title_stamp(short["recording_date"], int(m.group(1)))
    except (KeyError, ValueError):
        return None
    return int(m.group(1)) if title.startswith(stamp + ":") else None


def _generate_lf(
    lf_shorts: List[dict],
    record: Optional[dict],
    location: str,
    lf_index: int,
    *,
    pinned_arms: Optional[Dict[int, dict]] = None,
) -> list:
    """Phase D's per-LF generation (``generate_all``) over ``lf_shorts``.

    With a bracket ``record``, those it may rewrite carry the label as location/regions
    and the rest are generated and discarded. Without one (``queue_short_text``), an
    eligible Short keeps its own sampled regions, roads and route record, and its
    ``pinned_arms`` entry (by ``short_index``) stands in for a fresh arm assignment."""
    from gomoveshift.pipeline.social_generator import SocialGenerator, TitleFacts

    label = str(record["label"]).strip() if record else None
    gen = SocialGenerator(
        location=location,
        project_name=f"{lf_shorts[0]['project_name']}_lf{lf_index:02d}",
        series_part=lf_index,
        spooled_texts={},  # never read the queue database
        **({"bracket_route": record} if record else {}),
        **({"pinned_arms": pinned_arms} if pinned_arms else {}),
    )
    facts, locs, regions, roads, routes = [], [], [], [], []
    for s in lf_shorts:
        n = None if s.get("_skip") else _ordinal(s)
        eligible = n is not None and not s.get("_skip")
        own = [label] if record else [str(r) for r in s.get("sampled_regions") or []]
        facts.append(
            TitleFacts(
                date=s["recording_date"],
                ordinal=n,
                secs=max(1, int(round(float(s["media"]["duration_seconds"])))),
                place_ok=True,
            )
            if eligible
            else None
        )
        locs.append(own[0] if eligible and len(own) == 1 else location)
        regions.append(own if eligible else [])
        roads.append(list(s.get("sampled_roads") or []) if eligible else [])
        routes.append(None if record or not eligible else (s.get("route_record") or None))
    return gen.generate_all(
        num_shorts=len(lf_shorts),
        per_short_title_facts=facts,
        global_indexes=[s["short_index"] for s in lf_shorts],
        per_short_locations=locs,
        per_short_regions=regions,
        per_short_roads=roads,
        per_short_route_records=routes,
    )


def _rewrite(short: dict, post, record: dict) -> Tuple[dict, Tuple[str, ...]]:
    """A copy of ``short`` carrying ``post``'s text and the bracket record, and the
    names of the text fields that changed."""
    from .plan_builder import build_short_payloads

    label = str(record["label"]).strip()
    destination = label.rsplit(" to ", 1)[-1].strip()
    new = copy.deepcopy(short)
    payloads = new["payloads"]
    tiktok = payloads.get("tiktok") or {}
    built = build_short_payloads(
        post,
        recording_date=short["recording_date"],
        parent_id=short["parent_lf_video_id"],
        tiktok_privacy_level=tiktok.get("privacy_level") or "SELF_ONLY",
    )
    fields: List[str] = []
    for name, value in (
        ("title", built.youtube.title),
        ("description", built.youtube.description),
        ("tags", list(built.youtube.tags)),
    ):
        if payloads["youtube"].get(name) != value:
            payloads["youtube"][name] = value
            fields.append(f"youtube.{name}")
    changed_platforms = set()
    for platform, field in _OTHER_TEXT:
        old_text = (payloads.get(platform) or {}).get(field)
        if not old_text or destination not in old_text or label in old_text:
            continue
        value = getattr(getattr(built, platform), field)
        if value != old_text:
            payloads[platform][field] = value
            fields.append(f"{platform}.{field}")
            changed_platforms.add(platform)
    templates = dict(new.get("caption_templates") or {})
    for platform in changed_platforms & set(post.caption_templates):
        templates[platform] = post.caption_templates[platform]
    if templates:
        new["caption_templates"] = templates
    if new.get("bracket_route") != record:
        new["bracket_route"] = dict(record)
        fields.append("bracket_route")
    return new, tuple(fields)


def _parsed(raw: dict, sidecar: Path):
    """``load_publish_plan`` of ``raw`` via the sidecar's temp file (always removed)."""
    tmp = sidecar.with_name(sidecar.name + ".sync-tmp")
    tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        return load_publish_plan(tmp), tmp
    except Exception:
        tmp.unlink()
        raise


def sync_bracket_short_text(sidecar: Path, *, apply: bool) -> List[BracketShortChange]:
    """Rebuild each eligible Short's text in the publish plan ``sidecar`` (see the module
    docstring); one :class:`BracketShortChange` per Short, in plan order."""
    from .youtube_adapter import short_review_refusal

    sidecar = Path(sidecar)
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    load_publish_plan(sidecar)  # the input must already be a valid plan
    shorts: List[dict] = raw["shorts"]
    verdict: Dict[int, Tuple[str, str]] = {}  # position -> (status, reason)
    candidate: Dict[int, Tuple[dict, Tuple[str, ...]]] = {}

    lfs: Dict[Tuple[str, int], List[int]] = {}
    for pos, s in enumerate(shorts):
        lfs.setdefault((s["project_name"], int(s["lf_index"])), []).append(pos)
    for (project, lf_index), positions in lfs.items():
        positions.sort(key=lambda p: shorts[p]["short_index"])
        record, location, why = _lf_record(sidecar.parent, project, lf_index)
        lf_view = []
        for pos in positions:
            s = shorts[pos]
            reason = _refusal(s, record, why)
            if not reason and _ordinal(s) is None:
                reason = "old title has no verifiable 'Nth stretch' stamp"
            if reason == "has a route record":
                verdict[pos] = ("unchanged", reason)
            elif reason:
                verdict[pos] = ("refused", reason)
            lf_view.append({**s, "_skip": pos in verdict})
        if len(verdict) and all(p in verdict for p in positions):
            continue
        try:
            posts = _generate_lf(lf_view, record, location, lf_index)
        except Exception as exc:  # the generator's own refusal is the reason
            for pos in positions:
                verdict.setdefault(pos, ("refused", f"generator: {exc}"))
            continue
        for pos, post in zip(positions, posts):
            if pos in verdict:
                continue
            if not getattr(post, "bracket_route", None):
                verdict[pos] = ("refused", "generator did not back the label with the record")
                continue
            new, fields = _rewrite(shorts[pos], post, record)
            if fields:
                candidate[pos] = (new, fields)
            else:
                verdict[pos] = ("unchanged", "text already matches the generator")

    # The review a publish runs, on each rewritten Short, before anything is written.
    trial = copy.deepcopy(raw)
    for pos, (new, _) in candidate.items():
        trial["shorts"][pos] = new
    plan, tmp = _parsed(trial, sidecar)
    tmp.unlink()
    for pos, (_new, _fields) in list(candidate.items()):
        refused = short_review_refusal(plan.shorts[pos])
        if refused:
            del candidate[pos]
            verdict[pos] = ("refused", f"still refused by the Short review: {refused}")
    final = copy.deepcopy(raw)
    for pos, (new, _) in candidate.items():
        final["shorts"][pos] = new
    if candidate:
        _, tmp = _parsed(final, sidecar)
        if apply:
            os.replace(tmp, sidecar)
        else:
            tmp.unlink()

    out: List[BracketShortChange] = []
    for pos, s in enumerate(shorts):
        old_title = s["payloads"]["youtube"]["title"]
        if pos in candidate:
            new, fields = candidate[pos]
            out.append(
                BracketShortChange(
                    s["project_name"],
                    s["short_index"],
                    "changed",
                    old_title,
                    new["payloads"]["youtube"]["title"],
                    fields=fields,
                    written=apply,
                )
            )
        else:
            status, reason = verdict[pos]
            out.append(
                BracketShortChange(
                    s["project_name"], s["short_index"], status, old_title, old_title, reason
                )
            )
    return out
