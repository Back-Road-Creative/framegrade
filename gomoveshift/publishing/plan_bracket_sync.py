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
old title is one the plan's generator wrote (the ``"<date>, Nth stretch"`` stamp, or a measured place). Anything else is
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
from datetime import date
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


def _stamp_ordinal(short: dict) -> Optional[int]:
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


def ordinal_basis(shorts: List[dict]) -> List[int]:
    """Each Short's 1-based position in its day, by start time, from the plan itself:
    the day sidecar beside the Short's media (``shorts_ost_audit_<DATE>.json``, the one
    Phase D reads) when it is there, else the Short's own verified title stamp (a plan
    written before fact titles), else its position among the plan's Shorts of the same
    recording date in plan order (LF, then short index)."""
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
        _load_day_sidecar_entries,
        _short_sidecar_ref,
    )
    from gomoveshift.pipeline.social_generator import day_title_ordinals

    cache: dict = {}
    by_day: Dict[str, List[int]] = {}
    for pos, s in enumerate(shorts):
        by_day.setdefault(str(s.get("recording_date")), []).append(pos)
    for day in by_day.values():
        day.sort(key=lambda p: (int(shorts[p]["lf_index"]), int(shorts[p]["short_index"])))
    out: List[int] = []
    for pos, s in enumerate(shorts):
        found: Optional[int] = None
        ref = _short_sidecar_ref((s.get("media") or {}).get("path"))
        if ref is not None:
            sidecar, _date, rank = ref
            try:
                entries = _load_day_sidecar_entries(sidecar, cache)
                if ("ordinals", sidecar) not in cache:
                    cache[("ordinals", sidecar)] = day_title_ordinals(entries)
                found = cache[("ordinals", sidecar)].get(rank)
            except (KeyError, TypeError, ValueError):
                found = None
        if found is None:
            found = _stamp_ordinal(s)
        if found is None:
            found = by_day[str(s.get("recording_date"))].index(pos) + 1
        out.append(found)
    return out


def generated_title(short: dict, extra_places: Tuple[str, ...] = ()) -> bool:
    """Whether the old YouTube title is one this plan's generator wrote: carrying its
    ``"<date>, Nth stretch"`` stamp, or exactly one of the Short's measured places (or the
    LF location / bracket label in ``extra_places``) before its hashtags."""
    from gomoveshift.pipeline.social_generator import _norm, measured_places
    from gomoveshift.pipeline.text_format import split_trailing_hashtags

    if _stamp_ordinal(short) is not None:
        return True
    body = _norm(split_trailing_hashtags(short["payloads"]["youtube"]["title"])[0])
    places = {
        _norm(b)
        for _, b in measured_places(
            short.get("sampled_regions") or [],
            short.get("sampled_roads") or [],
            short.get("route_record"),
        )
    }
    places |= {_norm(x) for x in extra_places if x}
    return bool(body) and body in places


def location_places(location: str, record: Optional[dict] = None) -> Tuple[str, ...]:
    """The LF location, its first component and the bracket label: what a stored title
    may name besides a Short's measured places."""
    label = str(record["label"]).strip() if record else ""
    return (location, location.split(",")[0].strip(), label)


def _stored_arms(short: dict) -> dict:
    """The Shorts-experiment arms (``SHORTS_FACTORS``) the plan recorded for ``short``."""
    from .experiments import SHORTS_FACTORS

    return {f: a for f, a in (short.get("experiment_arms") or {}).items() if f in SHORTS_FACTORS}


def _generate_lf(
    lf_shorts: List[dict],
    record: Optional[dict],
    location: str,
    lf_index: int,
    *,
    spooled_texts: Optional[dict] = None,
    experiment_today: Optional[date] = None,
    previous_places_at: Optional[Dict[int, dict]] = None,
    used_titles: Optional[dict] = None,
) -> list:
    """Phase D's per-LF generation (``generate_all``) over ``lf_shorts``.

    With a bracket ``record``, those it may rewrite carry the label as location/regions
    and the rest are generated and discarded. Without one (``queue_short_text``), an
    eligible Short keeps its own sampled regions, roads and route record. The bracket
    path reads no queue text (``spooled_texts`` ``{}``) and takes today's date."""
    from gomoveshift.pipeline.social_generator import SocialGenerator, TitleFacts

    label = str(record["label"]).strip() if record else None
    gen = SocialGenerator(
        location=location,
        project_name=f"{lf_shorts[0]['project_name']}_lf{lf_index:02d}",
        series_part=lf_index,
        spooled_texts={} if spooled_texts is None else spooled_texts,
        experiment_today=experiment_today,
        **({"bracket_route": record} if record else {}),
    )
    facts, locs, regions, roads, routes = [], [], [], [], []
    for s in lf_shorts:
        # ``_discard`` (queue_short_text): a stored Short rides along with stamped
        # stand-in text, discarded, so its legacy composition cannot raise for the others
        n = s.get("_ordinal")
        discard = bool(s.get("_discard"))
        if discard:
            n = n or 1
        elif s.get("_skip"):
            n = None
        eligible = n is not None
        own = [label] if record else [str(r) for r in s.get("sampled_regions") or []]
        if discard:
            own = []
        facts.append(
            TitleFacts(
                date=s["recording_date"],
                ordinal=n,
                secs=max(1, int(round(float(s["media"]["duration_seconds"])))),
                # a discarded Short names no place, so it takes no measured place
                place_ok=not discard,
            )
            if eligible
            else None
        )
        locs.append(own[0] if eligible and len(own) == 1 else location)
        regions.append(own if eligible else [])
        roads.append(list(s.get("sampled_roads") or []) if eligible and not discard else [])
        routes.append(
            None if record or not eligible or discard else (s.get("route_record") or None)
        )
    return gen.generate_all(
        num_shorts=len(lf_shorts),
        per_short_title_facts=facts,
        global_indexes=[s["short_index"] for s in lf_shorts],
        per_short_locations=locs,
        per_short_regions=regions,
        per_short_roads=roads,
        per_short_route_records=routes,
        previous_places_at=previous_places_at,
        used_titles=used_titles,
        # Each Short's stored arms, rendered verbatim: text must keep matching them.
        per_short_shorts_arms=[{} if s.get("_skip") else _stored_arms(s) for s in lf_shorts],
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
    ordinals = ordinal_basis(shorts)
    used_titles: dict = {}  # one ledger across the plan's LFs, as Phase D keeps one per drive
    for (project, lf_index), positions in lfs.items():
        positions.sort(key=lambda p: shorts[p]["short_index"])
        record, location, why = _lf_record(sidecar.parent, project, lf_index)
        lf_view = []
        for pos in positions:
            s = shorts[pos]
            reason = _refusal(s, record, why)
            if not reason and not generated_title(s, location_places(location, record)):
                reason = "old title is not one this plan generated (no stamp, no measured place)"
            if reason == "has a route record":
                verdict[pos] = ("unchanged", reason)
            elif reason:
                verdict[pos] = ("refused", reason)
            lf_view.append({**s, "_skip": pos in verdict, "_ordinal": ordinals[pos]})
        if len(verdict) and all(p in verdict for p in positions):
            continue
        try:
            posts = _generate_lf(lf_view, record, location, lf_index, used_titles=used_titles)
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
            stored = _stored_arms(shorts[pos])
            lost = {f: a for f, a in stored.items() if (post.experiment_arms or {}).get(f) != a}
            if lost:
                verdict[pos] = (
                    "refused",
                    f"stored arm(s) {lost!r} can no longer be rendered by the generator "
                    "(hand title, place not verified, unknown arm, or no usable text)",
                )
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
