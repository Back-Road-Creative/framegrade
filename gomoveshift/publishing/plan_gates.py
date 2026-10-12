"""Publish-plan gates — the shorts region, timing and grammar invariants,
enforced before ``plan_builder.build_publish_plan`` returns a plan.

These moved out of ``pipeline/csv_generator.py`` (the retired shorts CSV
writer) so the publish-plan sidecar owns its own upstream-of-irreversible-
artifact gates rather than borrowing them from a deliverable it doesn't
write. ``csv_generator.py`` keeps calling the same logic through a private
alias until its writer is deleted.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

# ``gomoveshift.pipeline``'s package __init__ eagerly imports csv_generator,
# which imports this module's public names back -- a module-level import
# here of anything under ``gomoveshift.pipeline`` would deadlock on that
# cycle whenever this module (or a caller of it, e.g. plan_builder) is
# imported before ``gomoveshift.pipeline`` itself. Absolute-but-deferred
# (never relative) imports at each call site sidestep it.


def assert_shorts_not_scheduled_before_publish(
    shorts_by_lf: "Dict[int, List]",
    slots_by_short_index: "Dict[int, object]",
    parent_publish_at_by_short_index: "Dict[int, Optional[str]]",
) -> None:
    """Runtime gate: no short may post before its parent LF's ``publish_at``.

    Each short's ``First Comment Text`` links to its parent LF's YouTube
    upload. LFs are uploaded ``private`` and flip public at ``publish_at``
    (``youtube_uploader`` forces ``privacyStatus='private'`` until then), so a
    short scheduled BEFORE that instant ships a first comment whose link 404s
    from post time until the LF publishes. When a promo cycle leads the
    drop by a day, the lead short's link is dead for that day.

    Slots carry America/New_York local ``date``/``time``; ``publish_at`` is an
    ISO-8601 UTC string. The two are compared as instants. A ``publish_at`` of
    ``None`` (immediate publish) imposes no constraint, and a caller that omits
    the map entirely (skip-upload audit / legacy) is unaffected. This is the
    upstream-of-irreversible-artifact gate — writing the publish plan is the
    irreversible step.
    """
    from gomoveshift.pipeline.schedule_constants import EST

    violations: "List[tuple]" = []
    for lf_idx in sorted(shorts_by_lf):
        for post in shorts_by_lf[lf_idx]:
            si = getattr(post, "short_index", None)
            pub_iso = (parent_publish_at_by_short_index or {}).get(si)
            slot = slots_by_short_index.get(si)
            if not pub_iso or slot is None:
                continue
            try:
                pub_dt = datetime.fromisoformat(pub_iso).astimezone(timezone.utc)
            except (ValueError, TypeError):
                continue
            date_s = (getattr(slot, "date", "") or "").strip()
            time_s = (getattr(slot, "time", "") or "").strip() or "00:00:00"
            slot_dt = None
            for tfmt in ("%H:%M:%S", "%H:%M"):
                try:
                    naive = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d " + tfmt)
                except ValueError:
                    continue
                slot_dt = naive.replace(tzinfo=EST).astimezone(timezone.utc)
                break
            if slot_dt is not None and slot_dt < pub_dt:
                violations.append((lf_idx, si, date_s, time_s, pub_iso))
    if violations:
        lf_idx, si, d, t, pub = violations[0]
        raise AssertionError(
            f"PUBLISH-PLAN-SHORT-BEFORE-PUBLISH: {len(violations)} short(s) "
            f"scheduled before their parent LF goes public — the first-comment "
            f"YouTube link 404s until publish. First: LF {lf_idx} short {si} posts "
            f"{d} {t} EST but parent publishes {pub}. Reschedule the short to "
            f">= publish_at, or drop the LF link from its first comment."
        )


def _post_surfaces(post) -> "List[tuple]":
    """Every plan text field a shipped short surfaces, paired with a label.

    Covers title, the legacy shared caption, per-platform ``captions``
    (skipping ``captions['youtube']`` — the YouTube payload never carries a
    caption; ``youtube_description``/``youtube_title`` are its surfaces),
    the discrete tag list, ``tiktok_title``, and ``youtube_description``.
    """
    tag_list = [
        t.strip() for t in (getattr(post, "youtube_tags", "") or "").split(",") if t.strip()
    ]
    surfaces = [
        ("title", getattr(post, "youtube_title", "") or ""),
        ("caption", getattr(post, "caption", "") or ""),
    ]
    surfaces += [("tag", t) for t in tag_list]
    for platform, text in (getattr(post, "captions", None) or {}).items():
        if platform == "youtube":
            continue
        surfaces.append((f"caption:{platform}", text or ""))
    surfaces.append(("tiktok_title", getattr(post, "tiktok_title", "") or ""))
    surfaces.append(("youtube_description", getattr(post, "youtube_description", "") or ""))
    return surfaces


def assert_shorts_metadata_in_region(shorts_by_lf: "Dict[int, List]") -> None:
    """Runtime gate: every shipped short surface names only in-region counties.

    A short's title, captions, and tags publish verbatim into the publish
    plan. Each is derived from that short's OWN resolved region, but the
    social generator also seeds a per-short clone with the parent-LF
    ``county`` — which is unconditionally emitted as a tag AND can fill the
    ``{county}`` title template — so a short cut from a different county can
    ship a cross-region county its own GPS never visited (the far-corner
    class; the shorts analog of the long-form ``location_invariant`` gate).

    ``SocialPost.sampled_regions`` is the short's own region label(s) (its
    region-verified OST label). The discrete, label-form **tag** surface is
    the ground truth for which counties the generator emitted: the
    ``content_gates`` county extractor is reliable on a clean label but
    MISFIRES on free-form prose (``"Rolling through County Kerry"`` yields
    the verb phrase, not the county), so a foreign county is only recognised
    from the tags, then every other surface is reported when it carries that
    SAME confirmed foreign county. Prose-misfire artefacts never match a real
    foreign county, so an in-region short never false-fails.

    Upstream-of-irreversible-artifact gate — raises BEFORE the publish plan
    is built so a cross-region short can never reach it. Reuses the
    ``content_gates`` region-token helper (never re-implements it). No-op
    when a short has no sampled regions or no county-form region (GPS-less /
    legacy / town-only label) — mirrors the long-form gate's legacy skip.
    """
    from gomoveshift.pipeline.validation.content_gates import _county_names

    failures: "List[str]" = []
    for lf_idx in sorted(shorts_by_lf):
        for post in shorts_by_lf[lf_idx]:
            regions = [
                r
                for r in (getattr(post, "sampled_regions", None) or [])
                if isinstance(r, str) and r.strip()
            ]
            region_counties: set = set()
            for r in regions:
                region_counties |= _county_names(r)
            if not region_counties:
                # No county-form region to compare against — GPS-less / legacy /
                # town-only label. Skip (fail-open), matching the LF gate.
                continue
            tag_list = [
                t.strip() for t in (getattr(post, "youtube_tags", "") or "").split(",") if t.strip()
            ]
            # Ground truth: counties named in the clean, label-form tag surface.
            foreign: set = set()
            for tag in tag_list:
                foreign |= _county_names(tag) - region_counties
            if not foreign:
                continue
            si = getattr(post, "short_index", "?")
            for kind, text in _post_surfaces(post):
                # A confirmed foreign county is looked up as a plain word in
                # every OTHER surface, never re-extracted with the label
                # regex — that regex is reliable on the clean, label-form tag
                # but misfires on free-form prose (it can swallow the county
                # name into a preceding verb phrase), so re-running it here
                # would silently drop a real cross-region caption/title hit.
                text_lower = (text or "").lower()
                hit = {
                    county
                    for county in foreign
                    if re.search(rf"\b{re.escape(county)}\b", text_lower)
                }
                if hit:
                    failures.append(
                        f"short #{si} {kind} {text!r}: county {sorted(hit)} absent "
                        f"from the clip's regions {regions}"
                    )
    if failures:
        raise ValueError(
            f"PUBLISH-PLAN-CROSS-REGION: {len(failures)} shorts surface(s) name a "
            f"county absent from the clip's own sampled regions — the shipped "
            f"title/caption/tags desynced from the short's footage (far-corner "
            f"class). Fix the source label / drop the cross-region tag before "
            f"the publish plan is built. {'; '.join(failures)}"
        )


def assert_plan_text_grammar(shorts_by_lf: "Dict[int, List]") -> None:
    """Runtime gate: no shipped short surface may carry a
    preposition-collision splice.

    Reuses ``content_gates.find_grammar_issues`` (string-based, independent
    of any structured location label) against every plan text field —
    title, captions, tags, ``tiktok_title``, ``youtube_description`` — the
    surfaces that publish verbatim into the plan. Upstream-of-irreversible-
    artifact gate: raises naming the offending short/surface BEFORE the
    publish plan is built, so a spliced string like ``"drive through To
    Clonakilty"`` or ``"#ToClonakilty"`` can never reach it.
    """
    from gomoveshift.pipeline.validation.content_gates import find_grammar_issues

    failures: "List[str]" = []
    for lf_idx in sorted(shorts_by_lf):
        for post in shorts_by_lf[lf_idx]:
            si = getattr(post, "short_index", "?")
            for kind, text in _post_surfaces(post):
                for msg in find_grammar_issues(text):
                    failures.append(f"short #{si} {kind}={text!r}: {msg}")
    if failures:
        raise AssertionError(
            f"PUBLISH-PLAN-GRAMMAR: {len(failures)} grammar issue(s) in the "
            f"publish plan. First: {failures[0]}"
        )
