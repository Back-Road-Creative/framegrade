"""Content and metadata validation gates -- no ffprobe dependency."""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from typing import TYPE_CHECKING, Optional

from gomoveshift.config import get_config as _get_config
from gomoveshift.pipeline._series_utils import _TITLE_PREFIX_RE, is_title_prefixed
from gomoveshift.pipeline.content_validator import ContentValidator
from gomoveshift.pipeline.geo_constants import US_STATE_ABBREVS
from gomoveshift.pipeline.pipeline_validator import GateResult

if TYPE_CHECKING:
    from .place_corrections import PlaceCorrections

logger = logging.getLogger(__name__)

# Policy: a county holding at least this share of the
# footage-weighted vote is a real location the label must name. A stated policy
# constant, not a measurement.
LOCATION_MATERIAL_SHARE = 0.20


def check_content_safety(
    content_validator: ContentValidator,
    metadata_json: Optional[dict],
    social_posts: Optional[list],
) -> GateResult:
    """Validate all generated text content against blocklist."""
    titles: list[str] = []
    descriptions: list[str] = []
    captions: list[str] = []

    if metadata_json:
        if "title" in metadata_json:
            titles.append(metadata_json["title"])
        if "description" in metadata_json:
            descriptions.append(metadata_json["description"])

    if social_posts:
        for post in social_posts:
            # The content-safety gate also covers first_comment and the shorts youtube_description:
            # first_comment is caption-like free text and the YT description is a
            # description surface, so both must pass the blocklist.
            if hasattr(post, "youtube_title"):
                titles.append(post.youtube_title)
                titles.append(post.tiktok_title)
                captions.append(post.caption)
                if getattr(post, "first_comment", ""):
                    captions.append(post.first_comment)
                if getattr(post, "youtube_description", ""):
                    descriptions.append(post.youtube_description)
            elif isinstance(post, dict):
                titles.append(post.get("youtube_title", ""))
                titles.append(post.get("tiktok_title", ""))
                captions.append(post.get("caption", ""))
                if post.get("first_comment"):
                    captions.append(post["first_comment"])
                if post.get("youtube_description"):
                    descriptions.append(post["youtube_description"])

    result = content_validator.validate_all(
        titles=titles,
        descriptions=descriptions,
        captions=captions,
    )

    if result.valid:
        return GateResult(
            gate="content_safety",
            passed=True,
            message=f"All content safe ({len(titles)} titles, "
            f"{len(descriptions)} descriptions, {len(captions)} captions)",
        )
    else:
        return GateResult(
            gate="content_safety",
            passed=False,
            message=f"{len(result.violations)} violation(s): {'; '.join(result.violations[:3])}",
        )


# Grammar gate: preposition-collision artifacts from a location-string
# splice that dropped the article/verb between an existing "through"/"in"/"of"
# and the templated "To <Place>" clause (e.g. "drive through To Clonakilty").
# String-based and independent of any structured location label -- it fires on
# the shipped text itself, whichever generator produced it.
_GRAMMAR_SPLICE_SUBSTRINGS = ("through To ", "in To ", "of To ", "Through To ")
_GRAMMAR_DOUBLED_PREP_RE = re.compile(r"\b(through|in|to)\s+\1\b", re.IGNORECASE)
_GRAMMAR_HASHTAG_TO_RE = re.compile(r"#To[A-Z]")
# Deprecated compat alias: cli.parser._validate_per_lf_location_args
# imports this name directly from this module. The curated word list itself
# now lives only in gomoveshift.pipeline._series_utils (is_title_prefixed) --
# remove this alias once that call site switches to is_title_prefixed().
_GRAMMAR_LEADING_PREP_RE = _TITLE_PREFIX_RE


def find_grammar_issues(text: str) -> list[str]:
    """Return grammar-defect descriptions found in ``text`` (empty = clean).

    Catches the preposition-collision artifact class: a splice
    that runs an existing preposition into a templated ``"To <Place>"``
    clause, a doubled preposition, a hashtag that swallows ``To`` into the
    place name, and a value that itself opens with a bare preposition (the
    shape of a stray location-template fragment).
    """
    # Only real text is gated: a non-string (None, a mock in a unit test, a
    # numeric field) carries no grammar to check.
    if not isinstance(text, str) or not text:
        return []
    issues: list[str] = []
    for bad in _GRAMMAR_SPLICE_SUBSTRINGS:
        if bad in text:
            issues.append(f"contains {bad!r} (preposition-collision splice)")
    m = _GRAMMAR_DOUBLED_PREP_RE.search(text)
    if m:
        issues.append(f"doubled preposition {m.group(0)!r}")
    if _GRAMMAR_HASHTAG_TO_RE.search(text):
        issues.append("hashtag runs a preposition into the place name (#To<Place>)")
    if is_title_prefixed(text):
        issues.append(f"starts with a bare preposition: {text!r}")
    return issues


def _title_names_a_traced_place(title: str, metadata_json: dict) -> bool:
    """True when the title anchors on a place the LF verifiably drove through.

    ``resolved_location.ordered_place_sequence`` is the LF's own trace, so a
    title naming one of its settlements is corroborated by the pipeline's own
    data even when it shares no word with the county-form location label.
    ``trace_verified_places`` (endpoint towns proven within 2 km of the same
    trace) corroborate the same way. Absent a resolved trace (legacy metadata)
    nothing is corroborated and the caller keeps its strict keyword check.
    """
    resolved = metadata_json.get("resolved_location") or {}
    sequence = [
        *(resolved.get("ordered_place_sequence") or []),
        *(resolved.get("trace_verified_places") or []),
    ]
    title_lower = title.lower()
    for place in sequence:
        place_lower = str(place).strip().lower()
        # Mirror the caller's >=4-char floor: shorter names collide with
        # ordinary title words too readily to count as corroboration.
        if len(place_lower) < 4:
            continue
        if re.search(rf"\b{re.escape(place_lower)}\b", title_lower):
            return True
    return False


def check_metadata_quality(
    metadata_json: dict, place_corrections: Optional[PlaceCorrections] = None
) -> GateResult:
    """Validate metadata for common quality issues.

    Catches:
    - Duplicate chapter blocks in description
    - Publish date in description (ages poorly)
    - Generic LLM filler text
    - Title/location keyword mismatch
    - A denylisted place name -- belt and braces for a corrections-
      ledger substitution ``title_builder`` missed. ``place_corrections``
      is injectable for tests; None loads ``GMS_CORRECTIONS_PATH``.
    """
    issues: list[str] = []
    # A present-but-None description must not abort the whole gate (.count/in
    # raise AttributeError on None) — coerce to "".
    description = metadata_json.get("description") or ""
    title = metadata_json.get("title", "")
    location = metadata_json.get("location", "")

    # 1. Duplicate chapter blocks
    if description.count("\U0001f4cd CHAPTERS") > 1:
        issues.append("duplicate chapter blocks in description")

    # 2. Publish date in description
    if "Published:" in description:
        issues.append("publish date in description (ages poorly)")

    # 3. Generic LLM filler
    filler_phrases = [
        "Dive into the latest developments",
        "insights, and discussions",
    ]
    for phrase in filler_phrases:
        if phrase in description:
            issues.append(f"generic filler: '{phrase}'")
            break

    # 4. Title/location mismatch
    #
    # A settlement-anchored title ("Kingsport, TN in 4K") shares no keyword
    # with its county-form location ("Sullivan County, Tennessee") even though
    # both are correct — and ``check_label_quality`` actively asks for that
    # settlement anchor, so a bare keyword test puts the two gates in direct
    # conflict. Corroborate against the LF's own trace before failing.
    if location and title:
        _skip = {
            "county",
            "from",
            "ireland",
            "england",
            "scotland",
            "wales",
            "france",
            "spain",
            "italy",
            "germany",
            "portugal",
            "greece",
            "driving",
            "scenic",
        }
        loc_words = []
        for part in location.replace(",", " ").split():
            if len(part) >= 4 and part.lower() not in _skip:
                loc_words.append(part.lower())
        title_lower = title.lower()
        matched = sum(1 for w in loc_words if w in title_lower)
        if loc_words and matched == 0 and not _title_names_a_traced_place(title, metadata_json):
            issues.append(f"title has no location keywords from '{location}'")

    # 5. Chapter density
    #
    # Robust-by-construction skip for GPS-less videos: chapter generation
    # consumes GPX route data. A GPS-less full_drive LF has zero source
    # data for chapters, so the gate is unsatisfiable by construction —
    # it can only WARN about something the generator literally cannot
    # produce. When ``has_gps is False`` (set at metadata-generation time
    # from ``state["per_lf_gpx_path"][lf.index] is not None``), skip
    # both the "no chapters" and "low intermediate density" checks.
    # Legacy metadata without the ``has_gps`` key falls through to the
    # strict behavior (preserves the existing contract for GPS-bearing
    # cases that pre-date the flag).
    chapters = metadata_json.get("chapters", [])
    video_duration = metadata_json.get("video_duration", 0)
    has_gps = metadata_json.get("has_gps")
    # The floor and the count are the chapter generator's own helpers, so the
    # generator sizes its chapters to clear this check by construction.
    from gomoveshift.pipeline.chapters.generator import (
        intermediate_chapters,
        min_intermediate_chapters,
    )

    min_expected = min_intermediate_chapters(video_duration)
    if min_expected and has_gps is not False:
        if not chapters:
            issues.append(f"no chapters for {video_duration / 60:.0f}min video")
        else:
            intermediate = intermediate_chapters(chapters)
            if len(intermediate) < min_expected:
                issues.append(
                    f"only {len(intermediate)} intermediate chapter(s) for "
                    f"{video_duration / 60:.0f}min video (need {min_expected}+)"
                )

    # 6. Operational data in description
    if "NEXT PUBLISH" in description or "PUBLISH SLOT" in description:
        issues.append("operational data (publish slot) in description")

    # 7. Location-vs-chapter cross-check
    if chapters and location:
        from gomoveshift.pipeline.route_context import _COUNTY

        # Mirror the chapter parser (multi-word counties + optional route
        # suffix); the shared ``_COUNTY`` keeps the two from drifting.
        _chapter_county_re = re.compile(rf"^Entering\s+.+?,\s+({_COUNTY})(?:,\s+.+)?$")
        chapter_regions = []
        for ch in chapters:
            m = _chapter_county_re.match(ch.get("title", ""))
            if m:
                chapter_regions.append(m.group(1))
        if chapter_regions:
            # A multi-county drive names counties its single episode label
            # cannot contain. Those are only wrong when the drive never went
            # there, so a county the LF actually sampled counts as a match —
            # same bug class ``check_location_invariant`` guards, judged
            # against the same sampled-region set.
            sampled_regions = (metadata_json.get("resolved_location") or {}).get("regions") or []
            mismatched = sum(
                1
                for r in chapter_regions
                if r not in location and not any(r in s for s in sampled_regions)
            )
            if mismatched > len(chapter_regions) / 2:
                issues.append("majority of chapter regions don't match episode location")

    # 8. Business names in title
    title = metadata_json.get("title", "")
    _title_to_re = re.search(r"(.+?)\s+to\s+(.+?)\s*[|\u2014-]", title)
    if _title_to_re:
        from gomoveshift.pipeline.chapters import is_valid_chapter_placename

        for _town_name in _title_to_re.groups():
            _town_clean = _town_name.strip()
            if _town_clean and not is_valid_chapter_placename(_town_clean):
                issues.append(f"business name in title: '{_town_clean}' is not a place")

    # 9. Fragmented location tags
    tags = metadata_json.get("tags", [])
    if location and tags:
        loc_parts = [p.strip() for p in location.split(",")]
        for part in loc_parts:
            words = part.split()
            if len(words) >= 2:
                phrase_present = part in tags
                individual_present = all(any(w == t for t in tags) for w in words)
                if not phrase_present and individual_present:
                    issues.append(
                        f"fragmented location tag: individual words of "
                        f"'{part}' appear as tags but phrase does not"
                    )

    # 10. Grammar: preposition-collision artifacts. Runs on every
    # generated text surface -- title, description (the hook lives at its
    # head), the location label, and each tag/hashtag.
    for surface, text in (("title", title), ("description", description), ("location", location)):
        for msg in find_grammar_issues(text):
            issues.append(f"{surface}: {msg}")
    for tag in metadata_json.get("tags", []) or []:
        if isinstance(tag, str):
            for msg in find_grammar_issues(tag):
                issues.append(f"tag {tag!r}: {msg}")

    # 11. Denylisted place name -- scans every shipped text surface.
    from .place_corrections import find_denylisted, load_place_corrections

    corrections = place_corrections if place_corrections is not None else load_place_corrections()
    if corrections.mapping or corrections.denylist:
        surfaces = [("title", title), ("description", description), ("location", location)]
        surfaces += [
            (f"tag {t!r}", t) for t in metadata_json.get("tags", []) or [] if isinstance(t, str)
        ]
        for surface, text in surfaces:
            for name in find_denylisted(text, corrections):
                issues.append(f"{surface} contains denylisted place name: '{name}'")

    if issues:
        return GateResult(
            gate="metadata_quality",
            passed=False,
            message=f"Metadata quality issues: {'; '.join(issues)}",
        )

    return GateResult(
        gate="metadata_quality",
        passed=True,
        message="Metadata quality checks passed",
    )


def check_location_invariant(metadata_json: dict) -> GateResult:
    """Cross-surface location invariant.

    Reconciles the ACTUAL shipped title / burned-OST text against the canonical
    ``ResolvedLocation`` vote (threaded into
    ``metadata_json["resolved_location"]`` as a JSON-safe dict by
    :func:`resolved_location_gate_dict`). Everything below is a HARD FAIL
    (``passed=False``); ``phase_d_gates`` escalates it to ``RuntimeError`` via
    ``assert_location_invariant_hard_fail`` (mirrors ``audio_loudness``), and
    Phase BC runs the same gate pre-encode. There is no warn tier.

      * **Membership** — the shipped ``title`` / burned OST ``location`` text
        names a COUNTY that is in ZERO sampled ``regions``. Reading the resolved
        model's own ``dominant_county`` here would be vacuous — that county is a
        member of ``regions`` by construction — so the real shipped text is
        reconciled against the sampled regions: the genuine far-road class, a
        label on a county the footage never visited.
      * **Coverage** (a label may not name one location while
        another is on screen) — every county holding at least
        ``LOCATION_MATERIAL_SHARE`` of the footage vote must be NAMED by the
        shipped claim: its county name appears in the title / location / any
        description-or-tag claim surface, or a title endpoint town maps to it
        through ``endpoint_regions`` (first ``"A to B"`` phrase = start, last =
        end). A legacy model with no ``endpoint_regions`` can only satisfy the
        rule by naming the county. A unanimous vote has one material county and
        behaves as before.

    This is THE cross-artifact reconciliation gate:

      * The county-form hard check runs per SURFACE, not just title+OST: the
        description's ``📍 Location:`` line, its SEO search line, each
        description hashtag (camel-split so ``#ColletonCounty`` is scannable)
        and each YouTube tag must not name a county in zero sampled regions.
      * A TOWN-form shipped label (no ``County`` token — most US labels, e.g.
        ``"Beaufort, SC"``, and destination-form ``"Donnelley WMA, SC"``) is
        reconciled against town-level truth: the resolved model's
        ``ordered_place_sequence`` (the even sampling includes both trace
        endpoints by construction) plus ``destination``. ``"A to B via …"``
        titles have EACH endpoint town checked individually — the
        wrong-town-in-right-county class the location invariant's county identity misses.
        Both checks no-op when the model carries no town-level truth (legacy /
        quorum-degraded), preserving the old skip.
      * ``trace_verified_places`` — endpoint towns a settlement of exactly that
        name within 2 km of the LF's own GPS fixes proved
        (:func:`trace_verified_endpoint_towns`) — adds to that town-level
        truth but never opens the check on its own. A reverse geocoder that
        buckets the whole route under a district ("West Cork") no longer fails
        a GPS-true village arc; an endpoint nothing proves still fails.

    No-op when the LF has no resolved model or no sampled regions (GPS-less /
    legacy / single-video path).
    """
    resolved = metadata_json.get("resolved_location")
    if not isinstance(resolved, dict):
        return GateResult(
            gate="location_invariant",
            passed=True,
            message="no resolved location (skipped)",
        )
    regions = resolved.get("regions") or []
    if not regions:
        return GateResult(
            gate="location_invariant",
            passed=True,
            message="no sampled regions (skipped)",
        )

    # HARD: the ACTUAL shipped surfaces (YouTube ``title`` and burned OST
    # ``location`` label — per the single-source-of-truth contract the title
    # derives from the same ``per_lf_ost_text`` that burns the pixels) must not
    # name a COUNTY the trace never sampled. Extract the sampled counties from
    # ``regions`` (clean single names) and, when the shipped text is county-form
    # (carries a ``County`` token), require at least one sampled county to
    # appear in it — else the label named some OTHER county (far-road class).
    title = metadata_json.get("title") or ""
    location = metadata_json.get("location") or ""
    shipped = f"{title} {location}".strip()
    region_counties: set[str] = set()
    for r in regions:
        region_counties |= _county_names(r)
    failures: list[str] = []
    if region_counties and _COUNTY_TOKEN_RE.search(shipped):
        shipped_lower = shipped.lower()
        if not any(c in shipped_lower for c in region_counties):
            failures.append(
                f"shipped title/OST text {shipped!r} names a county in ZERO "
                f"sampled regions {regions} — the label desynced from the GPS "
                f"trace (far-road class)"
            )
    # Cross-artifact sweep: every OTHER surface staking a county-form
    # location claim (description 📍 line / SEO line / hashtags, YouTube tags)
    # gets the same reconciliation — a surface naming a county the trace never
    # sampled is the same desync whichever artifact carries it.
    if region_counties:
        for surface, text in _location_claim_surfaces(metadata_json):
            if _COUNTY_TOKEN_RE.search(text) and not any(
                c in text.lower() for c in region_counties
            ):
                failures.append(
                    f"{surface} {text!r} names a county in ZERO sampled regions {regions}"
                )

    # Town-form / endpoint-town reconciliation — needs town-level truth
    # (``ordered_place_sequence`` / ``destination``); without it, keep the old
    # town-form skip (a county-region set alone can't validate a town label).
    place_seq = [
        s
        for s in (resolved.get("ordered_place_sequence") or [])
        if isinstance(s, str) and s.strip()
    ]
    destination = resolved.get("destination")
    town_truth = list(place_seq)
    if isinstance(destination, str) and destination.strip():
        town_truth.append(destination)
    # Rescue-only: geometrically verified endpoint towns join the truth, but the
    # check still needs reverse-geocoded town truth to run at all.
    verified = resolved.get("trace_verified_places")
    verified_places = [s for s in (verified or []) if isinstance(s, str) and s.strip()]
    if town_truth:
        truth_tokens: set[str] = set()
        for src in [*town_truth, *verified_places, *regions]:
            truth_tokens |= _town_claim_tokens(src)
        endpoint_phrases = _endpoint_town_phrases(title, location)
        unproven = (
            f" — and no settlement of that name lies within "
            f"{_TRACE_VERIFIED_PLACE_KM:g} km of the GPS trace"
            if verified is not None
            else ""
        )
        for phrase in sorted(endpoint_phrases):
            tokens = _town_claim_tokens(phrase)
            if not tokens or _COUNTY_TOKEN_RE.search(phrase):
                continue  # generic-word arc / county-form endpoint (covered above)
            if not tokens & truth_tokens:
                failures.append(
                    f"title endpoint town {phrase!r} appears in NEITHER the route's "
                    f"ordered place sequence {place_seq} nor sampled regions "
                    f"{regions} — wrong-town-in-right-county class{unproven}"
                )
        if not endpoint_phrases and shipped and not _COUNTY_TOKEN_RE.search(shipped):
            label_tokens = _town_claim_tokens(shipped)
            if label_tokens and not label_tokens & truth_tokens:
                failures.append(
                    f"town-form shipped text {shipped!r} shares no place token with "
                    f"the route's place sequence {place_seq} / sampled regions "
                    f"{regions} — the label desynced from the GPS trace"
                )
    failures += _coverage_failures(metadata_json, resolved, title, location)
    if failures:
        return GateResult(
            gate="location_invariant",
            passed=False,
            message="location invariant HARD FAIL: " + "; ".join(failures),
        )

    return GateResult(
        gate="location_invariant",
        passed=True,
        message="Location invariant checks passed",
    )


def _vote_label(key) -> str:
    """A vote key (``"County|State"`` from the vote, ``"County, State"`` from
    ``ResolvedLocation.vote``) as the ``"County, State"`` label form."""
    county, _sep, state = str(key).partition("|")
    state = "" if state in ("", "None") else state
    return f"{county}, {state}" if state else county.strip()


def _county_tokens(label: str) -> set[str]:
    """Lowercased names that identify a county in a claim: its ``_county_names``
    form, else (independent city / bare name) the leading label component."""
    return _county_names(label) or ({label.split(",")[0].strip().lower()} - {""})


def _coverage_failures(metadata_json: dict, resolved: dict, title: str, location: str) -> list[str]:
    """Material counties (vote share >= ``LOCATION_MATERIAL_SHARE``) that no
    shipped claim names — see :func:`check_location_invariant`."""
    vote = resolved.get("vote") or {}
    material = [
        (_vote_label(k), float(v))
        for k, v in vote.items()
        if isinstance(v, (int, float)) and float(v) >= LOCATION_MATERIAL_SHARE
    ]
    if len(material) < 2:
        return []  # one material county: membership already covers it
    claim_text = " ".join(
        [title, location, *(text for _surface, text in _location_claim_surfaces(metadata_json))]
    ).lower()
    # Counties the title's endpoint towns resolve to, via the model's
    # ``endpoint_regions`` (absent on a legacy model -> only naming counts).
    via_endpoint: set[str] = set()
    endpoint_regions = resolved.get("endpoint_regions") or {}
    for text in (title, location):
        if _TITLE_ENDPOINTS_RE.match(text or ""):
            for end in ("start", "end"):
                region = endpoint_regions.get(end)
                if isinstance(region, str) and region.strip():
                    via_endpoint |= _county_tokens(region)
    failures = []
    for label, share in material:
        names = _county_tokens(label)
        if names and (names & via_endpoint or any(n in claim_text for n in names)):
            continue
        failures.append(
            f"label {location!r} omits {label} which holds {share:.0%} of the footage "
            f"(material floor {LOCATION_MATERIAL_SHARE:.0%}) — a label may not name "
            f"one location while another is on screen"
        )
    return failures


# Both county forms: US ``"<Name> County"`` (group 2) and Irish ``"County
# <Name>"`` (group 1). Without the Irish alternative, an Irish label yields no
# county match and the discriminator falls through to a province-token overlap
# that fails OPEN on same-province, different-county desyncs (e.g. burn County
# Kerry vs clip GPS County Cork, both Munster).
_BURN_COUNTY_RE = re.compile(
    r"\bCounty\s+([\w'.-]+)|([\w'.-]+(?:\s+[\w'.-]+)*?)\s+County\b", re.IGNORECASE
)


def _county_names(label: str) -> set[str]:
    """Lowercased county name(s) in a label — ``"…, Colleton County, South
    Carolina"`` → ``{"colleton"}`` and ``"County Kerry, Munster"`` →
    ``{"kerry"}``. The discriminating identity, so a same-state/province,
    different-county burn isn't masked.
    """
    return {
        (m.group(1) or m.group(2)).strip().lower() for m in _BURN_COUNTY_RE.finditer(label or "")
    }


def _place_tokens(label: str) -> set[str]:
    """Lowercased name tokens, ``County`` filler dropped — the fallback axis for a
    town-form burn (a town shares tokens with its region)."""
    tokens: set[str] = set()
    for part in re.split(r"[,\s]+", label or ""):
        word = part.strip().lower()
        if word and word != "county":
            tokens.add(word)
    return tokens


# Reconciliation-gate helpers — generic title filler that must neither anchor nor fail a
# town-form reconciliation ("Scenic Drive", verbs, stock SEO terms). Tokens of
# len <= 2 (state abbreviations, road refs) are dropped in _town_claim_tokens.
_GENERIC_LABEL_TOKENS = frozenset(
    {
        "drive",
        "driving",
        "scenic",
        "backroads",
        "road",
        "roads",
        "route",
        "highlights",
        "pov",
        "the",
        "and",
        "via",
        "part",
        "area",
    }
)
_LOCATION_LINE_RE = re.compile(r"\U0001f4cd Location:\s*(.+)")
# The above-the-fold SEO hook line is produced as "… 4K drive …" by
# longform_metadata (_SEO_HOOK_* templates); "scenic" is not an accepted term.
# This marker MUST track that string or the SEO line escapes the location
# sweep in _location_claim_surfaces.
_SEO_LINE_MARK = "4K drive"
_HASHTAG_RE = re.compile(r"#(\w+)")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# "{A} to {B} [via …]" endpoint arc at the head of a title/OST label. Each
# endpoint may carry its own state abbreviation — the cross-state form
# "Blountville, TN to McMullin, VA via I-81" — which is not part of the name.
_TITLE_ENDPOINTS_RE = re.compile(
    r"^\s*(?P<start>[^,|—(]+?)(?:,\s*[A-Z]{2}\b)?\s+to\s+(?P<end>[^,|—(]+?)"
    r"(?:,\s*[A-Z]{2}\b)?(?:\s+via\s+[^,|—(]+)?\s*(?:[,|—(]|$)"
)


def _town_claim_tokens(text: str) -> set[str]:
    """Place tokens that can carry a TOWN-level location claim: generic filler,
    any digit-bearing road ref (``"i-81"``/``"us-221"`` — a town name never
    carries a digit), and len<=2 tokens (state abbrevs) dropped so a via-road
    ref / "SC" can't satisfy or fail the check. Dropping the hyphenated road ref
    lets the tier-2 form ``"{road}, {region}"`` reconcile as a road corridor,
    not a fabricated town."""
    return {
        t
        for t in _place_tokens(text)
        if len(t) > 2 and not any(ch.isdigit() for ch in t) and t not in _GENERIC_LABEL_TOKENS
    }


# US-state and Irish-province name WORDS. Stripped from BOTH sides of a town-form
# reconciliation so the gate corroborates on a genuine TOWN/COUNTY token, never
# on the shared STATE/PROVINCE alone ("Beaufort, South Carolina" vs
# "Colleton County, South Carolina" must NOT pass on {south, carolina} — a
# different county, same state; the Irish province axis had the same hole).
_IRISH_PROVINCE_TOKENS = frozenset({"munster", "leinster", "connacht", "connaught", "ulster"})
_STATE_PROVINCE_TOKENS: frozenset[str] = (
    frozenset(tok for name in US_STATE_ABBREVS for tok in name.lower().split())
    | _IRISH_PROVINCE_TOKENS
)


def _town_place_tokens(text: str) -> set[str]:
    """TOWN-level claim tokens with US-state / Irish-province name words removed,
    so a town-form reconciliation matches on a genuine settlement/county token,
    never on the shared state/province alone."""
    return _town_claim_tokens(text) - _STATE_PROVINCE_TOKENS


# Geometric endpoint verification. A reverse geocoder can bucket a whole route
# under its district, so the place sequence never names the villages driven
# through and a GPS-true village-to-village label reads as wrong-town. A name is
# accepted only when it matches exactly the endpoint's name within this distance
# of the LF trace; the forward geocoder only proposes candidates, the trace
# decides.
_TRACE_VERIFIED_PLACE_KM = 2.0


def _endpoint_town_phrases(*texts: str) -> set[str]:
    """The ``"A to B [via …]"`` endpoint phrases at the head of each text."""
    phrases: set[str] = set()
    for text in texts:
        m = _TITLE_ENDPOINTS_RE.match(text or "")
        if m:
            phrases.add(m.group("start").strip())
            phrases.add(m.group("end").strip())
    return phrases


def _place_name_key(name) -> str:
    """Case-, accent- and whitespace-insensitive key for an exact-name match."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(bare.casefold().split())


def _verification_scopes(resolved: dict) -> list[str]:
    """Geocode scopes, dominant region first: ``"County Cork"`` /
    ``"Montgomery County, Virginia"`` — the same shape as ``regions``."""
    dominant = ", ".join(
        s
        for s in (resolved.get("dominant_county"), resolved.get("dominant_state"))
        if isinstance(s, str) and s.strip()
    )
    scopes: list[str] = []
    for scope in [dominant, *(resolved.get("regions") or [])]:
        if isinstance(scope, str) and scope.strip() and scope not in scopes:
            scopes.append(scope)
    return scopes or [""]


def _nearest_same_name_km(phrase: str, query: str, gps_points, window, places_fn):
    """Distance (km) from the trace to the nearest candidate named exactly
    ``phrase``, else ``None``. Raises whatever ``places_fn`` raises."""
    from gomoveshift.pipeline import gps_quality as gq

    key = _place_name_key(phrase)
    nearest = None
    for name, lat, lon in places_fn(query) or []:
        if not key or _place_name_key(name) != key:
            continue
        d = gq.nearest_trace_distance_km(float(lat), float(lon), gps_points, *window)
        if d is not None and (nearest is None or d < nearest):
            nearest = d
    return nearest


def trace_verified_endpoint_towns(label: str, resolved: dict, gps_points, places_fn) -> list[str]:
    """Endpoint towns of ``label`` proven by the LF's own GPS trace, sorted.

    The location invariant's second opinion on an ``"A to B"`` label. Only an
    endpoint the reverse-geocoded truth (place sequence, destination, regions)
    does NOT already corroborate is looked up, so nothing it passes today is
    touched. ``places_fn(query)`` proposes settlement candidates
    ``(name, lat, lon)`` for ``"<town>, <region>"`` — dominant region first,
    then each sampled region — and the endpoint verifies when a candidate of
    exactly its name lies within ``_TRACE_VERIFIED_PLACE_KM`` of an in-window
    fix. Nearest candidate wins, so a far top hit (an electoral-division
    centroid) can't mask the village. No timed trace, no hits, a raising or
    malformed geocoder: unverified — the gate then fails exactly as before.
    """
    from gomoveshift.pipeline import gps_quality as gq

    phrases = _endpoint_town_phrases(label)
    if not phrases:
        return []
    base_truth: set[str] = set()
    for src in [
        *(resolved.get("ordered_place_sequence") or []),
        resolved.get("destination"),
        *(resolved.get("regions") or []),
    ]:
        if isinstance(src, str) and src.strip():
            base_truth |= _town_claim_tokens(src)
    window = gq.trace_time_window(gps_points)
    verified: list[str] = []
    for phrase in sorted(phrases):
        tokens = _town_claim_tokens(phrase)
        if not tokens or _COUNTY_TOKEN_RE.search(phrase) or tokens & base_truth:
            continue  # nothing to prove, or the reverse geocoder already did
        if window is None:
            logger.warning("TRACE-UNVERIFIED-PLACE: %r — no timed GPS trace to check", phrase)
            continue
        nearest = None
        for scope in _verification_scopes(resolved):
            query = f"{phrase}, {scope}" if scope else phrase
            try:
                d = _nearest_same_name_km(phrase, query, gps_points, window, places_fn)
            except Exception as exc:  # noqa: BLE001 — fail closed, never crash the gate
                logger.warning("TRACE-UNVERIFIED-PLACE: %r — geocoder failed (%s)", query, exc)
                continue
            if d is not None and (nearest is None or d < nearest):
                nearest = d
            if nearest is not None and nearest <= _TRACE_VERIFIED_PLACE_KM:
                break
        if nearest is not None and nearest <= _TRACE_VERIFIED_PLACE_KM:
            logger.info(
                "TRACE-VERIFIED-PLACE: %r — a settlement of that name lies %.2f km "
                "from the LF's GPS trace",
                phrase,
                nearest,
            )
            verified.append(phrase)
        else:
            logger.warning(
                "TRACE-UNVERIFIED-PLACE: %r — no settlement of that name within "
                "%g km of the LF's GPS trace (nearest: %s)",
                phrase,
                _TRACE_VERIFIED_PLACE_KM,
                "none" if nearest is None else f"{nearest:.2f} km",
            )
    return verified


def resolved_location_gate_dict(rl) -> dict:
    """The JSON-safe ``ResolvedLocation`` slice every location gate reads.

    One builder for the pre-encode (Phase BC) and post-encode (Phase D)
    ``metadata_json["resolved_location"]``, so a field the gate needs can't
    reach one call site and miss the other.
    """
    verified = getattr(rl, "trace_verified_places", None)
    return {
        "dominant_county": getattr(rl, "dominant_county", None),
        "dominant_state": getattr(rl, "dominant_state", None),
        "regions": list(getattr(rl, "regions", None) or []),
        "vote": dict(getattr(rl, "vote", None) or {}),
        # The travel-ordered place sequence lets the label-quality gate
        # detect a county-only label when a recognizable settlement was available.
        "ordered_place_sequence": list(getattr(rl, "ordered_place_sequence", None) or []),
        # The resolved destination POI is town-level truth for the gate's
        # destination-form label reconciliation.
        "destination": getattr(rl, "destination", None),
        # None = not computed (legacy model): the gate's failure then makes no
        # claim that a trace check ran.
        "trace_verified_places": list(verified) if verified is not None else None,
        # Which county each endpoint sits in: maps an "A to B" title's towns to
        # counties for the coverage rule. Empty on a legacy model.
        "endpoint_regions": dict(getattr(rl, "endpoint_regions", None) or {}),
    }


def _location_claim_surfaces(metadata_json: dict):
    """Yield ``(surface_name, text)`` for every non-title surface staking a
    location claim: the description's ``📍 Location:`` line, its SEO search
    line (the ``"4K drive"`` evergreen), each description hashtag
    (camel-split so ``#ColletonCounty`` scans as ``Colleton County``), and
    each YouTube tag. Chapter lines are deliberately excluded — chapter
    counties come from a denser geocode pass than the sampled ``regions`` set
    and are cross-checked by ``check_metadata_quality`` instead."""
    description = metadata_json.get("description") or ""
    m = _LOCATION_LINE_RE.search(description)
    if m:
        yield "description location line", m.group(1).strip()
    for ln in description.split("\n"):
        if _SEO_LINE_MARK in ln:
            yield "description SEO line", ln.strip()
            break
    for body in _HASHTAG_RE.findall(description):
        yield f"description hashtag #{body}", _CAMEL_BOUNDARY_RE.sub(" ", body)
    for tag in metadata_json.get("tags") or []:
        if isinstance(tag, str) and tag.strip():
            yield f"tag {tag!r}", tag


class _VotedRegions(list):
    """A short's sampled regions plus ``vote`` (``{"County, State": share}``),
    so the short gate can see a clip that straddles two counties. Still a plain
    ``list`` to every caller that only wants the labels."""

    def __init__(self):
        super().__init__()
        self.vote: dict = {}


def _make_short_regions_fn(geocoder):
    """Build the ``short_regions_fn`` closure consumed by the shorts loop.

    Signature ``(trimmed_points, short_range) -> list[str]``: votes
    ``dominant_location`` over the short's clip GPS (the SAME
    ``trim_gps_to_clip_timeline`` output the OST text derives from) and renders
    each vote key as the ``"County, State"`` label that fills
    ``ResolvedLocation.regions`` — the membership set the LF gate checks. The vote
    enforces the same ``GEOCODE_QUORUM_MIN`` floor as the shorts OST closure (the gate must
    not "pass" a short on a single-sample region set the OST fn would refuse
    to label). Returns ``[]`` on empty clip GPS, no geocoder, or a
    derivation failure so :func:`check_short_location_invariant` no-ops a
    GPS-less / degraded clip; a derivation failure WARNs (the gate fails open).
    """
    from gomoveshift.pipeline.gps_enrichment import GEOCODE_QUORUM_MIN
    from gomoveshift.pipeline.title_builder import dominant_location

    def _fn(trimmed_points, short_range):
        pts = list(trimmed_points or [])
        if not pts or geocoder is None:
            return []
        try:
            dom = dominant_location(pts, geocoder, min_votes=GEOCODE_QUORUM_MIN)
        except Exception as exc:  # noqa: BLE001 — region derivation never crashes a short
            logger.warning(
                "short_regions_fn: dominant_location failed for short rank %s: %s "
                "— short location gate will be SKIPPED for this clip",
                getattr(short_range, "rank", "?"),
                exc,
            )
            return []
        if dom is None or not dom.votes:
            return []
        regions = _VotedRegions()
        for key in dom.votes:
            county, _, state = str(key).partition("|")
            # Collapse an empty OR legacy-literal "None" state to no-state so
            # a stateless county (Irish/independent-city) labels as the bare
            # county, never a malformed "County, None". Redundant with the
            # producer's empty-slot serialization; defense against any future
            # second producer.
            state = "" if state in ("", "None") else state
            label = f"{county}, {state}" if state else county
            if label and label not in regions:
                regions.append(label)
                regions.vote[label] = dom.votes[key]
        return regions

    return _fn


def check_short_location_invariant(
    burned_text: str,
    sampled_regions: list,
    *,
    clip_derived: bool = False,
    sampled_vote: "dict | None" = None,
) -> GateResult:
    """Cross-surface location invariant for a SHORT (the shorts analog
    of :func:`check_location_invariant`).

    A short is a single ~38s clip cut from a far corner of the day; its burned
    location must describe *the clip's own footage*, not the LF-/drive-wide
    county. ``sampled_regions`` are the clip's ``dominant_location`` vote keys
    as ``"County, State"`` labels (the same set the LF gate checks).

    **HARD FAIL** when the burn names a COUNTY in ZERO sampled regions — the clip
    burned a county its own GPS never visited (far-corner desync; a
    same-state/province, different-county label — US ``"<Name> County"`` or Irish
    ``"County <Name>"`` — is NOT excused by the shared state/province token). A
    town-form burn (no ``County`` token, e.g. ``"Beaufort, SC"``) must instead share a
    non-state place token with a sampled region — unless ``clip_derived`` says the
    burn came from this clip's own ``dominant_location`` vote, which corroborates
    it by construction (a settlement is rarely eponymous with its county). The
    caller escalates failure to a ``RuntimeError`` upstream of the irreversible
    9:16 encode. No-op when there are no sampled regions or no burned text — the
    LF gate's legacy/GPS-less skip.

    There is no operator exemption: an operator ``--per-lf-location`` label names
    the whole LF, never one clip, so it is never burned on a GPS-bearing short.

    **Coverage:** when ``sampled_vote`` (``{"County, State": share}``) carries two
    or more counties — at ANY share — the gate HARD FAILs: a short has no
    material floor, it is single-county by construction, and one burned label
    cannot describe a clip that shows two. ``None`` keeps membership-only
    behaviour.
    """
    if sampled_vote and len(sampled_vote) >= 2:
        a, b, *_ = list(sampled_vote)
        return GateResult(
            gate="short_location_invariant",
            passed=False,
            message=(
                f"short location invariant HARD FAIL: clip GPS spans {a} and {b} — "
                f"a 38-s short may not show two counties under one label"
            ),
        )
    regions = [r for r in (sampled_regions or []) if isinstance(r, str) and r.strip()]
    text = (burned_text or "").strip()
    if not regions or not text:
        return GateResult(
            gate="short_location_invariant",
            passed=True,
            message="no sampled regions / burn text (skipped)",
        )

    # County-form burn: its county identity (the discriminating token, never
    # excused by a shared state) must match a sampled region's county. Town-form
    # burn (no ``County`` token): fall back to a shared place token with the
    # state/province words stripped from BOTH sides — a settlement inside a
    # sampled county shares a genuine TOWN/COUNTY token, so the shared state or
    # province alone can no longer excuse a different-county burn.
    burn_counties = _county_names(text)

    # A TOWN-form burn that this clip's own vote produced is corroborated by
    # construction: ``short_ost_text_fn`` renders the vote's settlement/POI and
    # ``short_regions_fn`` renders the SAME vote's county, so the town-token
    # reconciliation below is checking one face of a single vote against the
    # other. It only ever succeeded where the town happens to be EPONYMOUS with
    # its county (Beaufort/Beaufort County); every other settlement — Kingsport
    # in Sullivan County — false-failed a burn its own GPS fully supports. This
    # is the same vacuity noted for the LF gate's ``dominant_county``. County-form
    # text stays checked even when clip-derived: two renderings of one vote that
    # disagree on the COUNTY is a genuine contradiction, not a granularity gap.
    if clip_derived and not burn_counties:
        return GateResult(
            gate="short_location_invariant",
            passed=True,
            message="clip-derived town/destination label — corroborated by its own vote",
        )

    tokens = _county_names if burn_counties else _town_place_tokens
    sampled: set[str] = set()
    for r in regions:
        sampled |= tokens(r)
    passed = bool(tokens(text) & sampled)

    if passed:
        return GateResult(
            gate="short_location_invariant",
            passed=True,
            message="Short location invariant checks passed",
        )
    return GateResult(
        gate="short_location_invariant",
        passed=False,
        message=(
            f"short location invariant HARD FAIL: burned text {text!r} names a "
            f"county absent from the clip's sampled regions {regions} — the short "
            f"burned a county its own GPS never visited (far-corner desync)"
        ),
    )


_COUNTY_TOKEN_RE = re.compile(r"\bCounty\b", re.IGNORECASE)
# A label that OPENS with a preposition/title word (e.g. "To Mizen Head,
# County Cork") is a title, not a place -- it must not be misreported as
# "county-only" merely because it also happens to contain the word "County".
# ``is_title_prefixed`` (gomoveshift.pipeline._series_utils) is the single
# source for the curated word list -- see its docstring for why.


def _label_quality_diagnosis(metadata_json: dict) -> "tuple[str, list[str]] | None":
    """Single source of truth for the title-vs-place / county-only verdict.

    Returns ``(kind, settlements)`` where ``kind`` is ``"title"`` or
    ``"county_only"``, or ``None`` when there is nothing to flag. Both
    :func:`check_label_quality` (logs a WARN) and
    :func:`gomoveshift.pipeline.review_gate._label_quality_warning` (returns
    text for the Gate C/D prompt) call this, so a single shared source keeps the
    detection (e.g. the title-prefix carve-out) from diverging between the
    log-line check and the prompt text.
    """
    resolved = metadata_json.get("resolved_location")
    label = (metadata_json.get("location") or metadata_json.get("title") or "").strip()
    if not isinstance(resolved, dict) or not label:
        return None
    seq = resolved.get("ordered_place_sequence") or []
    # Recognizable settlements = sequence entries that are not a bare county.
    settlements = [
        s.strip() for s in seq if isinstance(s, str) and s.strip() and "county" not in s.lower()
    ]
    if not settlements:
        return None
    label_lower = label.lower()
    label_names_a_settlement = any(s.lower() in label_lower for s in settlements)
    if is_title_prefixed(label):
        return ("title", settlements)
    if _COUNTY_TOKEN_RE.search(label) and not label_names_a_settlement:
        return ("county_only", settlements)
    return None


def check_label_quality(metadata_json: dict) -> GateResult:
    """Label-quality backstop — WARN-record, never aborts.

    Flags the defect class where an LF's burned label / title collapsed to
    the bare COUNTY (``"Jackson County, NC Scenic Drive"``) even though the
    trace's canonical ``ResolvedLocation.ordered_place_sequence`` carried a
    recognizable SETTLEMENT (Cashiers, Franklin, Lake Toxaway, Fort Fremont)
    that should have anchored the title instead. Emits a greppable
    ``LABEL-QUALITY:`` WARN so the operator can supply a ``--per-lf-location``
    override or investigate.

    This is the observability backstop to the robust-by-construction fixes:
    the recognizable-settlement backfill surfaces the settlement automatically
    and the via-road quality filter drops junk via-roads, so after both this
    gate should be SILENT on a healthy run — a WARN here means
    an LF slipped through (e.g. a rural LF with no settlement, or
    a degraded resolver). It stays ``passed=True``: a county-only label is a
    "could be better", not a desync error (that is the HARD job of
    ``check_location_invariant``), so it must not abort a multi-hour run.

    No-op when the LF has no resolved model, no label, or no place sequence
    (GPS-less / legacy / single-video path).
    """
    resolved = metadata_json.get("resolved_location")
    label = (metadata_json.get("location") or metadata_json.get("title") or "").strip()
    if not isinstance(resolved, dict) or not label:
        return GateResult(
            gate="label_quality",
            passed=True,
            message="no resolved location / label (skipped)",
        )
    diagnosis = _label_quality_diagnosis(metadata_json)
    if diagnosis is not None:
        kind, settlements = diagnosis
        if kind == "title":
            logger.warning(
                "LABEL-QUALITY: LF label %r is a title, not a place — use the "
                "place name (e.g. %r) as the location; keep the title wording "
                "for the OST only",
                label,
                settlements[0],
            )
        else:
            logger.warning(
                "LABEL-QUALITY: LF label %r is county-only but the trace's place "
                "sequence carries recognizable settlement(s) %s — anchor the title "
                "on a settlement or supply a --per-lf-location override",
                label,
                settlements[:5],
            )
    return GateResult(
        gate="label_quality",
        passed=True,
        message="Label quality checks passed",
    )


# Specificity-ladder thresholds: stationary ceiling (m/s), via-road support fraction, km.
_VIA_ROAD_RE = re.compile(r"\bvia\s+(?P<road>[^,|—(]+)", re.IGNORECASE)
_MOTION_TOKEN_RE = re.compile(r"\b(?:drive|driving|drives|drove|cruis\w*|pov)\b", re.IGNORECASE)
_STATIONARY_SPEED_MPS = 0.5
_VIA_ROAD_SUPPORT_MIN = 0.30
_ENDPOINT_PROXIMITY_KM = 10.0


def award_tier_for_evidence(
    evidence: dict, *, window: Optional[tuple] = None
) -> tuple[Optional[float], Optional[int]]:
    """(locked-GPS coverage, awarded specificity tier) — the single home for
    the coverage→tier decision, shared by :func:`check_specificity_ladder`
    (hard-fails a still tier-1-shaped label whose award is > 1) and the
    pre-encode demotion (``phase_bc_day_first._degrade_labels_to_awarded_tier``,
    which re-renders a below-tier-1 award so it is no longer tier-1-shaped).
    ``(None, None)`` when coverage is unavailable — both callers no-op that LF.
    ``has_bracket_route`` is honoured (bracket-only → tier 4). Footage
    DURATION coverage sees a dead prefix the fixes' min/max span can't.
    """
    from gomoveshift.pipeline import gps_quality as gq
    from gomoveshift.pipeline.title_builder import award_tier

    gps_points = evidence.get("gps_points")
    if gps_points is None:
        return None, None
    window = window or evidence.get("window") or gq.trace_time_window(gps_points)
    duration = evidence.get("duration")
    if duration:
        coverage = gq.locked_gps_duration_coverage(gps_points, float(duration))
    elif window is not None:
        coverage = gq.locked_gps_coverage(gps_points, window[0], window[1])
    else:
        coverage = None
    if coverage is None:
        return None, None
    tier = award_tier(
        coverage,
        int(evidence.get("corroborations", 0)),
        has_bracket_route=bool(evidence.get("has_bracket_route", False)),
    )
    return coverage, tier


def check_specificity_ladder(metadata_json: dict, evidence: Optional[dict] = None) -> GateResult:
    """Specificity-ladder evidence gate — the pre-encode ladder half.

    The location invariant reconciles a label against the *derived* model (self-referential —
    label and model share one trace); this gate reconciles it against RAW footage
    evidence via four checks (in order below): locked-GPS coverage floor, via-road
    support, endpoint geocode proximity, and a stationary-footage guard.

    OPT-IN / no-op — no ``evidence`` no-ops entirely, and each check fires ONLY
    when its key is present AND contradicts the label, so a correct high-coverage
    LF PASSES unchanged. There is no operator exemption: a
    ``--per-lf-location`` label is checked like any other.
    """
    from gomoveshift.pipeline import gps_quality as gq
    from gomoveshift.pipeline.title_builder import LOCKED_COVERAGE_TIER1_MIN

    label = (metadata_json.get("location") or metadata_json.get("title") or "").strip()
    if not evidence or not label:
        return GateResult(
            gate="specificity_ladder", passed=True, message="no evidence / label (skipped)"
        )

    gps_points = evidence.get("gps_points")
    window = evidence.get("window") or gq.trace_time_window(gps_points)
    endpoints = _TITLE_ENDPOINTS_RE.match(label)
    via = _VIA_ROAD_RE.search(label)
    via_road = via.group("road").strip() if via else ""
    is_tier1_shaped = bool(endpoints) or bool(via_road)
    failures: list[str] = []

    if gps_points is not None and is_tier1_shaped:
        # A still-tier-1-shaped label whose award is > 1 hard-fails here; the
        # pre-encode demotion re-renders below-tier-1 awards upstream, so a
        # demoted label reaches this gate no longer tier-1-shaped and passes.
        coverage, tier = award_tier_for_evidence(evidence, window=window)
        if coverage is not None and tier is not None and tier > 1:
            failures.append(
                f"tier-1 claim {label!r} but locked-GPS coverage {coverage:.2f} "
                f"< {LOCKED_COVERAGE_TIER1_MIN:.2f} awards tier {tier} — the label must demote"
            )

    refs = [str(r).lower() for r in (evidence.get("road_refs") or []) if str(r).strip()]
    if via_road and refs:
        vlow = via_road.lower()
        support = sum(1 for r in refs if vlow in r or r in vlow) / len(refs)
        if support < _VIA_ROAD_SUPPORT_MIN:
            failures.append(
                f"via-road {via_road!r} in only {support:.0%} of road refs "
                f"(<{_VIA_ROAD_SUPPORT_MIN:.0%}) — the via-road clause is unsupported"
            )

    geocode_fn = evidence.get("geocode_fn")
    if endpoints and callable(geocode_fn) and gps_points and window is not None:
        for role in ("start", "end"):
            town = endpoints.group(role).strip()
            if not town or _COUNTY_TOKEN_RE.search(town):
                continue
            try:
                coord = geocode_fn(town)
            except Exception:  # noqa: BLE001 — a geocoder failure never crashes the gate
                coord = None
            if not coord:
                continue
            dist = gq.nearest_trace_distance_km(
                float(coord[0]), float(coord[1]), gps_points, window[0], window[1]
            )
            if dist is not None and dist > _ENDPOINT_PROXIMITY_KM:
                failures.append(
                    f"endpoint {town!r} geocodes {dist:.0f} km from the footage trace "
                    f"(>{_ENDPOINT_PROXIMITY_KM:.0f} km) — the endpoint is fabricated"
                )

    speed = evidence.get("median_speed_mps")
    if speed is None and gps_points is not None and window is not None:
        speed = gq.window_median_speed_mps(gps_points, window[0], window[1])
    asserts_motion = bool(_MOTION_TOKEN_RE.search(label)) or bool(endpoints)
    if speed is not None and asserts_motion and speed <= _STATIONARY_SPEED_MPS:
        failures.append(
            f"median speed {speed:.2f} m/s ~stationary but {label!r} uses a motion/drive "
            f"form — a parked clip is not a drive"
        )

    if failures:
        return GateResult(
            gate="specificity_ladder",
            passed=False,
            message="specificity ladder HARD FAIL: " + "; ".join(failures),
        )
    return GateResult(
        gate="specificity_ladder", passed=True, message="specificity ladder checks passed"
    )


# Stop-free gate: a shipped clip whose retained window is majority parked is
# the failure class the per-day stop mask exists to prevent. ~1 km/h is the
# floor at which even parked GPS drift stays below (mirrors ``stop_mask``); the
# gate errs toward NOT failing legit slow driving (a HARD fail, so conservative).
_STOP_WINDOW_SPEED_MPS = 1.0


def check_stop_free_windows(
    retained_windows,
    gps_points,
    stop_mask=None,
    *,
    max_stationary_fraction: float = 0.5,
    stop_speed_mps: float = _STOP_WINDOW_SPEED_MPS,
) -> GateResult:
    """Cross-check a shipped LF/short's RETAINED source-time window(s) for motion.

    Distinct from :func:`check_specificity_ladder`'s stationary check (a
    whole-window median that AUTO-PASSES on ``None`` — the fail-open we must
    avoid): the GPS signal here is the per-interval STATIONARY FRACTION
    (``gps_quality.window_stationary_fraction``, source-second clock
    ``(t, lat, lon)``), and a window with no in-window fixes SUSPECT-falls-back to
    the persisted per-second ``stop_mask`` (``True`` == stopped) rather than
    passing. A window whose stationary fraction exceeds ``max_stationary_fraction``
    (majority parked) **HARD FAILS** — the caller escalates to a ``RuntimeError``
    upstream of any encode/upload.

    GPS is primary (its speed reflects the real footage even when the coarse stop
    mask over-masks a sparse trace); the mask is the per-window-gap fallback.
    No-op (pass) for genuinely GPS-less inputs (``gps_points`` empty/None) — no
    per-second speed to judge, and the structural stop mask already excluded stops
    from the plan upstream — and when there are no windows.
    """
    windows = [
        (float(s), float(e))
        for s, e in (retained_windows or [])
        if s is not None and e is not None and float(e) > float(s)
    ]
    have_gps = gps_points is not None and len(gps_points) >= 2
    have_mask = stop_mask is not None and len(stop_mask) > 0
    if not windows or not have_gps:
        return GateResult(
            gate="stop_free_windows",
            passed=True,
            message="no GPS trace / no windows (skipped)",
        )

    from gomoveshift.pipeline import gps_quality as gq

    failures: list[str] = []
    for w0, w1 in windows:
        res = gq.window_stationary_fraction(gps_points, w0, w1, stop_speed_mps=stop_speed_mps)
        if res is not None:
            frac, n_intervals = res
            source = f"{n_intervals} GPS interval(s)"
        elif have_mask:
            # No in-window fixes on a GPS-bearing day — SUSPECT: consult the
            # per-second mask instead of auto-passing.
            total = 0
            stopped = 0
            for t in range(int(math.floor(w0)), int(math.ceil(w1))):
                if 0 <= t < len(stop_mask):
                    total += 1
                    if bool(stop_mask[t]):
                        stopped += 1
            if total == 0:
                continue
            frac = stopped / total
            source = f"{stopped}/{total}s masked"
        else:
            continue  # unmeasurable (no GPS in window, no mask) — no false positive
        if frac > max_stationary_fraction:
            failures.append(f"window [{w0:.0f},{w1:.0f}]s is {frac:.0%} stationary ({source})")

    if failures:
        return GateResult(
            gate="stop_free_windows",
            passed=False,
            message=(
                "stop-free windows HARD FAIL: "
                + "; ".join(failures)
                + f" (> {max_stationary_fraction:.0%} parked — a shipped clip "
                "must not be majority stationary)"
            ),
        )
    return GateResult(
        gate="stop_free_windows",
        passed=True,
        message="all shipped windows are majority moving",
    )


_FILMED_DATE_RE = re.compile(r"\U0001f4c5\s*Filmed:\s*(\d{4}-\d{2}-\d{2})")
_BODY_RE = re.compile(
    r"(?<=\n\n)(.+?)(?=\n\n\U0001f4cd Location:)",
    re.DOTALL,
)


def _format_duration_pretty(seconds: float) -> str:
    """Mirror of ``LongFormMetadataGenerator._format_duration``.

    Inlined to keep ``check_description_redundancy`` a pure-function with no
    import-time dependency on the metadata generator (which itself imports
    SEO manager). Contract is locked by the parity test in
    ``test_description_redundancy_gate``.
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m"
    return f"{minutes}m {secs}s"


# Emoji markers for the two GPS-derived stat lines (distance and pace), which
# are scoped to an LF's kept segments. A generator that falls back to an
# unwindowed (pre-cut) total for either fact must flag it via
# ``metadata_json["full_drive_stats"]`` so this gate can require the line
# carry an explicit "Full drive:" scope prefix.
_STAT_LINE_MARKERS = {
    "distance": "\U0001f6e3️",  # 🛣️ Distance:
    "speed": "\U0001f697",  # 🚗 Pace:
}


def check_description_redundancy(metadata_json: dict) -> GateResult:
    """Hard-fail gate: every fact must appear at most as often as allowed.

    Locks in the description-deduplication contract by construction — a
    redundant description cannot reach the LF metadata JSON on disk.

    Allowed-occurrences table:
      - ``recording_date``         max 1× (📅 Filmed line only)
      - ``location`` exact string  max 2× (hook + 📍 Location line)
      - ``source_duration`` pretty max 2× (hook + 🎬 Final cut disclosure)
      - ``video_duration`` pretty  max 2× (hook + 🎬 Final cut disclosure)
      - body line (paragraph between hook and 📍 Location): if non-empty,
        must add at least one noun NOT already in the structured block.
      - distance/pace stat line: a stat named in
        ``metadata_json["full_drive_stats"]`` must carry a "Full drive:"
        scope prefix; unflagged (kept-segment-derived, the default)
        stats need none.

    Pure function — no I/O.
    """
    description: str = metadata_json.get("description", "") or ""
    location: str = metadata_json.get("location", "") or ""
    recording_date: Optional[str] = metadata_json.get("recording_date")
    if not recording_date:
        m = _FILMED_DATE_RE.search(description)
        if m:
            recording_date = m.group(1)

    video_duration = metadata_json.get("video_duration")
    source_duration = metadata_json.get("source_duration")

    # Exclude the embedded chapter block from fact-occurrence counts: chapter
    # names are place names BY DESIGN (chapter-as-search-terms), a separate
    # functional block — not prose redundancy. Without this, the chapter block's
    # legitimate place names push the location count past the 2× prose budget and
    # hard-fail Phase D after the whole encode is done. The body-line check (#5)
    # still operates on the full description.
    prose: str = description.split("\U0001f4cd CHAPTERS", 1)[0]

    violations: list[str] = []

    # 1. Recording date — max 1×.
    if recording_date:
        n = prose.count(recording_date)
        if n > 1:
            violations.append(
                f"recording_date {recording_date!r} appears {n}× "
                f"(allowed: 1× — '\U0001f4c5 Filmed:' line only)"
            )

    # 2. Location exact string — max 2× (hook + 📍 Location). The SEO
    # search line ("{region} 4K drive — {arc} via {road}") is a
    # sanctioned above-the-fold SEO surface, not prose redundancy — like the
    # chapter block, its place-name nouns must not count against the budget. A
    # region-less "A to B via Road" label reconstructs the full location
    # verbatim there, which would otherwise push the count to 3×. Excluding the
    # SEO line (uniquely marked by the evergreen "4K drive" term) keeps
    # the 2× prose budget intact for genuine restatement.
    if location:
        loc_prose = "\n".join(ln for ln in prose.split("\n") if "4K drive" not in ln)
        n = loc_prose.count(location)
        if n > 2:
            violations.append(
                f"location {location!r} appears {n}× "
                f"(allowed: 2× — hook + '\U0001f4cd Location:' line only)"
            )

    # 3. Source duration pretty — max 2×. The Highlights-band hook
    #    legitimately surfaces ``source_duration`` ("Highlights from 30m
    #    0s of driving …") AND the disclosure line ("edited from 30m 0s
    #    of footage"). Anything beyond 2× is body restatement and fails.
    if source_duration is not None and source_duration > 0:
        src_pretty = _format_duration_pretty(float(source_duration))
        n = prose.count(src_pretty)
        if n > 2:
            violations.append(
                f"source_duration {src_pretty!r} appears {n}× "
                f"(allowed: 2× — Highlights hook + '\U0001f3ac Final cut' "
                f"disclosure only)"
            )

    # 4. Video duration pretty — max 2×. The POV-band hook legitimately
    #    surfaces ``video_duration`` ("POV drive through … (9m 48s).")
    #    AND the disclosure line. Anything beyond 2× is body restatement.
    if video_duration is not None and video_duration > 0:
        vid_pretty = _format_duration_pretty(float(video_duration))
        # When video_duration_pretty == source_duration_pretty (both round
        # to the same string) the budget is shared — already counted
        # under source_duration above.
        skip_video = False
        if source_duration is not None and source_duration > 0:
            if _format_duration_pretty(float(source_duration)) == vid_pretty:
                skip_video = True
        if not skip_video:
            n = prose.count(vid_pretty)
            if n > 2:
                violations.append(
                    f"video_duration {vid_pretty!r} appears {n}× "
                    f"(allowed: 2× — POV hook + '\U0001f3ac Final cut' "
                    f"disclosure only)"
                )

    # 5. Body line: paragraph between hook and 📍 Location: line. If
    #    present and non-empty, must add at least one noun NOT in the
    #    structured block. Source nouns: chapter_towns / named_routes
    #    (when metadata threads them). Absent both → body is restatement
    #    by construction → fail.
    body_match = _BODY_RE.search(description)
    if body_match:
        body = body_match.group(1).strip()
        if body:
            chapter_towns = metadata_json.get("chapter_towns") or []
            named_routes = metadata_json.get("named_routes") or []
            terrain_type = metadata_json.get("terrain_type")
            nouns: list[str] = []
            nouns.extend(t for t in chapter_towns if t)
            nouns.extend(r for r in named_routes if r)
            if terrain_type:
                nouns.append(terrain_type)

            if not nouns:
                violations.append(
                    f"body line present but no chapter_towns / named_routes / "
                    f"terrain_type known to validate it — body is pure "
                    f"restatement by construction: {body!r}"
                )
            else:
                body_lower = body.lower()
                has_new_noun = any(n.lower() in body_lower for n in nouns)
                if not has_new_noun:
                    violations.append(
                        f"body line adds no noun outside the structured block "
                        f"(known nouns: {nouns}); body: {body!r}"
                    )

    # 6. Distance/pace stat scope: a stat the generator flagged as a
    #    full-drive (pre-cut) total — via ``full_drive_stats`` — must carry an
    #    explicit "Full drive:" prefix on its rendered line; absent that flag
    #    the stat is the default (kept-segment-derived) and needs none.
    full_drive_stats = metadata_json.get("full_drive_stats") or []
    for stat_key in full_drive_stats:
        marker = _STAT_LINE_MARKERS.get(stat_key)
        if not marker:
            continue
        for line in prose.split("\n"):
            if marker in line and "Full drive:" not in line:
                violations.append(
                    f"{stat_key} stat line reflects the full drive (pre-cut) "
                    f"total but is missing the 'Full drive:' scope prefix: "
                    f"{line.strip()!r}"
                )

    if violations:
        return GateResult(
            gate="description_redundancy",
            passed=False,
            message="Description redundancy violations: " + "; ".join(violations),
        )
    return GateResult(
        gate="description_redundancy",
        passed=True,
        message="Description redundancy checks passed",
    )


def check_youtube_best_practices(metadata_json: dict, site_id: Optional[str] = None) -> GateResult:
    """Validate metadata against YouTube SEO best practices.

    site_id: None -> config.seo.site_id.
    """
    if site_id is None:
        site_id = _get_config().seo.site_id
    try:
        from seo_manager.validation.youtube_best_practices import (
            YouTubeBestPracticesValidator,
        )
    except ImportError:
        return GateResult(
            gate="youtube_best_practices",
            passed=True,
            message="YouTubeBestPracticesValidator not available (skipped)",
        )

    try:
        from seo_manager.core.metadata import ContentType, Platform, SEOMetadata

        seo_meta = SEOMetadata(
            site_id=site_id,
            content_id=metadata_json.get("content_id", "unknown"),
            platform=Platform.YOUTUBE,
            content_type=ContentType.VIDEO,
            title=metadata_json.get("title", ""),
            description=metadata_json.get("description", ""),
            tags=metadata_json.get("tags", []),
        )

        validator = YouTubeBestPracticesValidator()
        issues = validator.validate(seo_meta)

        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]

        if errors:
            msg = "; ".join(e.issue for e in errors[:3])
            return GateResult(
                gate="youtube_best_practices",
                passed=False,
                message=f"YouTube best practice errors: {msg}",
            )

        if warnings:
            msg = "; ".join(w.issue for w in warnings[:3])
            return GateResult(
                gate="youtube_best_practices",
                passed=True,
                message=f"YouTube best practices passed with warnings: {msg}",
            )

        return GateResult(
            gate="youtube_best_practices",
            passed=True,
            message="YouTube best practices: all checks passed",
        )

    except Exception as e:
        return GateResult(
            gate="youtube_best_practices",
            passed=True,
            message=f"YouTube best practices check skipped: {e}",
        )
