"""Single canonical region / US-ness derivation.

The ONE place country / US-ness / units / playlist region is decided. Both
``longform_metadata`` (units) and ``playlist_resolver`` (playlist) call
``derive_region`` / ``is_us_region``; the duplicate ``_derive_country`` parses
they each carried are deleted. Reads ``is_us``
/ ``country_code`` off the canonical ``ResolvedLocation`` when supplied; the
label parse is the back-compat fallback only. Returns ``"usa"`` / ``"ireland"``
/ … or ``""`` (resolver is total: ``""`` → default).
"""

from __future__ import annotations

import re
from typing import Any, Optional

from ._series_utils import _strip_series_suffix

# Full-word US country aliases. A trailing US *state* is handled separately via
# ``title_builder.is_us_state`` (county labels carry no country).
_US_COUNTRY_ALIASES = frozenset(
    {"usa", "us", "u.s.", "u.s.a.", "united states", "united states of america"}
)

# Irish county-form tail: ``"County Kerry"`` — leading "County" + a single
# alphabetic name (every Irish county is one word). The word-only restriction
# keeps a stray ``"County Road 12"`` tail from routing to Ireland.
_IRISH_COUNTY_TAIL_RE = re.compile(r"county\s+[a-z]+")

# The 32 traditional counties of Ireland (lowercase, bare token — no
# "County" prefix). A label whose last comma component is a BARE county
# name — e.g. ``"Tralee to Killarney, Kerry"`` — must canonicalize to
# ``"ireland"`` the same as the "County <X>" form, or it silently leaks past
# the playlist region rule into the "All Drives" fallback.
_IRISH_COUNTIES = frozenset(
    {
        "antrim",
        "armagh",
        "carlow",
        "cavan",
        "clare",
        "cork",
        "donegal",
        "down",
        "dublin",
        "fermanagh",
        "galway",
        "kerry",
        "kildare",
        "kilkenny",
        "laois",
        "leitrim",
        "limerick",
        "londonderry",
        "longford",
        "louth",
        "mayo",
        "meath",
        "monaghan",
        "offaly",
        "roscommon",
        "sligo",
        "tipperary",
        "tyrone",
        "waterford",
        "westmeath",
        "wexford",
        "wicklow",
    }
)

# Country-code / name aliases for Ireland, mirroring ``_US_COUNTRY_ALIASES``
# for the ``resolved.country_code`` passthrough in ``derive_region``.
_IRELAND_COUNTRY_ALIASES = frozenset({"ie", "irl", "ireland", "éire", "eire"})


def _region_from_label(location: str) -> str:
    """Parse the region key from the display label (back-compat fallback).

    The trailing ``(road)`` / ``(Part N)`` disambiguator is stripped
    before splitting — ``"…, South Carolina (Blocker Run Road)"`` must
    resolve via the state token, never the garbage compound. A tail that is
    itself a county designation canonicalizes to the country: Irish
    ``"County <X>"`` ⇒ ``"ireland"``, US ``"<X> County"`` ⇒ ``"usa"`` —
    matching the region keys used by the playlist rules / footer config.
    """
    if not location:
        return ""
    from .title_builder import _strip_paren_suffix, is_us_state

    base = _strip_paren_suffix(_strip_series_suffix(location))
    parts = [p.strip() for p in base.split(",") if p.strip()]
    if not parts:
        return ""
    last = parts[-1]
    if last.lower() in _US_COUNTRY_ALIASES:
        return "usa"
    if is_us_state(last):  # trailing US state ⇒ usa
        return "usa"
    low = last.lower()
    if _IRISH_COUNTY_TAIL_RE.fullmatch(low):
        return "ireland"
    if low.endswith(" county"):
        return "usa"
    if low in _IRISH_COUNTIES:
        return "ireland"
    return low


def derive_region(location: Optional[str], resolved: Optional[Any] = None) -> str:
    """Lowercase region key for ``location``.

    A ``resolved`` model with a usable signal (``is_us`` True or non-empty
    ``country_code``) is authoritative — the label is NOT re-parsed (one
    decision, not two). Otherwise parse the label. Returns ``"usa"``, a
    lowercased country/state token, or ``""``.
    """
    if resolved is not None:
        if getattr(resolved, "is_us", False):
            return "usa"
        cc = getattr(resolved, "country_code", None)
        if isinstance(cc, str) and cc.strip():
            cc = cc.strip().lower()
            if cc in _US_COUNTRY_ALIASES:
                return "usa"
            if cc in _IRELAND_COUNTRY_ALIASES:
                return "ireland"
            return cc
    return _region_from_label(location or "")


def is_us_region(location: Optional[str], resolved: Optional[Any] = None) -> bool:
    """True when the LF resolves to the United States (miles/feet, US playlist)."""
    return derive_region(location, resolved=resolved) == "usa"


# --- Free-form short titles (reconciler) ------------------------------
# A short's title is a template phrase around a place ("Dingle to Conor Pass
# — Raw Dashcam", "Colleton County Vibes"), with no structured trailing
# country. ``derive_short_title_region`` reads ONLY explicit evidence and
# returns ``None`` otherwise, so the caller never has to guess a default.

_IRELAND_WORD_RE = re.compile(r"\b(?:ireland|irish|éire|eire)\b|🇮🇪", re.IGNORECASE)
_IRISH_COUNTY_ALT = "|".join(sorted(_IRISH_COUNTIES))
# "County <Irish county>": the Irish word order (the US form is "<X> County").
# "County Down" needs a capital D: lowercase "county down south" is English.
_IRISH_COUNTY_ALT_NO_DOWN = "|".join(sorted(c for c in _IRISH_COUNTIES if c != "down"))
_COUNTY_FORM_RE = re.compile(rf"\b[Cc]ounty\s+(?:(?i:{_IRISH_COUNTY_ALT_NO_DOWN})|Down)\b")
# Lead word / ``<X>`` of "<X> County": any capitalised token, so McDowell, DeKalb
# and WYTHE read like Wythe (a Title-case-only pattern let them fall through to
# "strong Ireland").
_LEADING_WORD_RE = re.compile(r"\b([A-Z][A-Za-z'’.]*)\s+$")
_US_WORD_RE = re.compile(r"\b(?:usa|united states)\b|🇺🇸", re.IGNORECASE)
_X_COUNTY_RE = re.compile(r"\b([A-Z][A-Za-z'’.]*)\s+County\b")
# "<Irish county> County" is the US word order (Clare, Antrim and Wexford are
# Michigan counties): ambiguous, so the span counts for neither country.
_IRISH_X_COUNTY_RE = re.compile(rf"\b(?i:{_IRISH_COUNTY_ALT})\s+County\b")
_X_COUNTY_NOT_A_PLACE = frozenset(
    {"The", "This", "That", "Which", "Your", "My", "Our", "Any"}
    | {"In", "Through", "Around", "Across", "Near", "Toward", "Towards", "Into"}
    | {"From", "To", "Of", "On", "At", "By", "Via", "Visit", "Explore", "Exploring"}
)
_US_TOWNSHIP_RE = re.compile(r"\b[A-Z][a-z]+\s+Township\b")
_US_INTERSTATE_RE = re.compile(r"\bI-\d{1,3}\b")
# "County Down" is an Irish county but "Down" is a common word: bare "Down"
# never signals Ireland ("Rolling Down the Roads"), only "County Down" does.
# Counties that are also towns in US states (Dublin VA,
# Tyrone/Limerick PA, Waterford VA, Wexford PA, Antrim NH, Mayo SC) are likewise
# never Irish on their own; they still count through "County X", "Ireland" or 🇮🇪.
_BARE_AMBIGUOUS_COUNTIES = frozenset(
    {"down", "dublin", "tyrone", "limerick", "waterford", "wexford", "antrim", "mayo"}
)
_BARE_IRISH_COUNTY_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(c.title() for c in sorted(_IRISH_COUNTIES - _BARE_AMBIGUOUS_COUNTIES))
)
# USPS codes that are also English words/names: only trusted behind a comma,
# pipe, hash or paren ("Highlights, NC" / "#VA"), never standing alone.
_AMBIGUOUS_STATE_CODES = frozenset(
    {"IN", "OR", "ME", "OK", "HI", "OH", "ID", "LA", "PA", "MA", "DE", "NE", "CO", "AL", "MD"}
)
_PREFIXED_STATE_CODE_RE = re.compile(r"(?:[,|#(]\s*|\s#)([A-Z]{2})\b")
_BARE_STATE_CODE_RE = re.compile(r"(?<![\w#])([A-Z]{2})(?![\w])")

# Seed gazetteer of distinctive Irish places that routinely appear WITHOUT the
# word Ireland or a county (every entry is a distinctive Irish place and
# none is a US state/county name or a well-known US town:
# Derry and Emerald Isle are American places too, so they are NOT here. Some
# entries do exist as small US towns (Killarney FL,
# Kenmare ND, Dingle ID, Malin OR) and are accepted only because a title with
# any US signal never reaches this list). A MISS is safe: the reconciler then
# falls back to the short's live Shorts membership and finally reports it
# unrouted. Add a place here (or a videoId to the overrides file) rather than
# loosening the matching rules.
_IRISH_PLACES = (
    "adrigole", "annascaul", "ballingeary", "ballylickey", "ballynahulla", "beara",
    "black valley", "caha pass", "carrickart", "carrigart", "clonakilty", "conor pass",
    "connemara", "creeslough", "cummeenduff", "dingle", "downings", "dunlewey",
    "dursey", "durrus", "errigal", "fintown", "gap of dunloe",
    "glengarriff", "glenties", "glenveagh", "gortmore", "healy pass", "horn head",
    "inch island", "kenmare", "kilgarvan", "killarney", "killary", "killorglin", "lough",
    "malin", "mizen head", "molls gap", "murder hole", "river swilly", "sheepshead",
    "sheephaven", "singing pub", "sperrin", "stradbally", "tralee", "wild atlantic way",
)  # fmt: skip
_IRISH_PLACE_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(re.escape(p) for p in _IRISH_PLACES), re.IGNORECASE
)


def _has_us_signal(title: str) -> bool:
    from .title_builder import _US_STATE_ABBR, _US_STATE_ABBR_VALUES

    if _US_WORD_RE.search(title) or _US_TOWNSHIP_RE.search(title):
        return True
    if _US_INTERSTATE_RE.search(title):
        return True
    low = title.lower()
    if any(re.search(rf"\b{re.escape(state)}\b", low) for state in _US_STATE_ABBR):
        return True
    for match in _PREFIXED_STATE_CODE_RE.finditer(title):
        if match.group(1) in _US_STATE_ABBR_VALUES:
            return True
    for match in _BARE_STATE_CODE_RE.finditer(title):
        code = match.group(1)
        if code in _US_STATE_ABBR_VALUES and code not in _AMBIGUOUS_STATE_CODES:
            return True
    for match in _X_COUNTY_RE.finditer(title):
        word = match.group(1)
        if word not in _X_COUNTY_NOT_A_PLACE and word.lower() not in _IRISH_COUNTIES:
            return True
    return False


def _split_county_forms(title: str) -> tuple[str, bool]:
    """``(title with every County-form span blanked, strong Irish County form found)``.

    A ``County <Irish county>`` span preceded by a capitalised word is also the
    tail of a US ``<X> County`` ("Wythe County Down the Back Roads"): that
    overlap is ambiguous, so the span is blanked and counts for neither
    country. Only a span with no such lead word ("Road Trip Through County
    Kerry", "County Kerry Vibes") is strong Irish evidence.
    """
    strong = False
    for match in _COUNTY_FORM_RE.finditer(title):
        lead = _LEADING_WORD_RE.search(title[: match.start()])
        if not lead or lead.group(1) in _X_COUNTY_NOT_A_PLACE:
            strong = True
    return _COUNTY_FORM_RE.sub(" ", title), strong


def derive_short_title_region(title: str) -> Optional[str]:
    """``"ireland"`` / ``"usa"`` off a short's free-form title, else ``None``.

    Strong evidence: Ireland — the word Ireland/Irish, the 🇮🇪 flag, or
    ``County <Irish county>``; USA — a state name or USPS code, ``<X> County``
    (the US word order), the 🇺🇸 flag. Strong evidence for BOTH countries is a
    contradiction and returns ``None`` (never guessed). Weaker Irish evidence
    — a bare Irish county name or a seed-gazetteer place — counts only when the
    title carries no US signal at all, so "Clare, Michigan" and "Beaufort
    County" stay American. No signal returns ``None``; the caller owns the
    fallback and must not default one.
    """
    if not title:
        return None
    title, county_form = _split_county_forms(title)
    title = _IRISH_X_COUNTY_RE.sub(" ", title)
    strong_ireland = bool(_IRELAND_WORD_RE.search(title)) or county_form
    us = _has_us_signal(title)
    if strong_ireland and us:
        return None
    if strong_ireland:
        return "ireland"
    if us:
        return "usa"
    if _BARE_IRISH_COUNTY_RE.search(title) or _IRISH_PLACE_RE.search(title):
        return "ireland"
    return None
