"""Text-shaping helpers shared by the publish plan and social generator.

Moved out of ``pipeline/csv_generator.py`` (the retired shorts CSV
writer) — pure text formatting with no CSV coupling.
``csv_generator.py`` keeps re-importing these under its old private/public
names until its writer is deleted.
"""

from __future__ import annotations

import re
from datetime import datetime


def format_recording_date_pretty(date_str: str) -> str:
    """Format an ISO ``YYYY-MM-DD`` date into ``"<MonthName> YYYY"``.

    Single source of truth for the ``"Filmed <Month YYYY>."`` suffix shown
    on every published short. Pure helper — any surface that wants the same
    prose form imports this rather than re-deriving.

    Raises ``ValueError`` on malformed input (no silent fallback —
    construction integrity for the recording-date enforcement contract).
    """
    if not isinstance(date_str, str) or not date_str.strip():
        raise ValueError(f"recording date must be a non-empty YYYY-MM-DD string; got {date_str!r}")
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").strftime("%B %Y")


# Twitter/X hard char-limit. Used only when the row actually targets Twitter.
TWITTER_CHAR_LIMIT = 280

# Safe upper bound for non-Twitter surfaces. YouTube descriptions cap at
# 5000; Instagram captions at 2200. We don't truncate at all when Twitter is
# off, but expose the constant for future per-platform shaping.
NON_TWITTER_CHAR_LIMIT = 5000


def truncate_for_platforms(
    text: str, *, twitter_enabled: bool, limit: int = TWITTER_CHAR_LIMIT
) -> str:
    """Return ``text`` shaped for the platforms this caption targets.

    A shared caption surface can target multiple platforms with different
    length caps; per-platform overrides are handled by the caller. So:

    * If ``twitter_enabled`` is False → return ``text`` unchanged. Truncating
      to Twitter's 280 char limit would clip the trailing emoji + hashtag
      block on YouTube/Instagram for no reason. (Truncating there once
      cost every long-form caption its hashtag block.)
    * If ``twitter_enabled`` is True → truncate to ``TWITTER_CHAR_LIMIT``,
      cutting at the last whitespace boundary before the limit (NEVER mid-
      word, NEVER mid-emoji-grapheme), and append ``…`` so the operator can
      see a cut happened.

    The word-boundary cut also dodges the "lone regional-indicator codepoint"
    failure mode where a naive ``text[:280]`` would split a flag emoji
    (two regional-indicator codepoints) into one orphan glyph.
    """
    if not text:
        return text
    if not twitter_enabled:
        return text
    if len(text) <= limit:
        return text
    # Reserve 1 char for the ellipsis.
    head = text[: limit - 1]
    cut = head.rsplit(" ", 1)[0] if " " in head else head
    cut = cut.rstrip()
    return cut + "…"


# A hashtag is "#" + a letter-initial word, so a number sign ("Stop #2",
# "Drive #1") is never read as one.
_HASHTAG_RE = re.compile(r"(?<!\w)#[^\W\d_]\w*")
_TRAILING_HASHTAGS_RE = re.compile(r"(?:(?:^|\s+)#[^\W\d_]\w*)+\s*$")


def split_trailing_hashtags(text: str) -> "tuple[str, str]":
    """``(body, hashtags)``: the caption's closing run of hashtags,
    space-joined, and the right-stripped text before it. ``hashtags`` is ""
    when the text does not end in one."""
    match = _TRAILING_HASHTAGS_RE.search(text)
    if not match:
        return text.rstrip(), ""
    return text[: match.start()].rstrip(), " ".join(match.group(0).split())


#: Endings that leave a sentence open: a word, a number, or a closing
#: bracket or quote after one. Anything else (".", "?", "!", "…", an emoji)
#: already ends it.
_OPEN_SENTENCE_TAILS = ")]\"'”’"


def close_sentence(text: str) -> str:
    """``text`` with a full stop when its last sentence is still open, so a
    sentence appended after it starts its own instead of running on ("Just
    driving through Asheville Filmed March 2026.").
    A closing run of hashtags is not a sentence and is left alone."""
    if not text or _TRAILING_HASHTAGS_RE.search(text):
        return text
    if text[-1].isalnum() or text[-1] in _OPEN_SENTENCE_TAILS:
        return text + "."
    return text


def hashtags_in(text: str) -> "list[str]":
    """Every hashtag in ``text`` in order, each only at its first
    (case-insensitive) occurrence."""
    seen: "set[str]" = set()
    tags: "list[str]" = []
    for tag in _HASHTAG_RE.findall(text):
        if tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)
    return tags
