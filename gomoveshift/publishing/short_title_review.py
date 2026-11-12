"""What a Short's generated title may say that the final-text review can back.

Waypost's review refuses every word of a title that no cited evidence backs and
that is not filler. A Short's title is not free prose: ``social_generator`` makes
it from a stamp ("Feb 24, 2nd stretch"), one of a few bodies and closing
hashtags. This module backs those template words from the record itself, so the
generator's own output is reviewable and nothing else is:

* the stamp only when it is exactly the stamp ``recording_date`` produces;
* a weekday only when it is ``recording_date``'s weekday;
* "seconds" only in the generator's two second-bearing bodies;
* a hashtag only when it is one of the generator's own tag lists, or a name of
  the clip's own region votes (the same rule the Short's tags follow).

Any other word stays an unbacked name and the Short is refused. The backed words
travel as the review's ``text_policy`` (``short_text_policy``), which is also what
``check_final_text`` verifies against, so there is no bypass flag and no second
policy to drift.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable

from ..pipeline.final_text_record import GMS_TEXT_POLICY
from ..pipeline.social_generator import (
    GENERIC_HASHTAGS,
    TERRAIN_HASHTAGS,
    title_stamp,
    weekday_name,
)
from ..pipeline.text_format import split_trailing_hashtags

_STAMPED = re.compile(r"(?P<stamp>[A-Z][a-z]{2} \d{1,2}, (?P<n>\d+)[a-z]{2} stretch): (?P<body>.+)")
_SECONDS_BODY = re.compile(r"\d+ seconds(?:, | on a (?P<weekday>[A-Z][a-z]+)$)")
_WEEKDAY_BODY = re.compile(r".*, on a (?P<weekday>[A-Z][a-z]+)$")
_WORD = re.compile(r"[^\W_]+")


def _words(text: str) -> set:
    return {w.casefold() for w in _WORD.findall(text)}


def _squash(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum())


def _clip_hashtag_names(regions: Iterable[str]) -> set:
    """``polkcounty``, ``northcarolina``: the squashed components of each region vote."""
    names = set()
    for label in regions:
        for part in str(label).split(","):
            if part.strip():
                names.add(_squash(part).casefold())
    return names


def short_title_filler(title: str, recording_date: str, regions: Iterable[str]) -> tuple:
    """Words of ``title`` that the record backs and so are not names to cite."""
    body, hashtags = split_trailing_hashtags(title or "")
    backed: set = set()
    allowed_tags = {t.casefold().lstrip("#") for t in (*GENERIC_HASHTAGS,)}
    for tags in TERRAIN_HASHTAGS.values():
        allowed_tags.update(t.casefold().lstrip("#") for t in tags)
    allowed_tags |= _clip_hashtag_names(regions)
    for tag in hashtags.split():
        name = tag.lstrip("#").casefold()
        if name in allowed_tags:
            backed.add(name)
    match = _STAMPED.fullmatch(body)
    if match is None:
        return tuple(sorted(backed))
    try:
        stamp_ok = title_stamp(recording_date, int(match["n"])) == match["stamp"]
    except ValueError:
        return tuple(sorted(backed))
    if not stamp_ok:
        return tuple(sorted(backed))
    backed |= _words(match["stamp"])
    weekday = weekday_name(recording_date)
    rest = match["body"]
    seconds = _SECONDS_BODY.match(rest)
    if seconds is not None:
        backed.add("seconds")
        if seconds["weekday"] == weekday:
            backed |= {"on", "a", weekday.casefold()}
    else:
        day = _WEEKDAY_BODY.fullmatch(rest)
        if day is not None and day["weekday"] == weekday:
            backed |= {"on", "a", weekday.casefold()}
    return tuple(sorted(backed))


def short_text_policy(record):
    """GMS's policy with the words ``record``'s own title has the record to back."""
    extra = short_title_filler(
        record.payloads.youtube.title, record.recording_date, record.sampled_regions
    )
    return replace(GMS_TEXT_POLICY, filler_words=(*GMS_TEXT_POLICY.filler_words, *extra))
