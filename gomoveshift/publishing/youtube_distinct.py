"""No two YouTube posts on one channel carry near-identical titles or descriptions.

2026-08: the channel was likely suspended under YouTube's repetitive-content
rules. Runs of Shorts shared one title ("Virginia to Pennsylvania"), and every
Short of a drive that named one place sent the same reviewed description.

The rule, :func:`near_identical`. Each text is cut to its fact words: accents
folded, case, punctuation, emoji, links and closing hashtags dropped, then the
filler words that state nothing (Waypost's filler -- "to", "drive", "4k",
"shorts" -- and the generator's "pov", "driving", "first-person"). Two texts are
near-identical when their fact words are the same, or when both have at least
:data:`MIN_RATIO_WORDS` and a word-level ``difflib`` ratio of :data:`RATIO` or
more (one word changed in ten). A number is a fact word: the day, the stretch,
the part or the length a row's data carries is what tells two posts apart, so
"Mar 5, 2nd stretch: X" and "Mar 5, 3rd stretch: X" are distinct, while
"Tryon", "Tryon drive" and "TRYON 🚗 #Shorts" are one title. Why words, not a
character ratio: two short place names differ by a letter when the places
differ ("Saluda", "Salida"), and shared boilerplate ("Drive through ... Full
drive: <link>") lifts every character ratio.

Scope: one channel (the row's account; YouTube rows record none, so they are
the one configured channel), over its live rows -- queued, in flight, and
published in the last :data:`RECENT_DAYS`. Shadow, cancelled and
``failed_permanent`` rows never go out, and a row this guard held is out of
the comparison.

Three doors: ``enqueue_from_plan_sidecar`` refuses a Short whose YouTube text
is near-identical to a live row's; ``YouTubeShortsAdapter.publish`` re-checks
against rows that already reached YouTube and holds the row; each drain pass
(:func:`hold_near_duplicate_rows`) holds the later of two queued rows that
collide; a row the auth pause held back (``queue.AUTH_PAUSED_ERROR_CLASS``)
keeps its reason and is left to the publish re-check. A hold parks the row at ``needs_operator`` with
:data:`HOLD_ERROR_CLASS` and the reason in ``error_detail``; nothing is deleted
and nothing is uploaded, and ``dispatch.sync_media_stall_incidents`` raises
the ``delivery_needs_operator`` incident the health check pushes.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..pipeline.final_text_record import FinalTextReviewError
from .anti_spam_gate import CROSS_BATCH_WINDOW_DAYS
from .queue import (
    AUTH_PAUSED_ERROR_CLASS,
    AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
    NEXT_ATTEMPT_NEEDS_OPERATOR,
    STATUS_MISSING_REMOTE,
    STATUS_NEEDS_OPERATOR,
    transition,
)

logger = logging.getLogger(__name__)

PLATFORM = "youtube"
RATIO = 0.9
MIN_RATIO_WORDS = 10
RECENT_DAYS = CROSS_BATCH_WINDOW_DAYS
#: ``error_class`` of a row this guard parked.
HOLD_ERROR_CLASS = "near_duplicate_text"

#: Rows that already reached YouTube (or may have: ``submitting``).
_REACHED = ("published", "uploaded", "submitting", STATUS_MISSING_REMOTE)
#: Rows the drain sweep may hold: waiting for their first insert.
_HOLDABLE = ("queued", "retry_wait", "deferred")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WORD = re.compile(r"[a-z0-9]+")
_FILLER = frozenset(
    {
        # Waypost's filler words (seo_manager DEFAULT_FILLER_WORDS)
        "to", "via", "and", "from", "loop", "drive", "in", "4k", "shorts", "the",
        "of", "at", "near", "through",
        # the generator's descriptors, which every Short shares
        "a", "an", "on", "pov", "driving", "drives", "first", "person", "front",
        "seat", "view", "full", "dashcam",
    }
)  # fmt: skip


def fact_words(text: Optional[str]) -> Tuple[str, ...]:
    """The words of ``text`` that state something (see the module docstring);
    all of its words when every one is filler."""
    from ..pipeline.text_format import split_trailing_hashtags

    body = _URL.sub(" ", split_trailing_hashtags(text or "")[0])
    folded = unicodedata.normalize("NFKD", body).encode("ascii", "ignore").decode("ascii")
    words = _WORD.findall(folded.casefold())
    return tuple(w for w in words if w not in _FILLER) or tuple(words)


def _near(a: Sequence[str], b: Sequence[str]) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    short, total = min(len(a), len(b)), len(a) + len(b)
    if short < MIN_RATIO_WORDS or 2 * short / total < RATIO:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= RATIO


def near_identical(a: Optional[str], b: Optional[str]) -> bool:
    """``a`` and ``b`` say the same thing by the module's rule."""
    return _near(fact_words(a), fact_words(b))


def near_any(text: Optional[str], others) -> bool:
    """``text`` is near-identical to one of ``others``."""
    key = fact_words(text)
    return any(_near(key, fact_words(o)) for o in others)


class NearIndex:
    """``text in index``: near-identical to one of ``texts``. Below
    :data:`MIN_RATIO_WORDS` fact words only equality can match, so most lookups
    are one set probe."""

    def __init__(self, texts=()) -> None:
        self._exact: set = set()
        self._long: List[Tuple[str, ...]] = []
        for text in texts:
            self.add(text)

    def add(self, text: Optional[str]) -> None:
        key = fact_words(text)
        if key:
            self._exact.add(key)
            if len(key) >= MIN_RATIO_WORDS:
                self._long.append(key)

    def __contains__(self, text: object) -> bool:
        key = fact_words(text if isinstance(text, str) else "")
        if not key:
            return False
        if key in self._exact:
            return True
        return len(key) >= MIN_RATIO_WORDS and any(_near(key, k) for k in self._long)


@dataclass(frozen=True)
class PostText:
    """One YouTube row's title and the description it sends, cut to fact words."""

    project_name: str
    short_index: int
    title: str
    description: str
    delivery_id: Optional[int] = None
    status: str = "queued"
    reached: bool = False
    holdable: bool = False

    @cached_property
    def keys(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        return fact_words(self.title), fact_words(self.description)


def sent_texts(record) -> Tuple[str, str]:
    """``(title, description)`` the YouTube adapter sends for ``record``: a Short
    with region votes sends its reviewed description; any other, or one the review
    does not agree with (it is never sent), is compared by its plan text."""
    payload = record.payloads.youtube
    description = payload.description or ""
    if record.sampled_regions:
        try:
            from seo_manager.analytics.final_text_surfaces import compose_description

            from .youtube_adapter import build_short_final_text_review

            review = build_short_final_text_review(record)
            if review.get("verdict") == "agree":
                description = compose_description(review["surfaces"])
        except (FinalTextReviewError, OSError, ValueError, KeyError, TypeError):
            # A refused or unbuildable review sends nothing: keep the plan text.
            pass
    return payload.title, description


def find_near_duplicate(text: PostText, others: Sequence[PostText]) -> Optional[str]:
    """Why ``text`` must not go out beside ``others`` (another Short's live row), or ``None``."""
    title, description = text.keys
    for other in others:
        if (other.project_name, other.short_index) == (text.project_name, text.short_index):
            continue
        other_title, other_description = other.keys
        field, mine, theirs = None, "", ""
        if _near(title, other_title):
            field, mine, theirs = "title", text.title, other.title
        elif _near(description, other_description):
            field, mine, theirs = "description", text.description, other.description
        if field:
            where = f"delivery {other.delivery_id}" if other.delivery_id else "this plan"
            return (
                f"near_duplicate:{PLATFORM} {field} is near-identical to {where} "
                f"({other.project_name} #{other.short_index}, {other.status}): "
                f"{mine[:60]!r} ~ {theirs[:60]!r}"
            )
    return None


def live_texts(
    conn, *, now: Optional[str] = None, account: str = "", reached_only: bool = False
) -> List[PostText]:
    """Every live YouTube row of ``account``'s channel (see the module docstring),
    reached rows first, then the rest in slot order. A row's text is read back from
    the sidecar it was queued from; an unreadable one falls back to the stored title."""
    from .plan_io import load_publish_plan

    now_dt = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
    cutoff = (now_dt - timedelta(days=RECENT_DAYS)).isoformat()
    rows = conn.execute(
        "SELECT id, project_name, short_index, status, sidecar_path, content_title_key, "
        "platform_upload_id, error_class, lease_expires_at, "
        "COALESCE(platform_published_at, published_at, updated_at) AS went_out "
        "FROM deliveries WHERE platform=? AND shadow=0 "
        "AND status NOT IN ('cancelled', 'failed_permanent') "
        "AND COALESCE(upload_account, destination_account, '') = ? "
        "ORDER BY COALESCE(scheduled_at, created_at), id",
        (PLATFORM, account),
    ).fetchall()
    plans: Dict[str, Optional[Dict[Tuple[str, int], Any]]] = {}
    out: List[PostText] = []
    for row in rows:
        status = row["status"]
        if status == STATUS_NEEDS_OPERATOR and row["error_class"] == HOLD_ERROR_CLASS:
            continue
        reached = status in _REACHED or bool(row["platform_upload_id"])
        if status in ("published", STATUS_MISSING_REMOTE) and (row["went_out"] or "") < cutoff:
            continue
        if reached_only and not reached:
            continue
        path = row["sidecar_path"]
        if path and path not in plans:
            try:
                plans[path] = {
                    (r.project_name, r.short_index): r for r in load_publish_plan(path).shorts
                }
            except (OSError, ValueError) as exc:  # gone or malformed: the stored title only
                logger.warning("youtube-distinct: unreadable sidecar %s: %s", path, exc)
                plans[path] = None
        record = (plans.get(path) or {}).get((row["project_name"], row["short_index"]))
        if record is not None and record.payloads.youtube is not None:
            title, description = sent_texts(record)
        else:
            title, description = row["content_title_key"] or "", ""
        leased = (row["lease_expires_at"] or "") > now_dt.isoformat()
        out.append(
            PostText(
                row["project_name"],
                row["short_index"],
                title,
                description,
                delivery_id=row["id"],
                status=status,
                reached=reached,
                # A row the auth pause held back keeps its own reason; the publish
                # re-check still holds it once the platform is resumed.
                holdable=status in _HOLDABLE
                and not reached
                and not leased
                and row["error_class"] != AUTH_PAUSED_ERROR_CLASS,
            )
        )
    return [t for t in out if t.reached] + [t for t in out if not t.reached]


def hold(conn, delivery_id: int, status: str, reason: str) -> bool:
    """Park the row at ``needs_operator`` with ``reason``: never uploaded, never
    deleted, never selected by the drain until a person decides it."""
    held = transition(
        conn,
        delivery_id,
        status,
        STATUS_NEEDS_OPERATOR,
        next_attempt_at=AWAITING_MANUAL_PUBLISH_NEXT_ATTEMPT,
        next_attempt_reason=NEXT_ATTEMPT_NEEDS_OPERATOR,
        error_class=HOLD_ERROR_CLASS,
        error_detail=reason[:500],
        lease_owner=None,
        lease_expires_at=None,
    )
    if held:
        logger.warning("youtube-distinct: delivery %s held: %s", delivery_id, reason)
    return held


def hold_near_duplicate_rows(conn, *, now: Optional[str] = None) -> List[int]:
    """Hold every queued YouTube row near-identical to one that reached YouTube or
    to an earlier-slotted queued row; returns the held ids. The earlier slot keeps
    its place, so of two colliding queued rows exactly the later one is held."""
    kept: List[PostText] = []
    held: List[int] = []
    for text in live_texts(conn, now=now):
        reason = find_near_duplicate(text, kept) if text.holdable else None
        if reason and hold(conn, text.delivery_id, text.status, reason):
            held.append(text.delivery_id)
        else:
            kept.append(text)
    return held
