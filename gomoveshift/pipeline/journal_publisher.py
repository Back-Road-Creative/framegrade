"""Merge YouTube upload records from a drive run into a journey page
``content/journeys/<slug>.md`` frontmatter ``days[N].videos[]`` array.

Pure merge logic lives here; CLI is in ``gomoveshift/cli/handlers/publish_journal.py``.

Idempotency contract:
- Skip-by-``youtube_id``: never overwrite an existing entry. Humans shorten
  titles after the fact; auto-overwrite would
  fight the human-side narrative.
- Append new entries in upload order under the matching ``days[].date``.
- Fall back through title sources: ``youtube_uploads[lf]["title"]`` ->
  ``per_lf_titles[lf]``. Truncate at 100 chars (YouTube's own limit).
- Duration from YouTube's published ``contentDetails`` for the LF's
  ``video_id`` (the source of truth once uploaded — survives a deleted local
  encode); ffprobe on ``regular_lf_paths[lf]`` is the fallback when YouTube
  has no duration for that id yet. Both resolvers are injected for
  testability.
"""

from __future__ import annotations

import io
import logging
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from ruamel.yaml import YAML

from .orchestrator_lock import ConcurrentRunError, _acquire_run_lock, _release_run_lock
from .site_sync.videos_yaml import VideoLibrary

logger = logging.getLogger(__name__)

MAX_TITLE_LEN = 100

# Hugo frontmatter delimiter — three hyphens at line start.
_FRONTMATTER_DELIM = "---"


@dataclass
class VideoEntry:
    """One new ``videos[]`` entry slated for insertion under a specific day."""

    lf_index: int
    youtube_id: str
    title: str
    duration: str
    day_date: str
    # ISO-8601 UTC string from the upload sidecar's ``publish_at`` field.
    # ``None`` means "no schedule — publish immediately" (legacy + manual flows).
    publish_at: Optional[str] = None


@dataclass
class MergePlan:
    inserts: list[VideoEntry]
    skips: list[tuple[int, str, str]]  # (lf_index, youtube_id, reason)
    errors: list[str]
    days_to_create: list[str] = field(default_factory=list)
    # Entries withheld because their ``publish_at`` is still in the future.
    # The CLI handler writes these to the pending sidecar; they are NOT
    # passed to ``apply_merge``.
    deferrals: list[VideoEntry] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def preview_text(self) -> str:
        lines: list[str] = []
        if self.inserts:
            lines.append("Planned inserts:")
            for entry in self.inserts:
                lines.append(
                    f"  + day {entry.day_date} LF {entry.lf_index}: "
                    f"{entry.youtube_id} '{entry.title}' ({entry.duration})"
                )
        if self.days_to_create:
            lines.append("Planned day stubs:")
            for d in self.days_to_create:
                lines.append(f"  + new day stub: {d}")
        if self.deferrals:
            lines.append("Deferred until publish_at fires:")
            for entry in self.deferrals:
                lines.append(
                    f"  ~ day {entry.day_date} LF {entry.lf_index}: "
                    f"{entry.youtube_id} '{entry.title}' (publish_at={entry.publish_at})"
                )
        if self.skips:
            lines.append("Skipped (already present):")
            for lf_idx, vid, reason in self.skips:
                lines.append(f"  - LF {lf_idx}: {vid} ({reason})")
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  ! {err}")
        if not lines:
            lines.append("(nothing to do)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# YAML round-trip — ruamel preserves key order / quoting / comments.
# ---------------------------------------------------------------------------


def _yaml() -> YAML:
    y = YAML(typ="rt")
    y.preserve_quotes = True
    y.width = 4096  # don't fold long strings
    y.indent(mapping=2, sequence=2, offset=0)
    return y


def load_journey(path: Path) -> tuple[dict, str]:
    """Parse a Hugo-style markdown file. Returns (frontmatter_dict, body_text)."""
    raw = Path(path).read_text()
    if not raw.startswith(_FRONTMATTER_DELIM):
        raise ValueError(f"{path}: no leading frontmatter delimiter")
    # Strip leading delim line.
    rest = raw[len(_FRONTMATTER_DELIM) :].lstrip("\r\n")
    end_marker = f"\n{_FRONTMATTER_DELIM}\n"
    end_pos = rest.find(end_marker)
    if end_pos == -1:
        # Allow trailing delim without final newline.
        end_marker = f"\n{_FRONTMATTER_DELIM}"
        end_pos = rest.find(end_marker)
        if end_pos == -1:
            raise ValueError(f"{path}: no closing frontmatter delimiter")
    fm_text = rest[:end_pos]
    body = rest[end_pos + len(end_marker) :]
    if body.startswith("\n"):
        body = body[1:]
    doc = _yaml().load(fm_text) or {}
    return doc, body


def serialize_journey(doc: dict, body: str) -> str:
    """Emit ``---\\n<yaml>\\n---\\n<body>``."""
    buf = io.StringIO()
    _yaml().dump(doc, buf)
    yaml_text = buf.getvalue()
    if not yaml_text.endswith("\n"):
        yaml_text += "\n"
    return f"{_FRONTMATTER_DELIM}\n{yaml_text}{_FRONTMATTER_DELIM}\n{body}"


class JourneyLockBusyError(RuntimeError):
    """Another writer's load -> mutate -> write window over this journey is open."""


@contextmanager
def journey_lock(journey_path: Path, *, skip_if_busy: bool = False):
    """Hold an exclusive flock sidecar (``<journey>.md.lock``) over one journey.

    EVERY journey writer — the photo ingest, the drive run's auto-publish merge
    and its ``--replay-pending`` cron, the channel reconciler — does its own
    load -> mutate -> atomic-write, so two overlapping writers are a
    last-writer-wins lost update: whoever serializes second writes back a doc
    that never saw the other's inserts. All of them take THIS lock across their
    whole window, so the exclusion is mutual rather than one-sided; the
    orchestrator's run-lock machinery (stale-PID recovery) is reused here.

    Yields True while held. A live holder raises :class:`JourneyLockBusyError` —
    right for a writer that has done nothing irreversible yet. ``skip_if_busy``
    yields False instead, for a caller whose irreversible work (the uploads)
    already happened: it warns loudly and skips a recoverable merge.
    """
    fd = lock_path = None
    try:
        fd, lock_path = _acquire_run_lock(
            journey_path.parent, journey_path.parent, lock_name=journey_path.name + ".lock"
        )
    except ConcurrentRunError as exc:
        if not skip_if_busy:
            raise JourneyLockBusyError(
                f"{journey_path}: another writer holds this journey's lock — retry once it finishes"
            ) from exc
    try:
        yield fd is not None
    finally:
        _release_run_lock(fd, lock_path)


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Match the existing site convention: 'MM:SS' if <1h, 'H:MM:SS' if >=1h.

    Hours have no leading zero (e.g. ``'1:32:41'``).
    """
    total = int(round(seconds))
    if total < 0:
        raise ValueError(f"negative duration: {seconds}")
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _no_youtube_durations(video_ids: list[str]) -> dict[str, float]:
    """The DEFAULT lookup: resolves nothing, touches no network.

    ``plan_merge`` documents itself as pure, so its default must not reach the
    network — otherwise every caller that does not inject a lookup (all of the
    tests) silently depends on live credentials, and a machine that has them
    takes a different code path from one that does not. Production injects
    :func:`_default_youtube_duration_lookup` explicitly at the CLI boundary,
    where I/O belongs.
    """
    return {}


def _default_youtube_duration_lookup(video_ids: list[str]) -> dict[str, float]:
    """Batched ``videos.list(contentDetails)`` duration lookup — the real
    network implementation, used automatically unless a test injects a fake.

    Any failure (no credentials, no network, API error) degrades to an empty
    result so callers fall back to ffprobe rather than hard-failing a merge
    over a lookup outage unrelated to the upload itself.
    """
    try:
        from .drive_phases.phase_d_scheduling import _build_readonly_youtube_service
        from .site_sync.channel_manifest import parse_iso8601_duration

        service = _build_readonly_youtube_service()
        out: dict[str, float] = {}
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start : start + 50]
            response = service.videos().list(part="contentDetails", id=",".join(batch)).execute()
            for video in response.get("items", []):
                duration = video.get("contentDetails", {}).get("duration", "")
                out[video["id"]] = float(parse_iso8601_duration(duration))
        return out
    except Exception as exc:
        # Never silent: without this line a credentials/network outage
        # surfaces only as the ffprobe fallback's "ffprobe failed" error,
        # which names the wrong cause and sends the operator to the encode
        # rather than to their auth.
        logger.warning(
            "YouTube duration lookup failed (%s: %s); falling back to ffprobe "
            "on the local encodes, which a cleaned-up run may no longer have.",
            type(exc).__name__,
            exc,
        )
        return {}


def ffprobe_duration_seconds(path: Path) -> float:
    """Default ffprobe shell-out. CLI handler uses this; tests inject a stub."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_int_key_dict(d) -> dict[int, object]:
    """Coerce a state-side container into ``{lf_index_1based: value}``.

    Three input shapes appear in pipeline_state.json:
    - dict with str keys (post-JSON round-trip of a 1-based int-keyed dict):
      ``{"1": ..., "2": ...}`` — `youtube_uploads`, `per_lf_recording_date`,
      `per_lf_titles`.
    - dict with int keys (in-process, pre-JSON):
      ``{1: ..., 2: ...}``.
    - list (0-based; orchestrator's actual on-disk shape for
      ``regular_lf_paths``): ``[path1, path2]`` → lf_index 1 = element 0.

    All three normalize to ``{1: v, 2: v, ...}``.
    """
    if not d:
        return {}
    if isinstance(d, list):
        return {i + 1: v for i, v in enumerate(d)}
    out: dict[int, object] = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _existing_youtube_ids_by_day(doc: dict) -> dict[str, set[str]]:
    """{day_date_str: {youtube_id, ...}} across the journey."""
    out: dict[str, set[str]] = {}
    for day in doc.get("days", []) or []:
        date_str = str(day.get("date"))
        ids: set[str] = set()
        for vid in day.get("videos", []) or []:
            yid = vid.get("youtube_id") if isinstance(vid, dict) else None
            if yid:
                ids.add(str(yid))
        out[date_str] = ids
    return out


def _parse_publish_at(raw) -> Optional[datetime]:
    """Best-effort parse of an upload-sidecar ``publish_at`` field.

    Returns a tz-aware UTC datetime, or None when:
    - the field is absent / empty / non-string
    - the field is present but unparseable (caller decides whether that's
      an error or a "treat as immediate")
    """
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        # ``Z`` suffix isn't accepted by fromisoformat() pre-3.11.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _all_existing_ids(doc: dict) -> dict[str, str]:
    """{youtube_id: day_date} across the whole journey, for cross-day collision detection."""
    out: dict[str, str] = {}
    for day in doc.get("days", []) or []:
        date_str = str(day.get("date"))
        for vid in day.get("videos", []) or []:
            yid = vid.get("youtube_id") if isinstance(vid, dict) else None
            if yid:
                out[str(yid)] = date_str
    return out


# ---------------------------------------------------------------------------
# plan_merge
# ---------------------------------------------------------------------------


def plan_merge(
    state: dict,
    doc: dict,
    *,
    ffprobe_duration: Callable[[Path], float] = ffprobe_duration_seconds,
    youtube_duration_lookup: Callable[[list[str]], dict[str, float]] = _no_youtube_durations,
    allow_day_create: bool = False,
    now: Optional[datetime] = None,
) -> MergePlan:
    """Compute the (inserts, skips, errors, deferrals) plan for merging ``state`` into ``doc``.

    Pure: does not mutate ``doc`` or read the filesystem (ffprobe and the
    YouTube duration lookup are both injected).

    ``now`` is the comparison clock for ``publish_at`` gating; defaults to
    :func:`datetime.now` in UTC. Entries whose ``publish_at`` is in the future
    are moved to ``deferrals`` rather than ``inserts`` so the CLI handler can
    write them to the pending sidecar.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    inserts: list[VideoEntry] = []
    deferrals: list[VideoEntry] = []
    skips: list[tuple[int, str, str]] = []
    errors: list[str] = []
    days_to_create: list[str] = []

    uploads = _normalize_int_key_dict(state.get("youtube_uploads"))
    rec_dates = _normalize_int_key_dict(state.get("per_lf_recording_date"))
    per_lf_titles = _normalize_int_key_dict(state.get("per_lf_titles"))
    lf_paths = _normalize_int_key_dict(state.get("regular_lf_paths"))

    yt_video_ids = sorted(
        {str(r["video_id"]) for r in uploads.values() if isinstance(r, dict) and r.get("video_id")}
    )
    yt_durations: dict[str, float] = youtube_duration_lookup(yt_video_ids) if yt_video_ids else {}

    existing_by_day = _existing_youtube_ids_by_day(doc)
    all_existing = _all_existing_ids(doc)
    existing_day_dates = sorted(existing_by_day.keys())

    # New days will be added as we go; track them so multiple LFs on the
    # same new date collapse into one stub.
    pending_new_dates: set[str] = set()

    for lf_idx in sorted(uploads.keys()):
        record = uploads[lf_idx]
        if not isinstance(record, dict):
            errors.append(f"LF {lf_idx}: upload record is not a dict")
            continue
        video_id = record.get("video_id")
        if not video_id:
            errors.append(f"LF {lf_idx}: upload record missing video_id")
            continue
        video_id = str(video_id)

        rec_date = rec_dates.get(lf_idx)
        if rec_date is None:
            errors.append(f"LF {lf_idx}: no per_lf_recording_date entry")
            continue
        rec_date = str(rec_date)

        # Cross-day collision: video_id already present in a different day.
        prior_day = all_existing.get(video_id)
        if prior_day is not None and prior_day != rec_date:
            errors.append(
                f"LF {lf_idx}: video_id {video_id} already present in day "
                f"{prior_day} but state says recording_date={rec_date}. "
                "Refusing to duplicate or move; resolve manually."
            )
            continue

        # Same-day skip — idempotent.
        if prior_day == rec_date:
            skips.append((lf_idx, video_id, f"already in day {rec_date}"))
            continue

        # Day must exist (or be allowed to create).
        if rec_date not in existing_by_day and rec_date not in pending_new_dates:
            if not allow_day_create:
                errors.append(
                    f"LF {lf_idx}: recording_date {rec_date} not in journey days "
                    f"{existing_day_dates}. Author the day stub manually or "
                    "re-run with --allow-day-create."
                )
                continue
            pending_new_dates.add(rec_date)
            days_to_create.append(rec_date)

        # Title cascade.
        title = record.get("title") or per_lf_titles.get(lf_idx) or ""
        title = str(title)[:MAX_TITLE_LEN]

        # Duration: prefer YouTube's published contentDetails (the source of
        # truth once uploaded — survives a deleted local encode). Fall back
        # to ffprobe on the local encode only when YouTube has no duration
        # for this video_id.
        yt_seconds = yt_durations.get(video_id)
        if yt_seconds is not None:
            seconds = yt_seconds
        else:
            lf_path = lf_paths.get(lf_idx)
            if not lf_path:
                errors.append(f"LF {lf_idx}: no regular_lf_paths entry for duration ffprobe")
                continue
            try:
                seconds = ffprobe_duration(Path(str(lf_path)))
            except Exception as exc:  # ffprobe IO is a system boundary
                errors.append(f"LF {lf_idx}: ffprobe failed for {lf_path}: {exc}")
                continue
        duration = format_duration(seconds)

        publish_at_raw = record.get("publish_at")
        publish_at_dt = _parse_publish_at(publish_at_raw)
        # Canonical storage: ISO-8601 Z form when parseable, else verbatim raw
        # so a downstream operator can still inspect the value.
        if publish_at_dt is not None:
            publish_at_iso: Optional[str] = publish_at_dt.isoformat().replace("+00:00", "Z")
        elif publish_at_raw:
            publish_at_iso = str(publish_at_raw)
        else:
            publish_at_iso = None

        entry = VideoEntry(
            lf_index=lf_idx,
            youtube_id=video_id,
            title=title,
            duration=duration,
            day_date=rec_date,
            publish_at=publish_at_iso,
        )

        # Date-gate: future publish_at → defer to pending sidecar.
        if publish_at_dt is not None and publish_at_dt > now:
            deferrals.append(entry)
        else:
            inserts.append(entry)

    return MergePlan(
        inserts=inserts,
        skips=skips,
        errors=errors,
        days_to_create=days_to_create,
        deferrals=deferrals,
    )


def plan_superlong_merge(
    state: dict,
    superlong_uploads: dict,
    doc: dict,
    *,
    now: Optional[datetime] = None,
) -> MergePlan:
    """ONE merge event per superlong day, listing every part in order,
    gated on the LAST part's ``publish_at`` so a 4h encode
    cap never fragments one day into weeks-apart entries. ``superlong_parts``
    on ``state`` is the plan; ``superlong_uploads`` (the standalone sidecar,
    injected) is upload truth. A day missing any part's upload is omitted;
    a day not yet fully public defers as a GROUP, gated on the day's
    LATEST ``publish_at`` — never written to the journey.
    """
    if now is None:
        now = datetime.now(tz=timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    superlong_parts = state.get("superlong_parts") or {}
    existing_by_day = _existing_youtube_ids_by_day(doc)
    all_existing = _all_existing_ids(doc)

    inserts: list[VideoEntry] = []
    deferrals: list[VideoEntry] = []
    errors: list[str] = []

    for date in sorted(superlong_parts):
        day_plan = superlong_parts[date]
        if not isinstance(day_plan, dict):
            continue
        records = sorted(
            (r for r in (day_plan.get("records") or []) if isinstance(r, dict)),
            key=lambda r: int(r.get("part_index", 0) or 0),
        )
        if not records:
            continue

        parts: list[tuple[dict, dict]] = []
        for record in records:
            idx = int(record.get("part_index", 0) or 0)
            upload = superlong_uploads.get(f"{date}:{idx:02d}")
            if not upload or not upload.get("video_id"):
                parts = []  # not fully uploaded
                break
            parts.append((record, upload))
        if not parts:
            continue

        video_ids = [str(u["video_id"]) for _r, u in parts]
        if any(vid in existing_by_day.get(date, set()) for vid in video_ids):
            continue  # already merged

        skip_day = False
        for vid in video_ids:
            prior_day = all_existing.get(vid)
            if prior_day is not None and prior_day != date:
                errors.append(
                    f"superlong part {vid}: already present in day {prior_day} "
                    f"but its own date is {date}. Refusing to duplicate or move."
                )
                skip_day = True
        if skip_day:
            continue

        publish_ats = [_parse_publish_at(u.get("publish_at")) for _r, u in parts]
        defined = [dt for dt in publish_ats if dt is not None]
        day_ready_at = max(defined) if defined else None
        is_public = day_ready_at is None or day_ready_at <= now

        for record, upload in parts:
            title = str(upload.get("title") or record.get("title") or "")[:MAX_TITLE_LEN]
            duration = format_duration(float(record.get("duration_sec") or 0))
            entry = VideoEntry(
                lf_index=int(record.get("part_index", 0) or 0),
                youtube_id=str(upload["video_id"]),
                title=title,
                duration=duration,
                day_date=date,
                publish_at=upload.get("publish_at"),
            )
            if is_public:
                inserts.append(entry)
            else:
                entry.publish_at = day_ready_at.isoformat().replace("+00:00", "Z")
                deferrals.append(entry)

    return MergePlan(inserts=inserts, skips=[], errors=errors, deferrals=deferrals)


# ---------------------------------------------------------------------------
# apply_merge
# ---------------------------------------------------------------------------


def _day_number_for_new_date(days: list, new_date: str) -> int:
    """Chronological ``day_number`` for a backfilled day dated ``new_date``.

    Numbers by DATE RANK: the stub slots between the existing day immediately
    before it (by date) and the one immediately after, taking ``prev + 1`` so a
    day appended at the end still gets ``max()+1`` while an earlier backfill
    takes its true chronological number instead of being tacked on the end.

    Raises ``ValueError`` when that number is already held by the following day
    — i.e. numbering the backfill correctly would force RENUMBERING an
    already-published day. We refuse rather than silently reorder published
    day numbers.
    """
    prev_num = 0
    next_num: Optional[int] = None
    for d in sorted(days, key=lambda x: str(x.get("date"))):
        date_str = str(d.get("date"))
        num = int(d.get("day_number", 0) or 0)
        if date_str < new_date:
            prev_num = num
        elif date_str > new_date:
            next_num = num
            break
    candidate = prev_num + 1
    if next_num is not None and candidate >= next_num:
        raise ValueError(
            f"backfilled day {new_date} would be numbered {candidate}, already "
            f"published as day {next_num}; inserting it in date order would "
            "renumber existing published days. Refusing to reorder — assign day "
            "numbers manually."
        )
    return candidate


def _day_of_week(date_str: str) -> str:
    from datetime import date as _date

    try:
        y, m, d = (int(x) for x in date_str.split("-"))
        return _date(y, m, d).strftime("%A")
    except Exception:
        return ""


def apply_merge(doc: dict, plan: MergePlan, *, library: VideoLibrary) -> None:
    """Mutate ``doc`` in place to insert new entries. Refuses if plan.errors.

    ``library`` is REQUIRED, not optional: every writer that embeds a video
    records it through the same call, so a published long-form cannot be
    embedded into a journey the site's ``data/videos.yaml`` never learned about
    (the site's video-coverage deploy check only WARNS about that, so it would
    ship silently). The library filters itself down to long-forms, so an
    embedded Short/stream records nothing. Flushed before the caller serializes
    the journey: a library row whose embed never landed is self-healing, an
    embed with no row is the defect.
    """
    if plan.has_errors:
        raise ValueError("apply_merge called on a plan with errors:\n" + "\n".join(plan.errors))

    days = doc.setdefault("days", [])

    # Stub any missing days, numbered by chronological (date) rank and inserted
    # in date order — never blindly appended, so a backfilled earlier day can't
    # ship as a higher-numbered day dated before a lower-numbered one.
    # Process ascending so multiple backfills compose against a sorted list.
    existing_dates = {str(d.get("date")) for d in days}
    for new_date in sorted(plan.days_to_create):
        if new_date in existing_dates:
            continue
        day_number = _day_number_for_new_date(days, new_date)
        insert_at = sum(1 for d in days if str(d.get("date")) < new_date)
        days.insert(
            insert_at,
            {
                "day_number": day_number,
                "date": new_date,
                "day_of_week": _day_of_week(new_date),
                "heading": "",
                "narrative": "",
                "videos": [],
            },
        )
        existing_dates.add(new_date)

    # Index days by date for fast insertion.
    day_by_date: dict[str, dict] = {str(d.get("date")): d for d in days}

    for entry in plan.inserts:
        day = day_by_date.get(entry.day_date)
        if day is None:
            # Shouldn't happen — plan_merge already errored or created stub.
            raise ValueError(f"apply_merge: day {entry.day_date} missing from doc")
        videos = day.setdefault("videos", [])
        # videos may be a ruamel CommentedSeq or a plain list — both .append.
        video_entry = {
            "youtube_id": entry.youtube_id,
            "title": entry.title,
            "duration": entry.duration,
        }
        if entry.publish_at:
            video_entry["publish_at"] = entry.publish_at
        videos.append(video_entry)
        library.record_embed(
            video_id=entry.youtube_id,
            title=entry.title,
            date=entry.day_date,
            duration=entry.duration,
        )

    library.flush()

    # Keep the top-level ``video_count`` in sync with the actual embedded
    # videos. The site repository's deploy CI hard-fails when
    # ``video_count`` != sum(days[].videos), and apply_merge is the single
    # write path for both the immediate insert and the replay/cron flush — so
    # a flush that didn't also bump the count would break the next deploy.
    # Only touch the key when it already exists: a journey that omits it opts
    # out of the check, and we don't impose it.
    if "video_count" in doc:
        doc["video_count"] = sum(len(d.get("videos") or []) for d in days if isinstance(d, dict))
