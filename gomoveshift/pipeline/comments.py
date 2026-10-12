"""YouTube comment pull into a shared JSONL ledger."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..paths import gms_data_home


def default_comments_path() -> Path:
    """Comments ledger path: ``GMS_COMMENTS_PATH``, else ``comments.jsonl`` under the data home.

    Reads the environment on every call, so a value set after import is honoured.
    """
    return Path(os.getenv("GMS_COMMENTS_PATH") or gms_data_home() / "comments.jsonl")


# Import-time snapshot kept for callers that still read the constant; new code
# should call ``default_comments_path()``.
DEFAULT_COMMENTS_PATH = default_comments_path()


def parse_since(raw: str, *, now: datetime | None = None) -> datetime:
    """Parse ``--since`` (``<N>d``) into a UTC cutoff datetime."""
    now = now or datetime.now(timezone.utc)
    raw = raw.strip()
    if not raw.endswith("d") or not raw[:-1].isdigit():
        raise ValueError(f"--since expects '<N>d' (got {raw!r})")
    return now - timedelta(days=int(raw[:-1]))


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            cid = json.loads(line).get("comment_id")
        except json.JSONDecodeError:
            cid = None
        if cid:
            ids.add(cid)
    return ids


def append_comments(path: Path, comments: list[dict]) -> int:
    """Append records whose ``comment_id`` isn't already on disk. Returns count written."""
    existing = load_existing_ids(path)
    new = [c for c in comments if c.get("comment_id") not in existing]
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for c in new:
            fh.write(json.dumps(c, sort_keys=True) + "\n")
    return len(new)


def _parse_rfc3339(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


CORRECTION_CUES: tuple[str, ...] = (
    "actually",
    "that's not",
    "thats not",
    "wrong",
    "it's called",
    "its called",
    "not in",
    "you mean",
    "this is",
    "isn't",
    "incorrect",
    "mislabel",
)

_PLACE_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
    r"|\b(?:in|near|at)\s+([A-Z][a-z]+)\b"
)


def find_cues(text: str) -> list[str]:
    """Return the correction cues (case-insensitive) present in ``text``."""
    lowered = (text or "").lower()
    return [cue for cue in CORRECTION_CUES if cue in lowered]


def find_places(text: str) -> list[str]:
    """Return place-name-shaped tokens: 2+ Title-case words, or a Title-case
    word following ``in``/``near``/``at``."""
    places: list[str] = []
    for match in _PLACE_RE.finditer(text or ""):
        place = match.group(1) or match.group(0)
        if place not in places:
            places.append(place)
    return places


def flag_reasons(text: str, known_text: str = "") -> list[str]:
    """Reasons ``text`` should be triaged: matched cues, plus any place name
    not already present in ``known_text`` (the video's own title/description)."""
    reasons = [f"cue:{cue}" for cue in find_cues(text)]
    lowered_known = (known_text or "").lower()
    for place in find_places(text):
        if place.lower() not in lowered_known:
            reasons.append(f"place:{place}")
    return reasons


def load_comments(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def load_video_metadata(
    seo_ledger_path: Path | None = None, videos_yaml_path: Path | None = None
) -> dict[str, dict]:
    """Best-effort video_id -> {title, description} lookup. Both sources are
    optional; absence of either (or both) means callers skip that check."""
    metadata: dict[str, dict] = {}
    if seo_ledger_path and seo_ledger_path.exists():
        for line in seo_ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = event.get("video_id")
            if vid:
                metadata.setdefault(vid, {})
                metadata[vid]["title"] = event.get("title", metadata[vid].get("title", ""))
                metadata[vid]["description"] = event.get(
                    "description", metadata[vid].get("description", "")
                )
    if videos_yaml_path and videos_yaml_path.exists():
        import yaml

        try:
            data = yaml.safe_load(videos_yaml_path.read_text(encoding="utf-8")) or []
        except yaml.YAMLError:
            data = []
        entries = data.values() if isinstance(data, dict) else data
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            vid = entry.get("video_id") or entry.get("id")
            if vid and vid not in metadata:
                metadata[vid] = {
                    "title": entry.get("title", ""),
                    "description": entry.get("description", ""),
                }
    return metadata


def build_triage_report(comments: list[dict], metadata: dict[str, dict] | None = None) -> str:
    """Group comments by video and render a markdown triage report of the
    ones that match a correction cue or contain an unrecognised place name."""
    metadata = metadata or {}
    by_video: dict[str, list[dict]] = {}
    for comment in comments:
        vid = comment.get("video_id") or "unknown"
        by_video.setdefault(vid, []).append(comment)

    lines = ["# Comment Triage Report", ""]
    for vid, video_comments in by_video.items():
        title = metadata.get(vid, {}).get("title", "")
        description = metadata.get(vid, {}).get("description", "")
        known_text = f"{title} {description}"
        flagged = []
        for comment in video_comments:
            reasons = flag_reasons(comment.get("text", ""), known_text)
            if reasons:
                flagged.append((comment, reasons))
        if not flagged:
            continue
        heading = f'## {vid} — "{title}"' if title else f"## {vid}"
        lines.append(heading)
        lines.append("")
        for comment, reasons in flagged:
            lines.append(
                f"- **{', '.join(reasons)}** | {comment.get('author', '')} | "
                f"{comment.get('published_at', '')} | {comment.get('platform', '')} | "
                f'"{comment.get("text", "")}"'
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_CSV_VIDEO_COLS = ("video_id", "video", "url")
_CSV_AUTHOR_COLS = ("author", "user", "username")
_CSV_TEXT_COLS = ("text", "comment", "content")
_CSV_DATE_COLS = ("date", "created", "published_at", "time")


def _video_id_from_value(raw: str) -> str:
    """A TikTok watch URL's video id is its trailing run of digits; a plain
    id column value passes through untouched."""
    raw = raw.strip()
    if "tiktok.com" in raw or raw.startswith("http"):
        digits = re.findall(r"\d+", raw)
        return digits[-1] if digits else raw
    return raw


def _tiktok_comment_id(video_id: str, author: str, text: str) -> str:
    payload = f"tiktok|{video_id}|{author}|{text}".encode()
    return hashlib.sha1(payload).hexdigest()


def _find_csv_col(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def import_tiktok_csv(raw_csv: str, video_id: str | None = None) -> list[dict]:
    """Parse a header-detected TikTok comment export CSV into the shared
    comment schema. Column names are matched case-insensitively; any of
    ``video_id|video|url``, ``author|user|username``, ``text|comment|content``,
    ``date|created|published_at|time`` is accepted."""
    pulled_at = datetime.now(timezone.utc).isoformat()
    reader = csv.DictReader(io.StringIO(raw_csv))
    fieldnames = reader.fieldnames or []
    video_col = _find_csv_col(fieldnames, _CSV_VIDEO_COLS)
    author_col = _find_csv_col(fieldnames, _CSV_AUTHOR_COLS)
    text_col = _find_csv_col(fieldnames, _CSV_TEXT_COLS)
    date_col = _find_csv_col(fieldnames, _CSV_DATE_COLS)

    out: list[dict] = []
    for row in reader:
        raw_video = (row.get(video_col) or "") if video_col else ""
        vid = _video_id_from_value(raw_video) if raw_video else (video_id or "")
        author = (row.get(author_col) or "") if author_col else ""
        text = (row.get(text_col) or "") if text_col else ""
        published_at = (row.get(date_col) or "") if date_col else ""
        out.append(
            {
                "platform": "tiktok",
                "comment_id": _tiktok_comment_id(vid, author, text),
                "video_id": vid,
                "author": author,
                "text": text,
                "published_at": published_at,
                "like_count": None,
                "pulled_at": pulled_at,
            }
        )
    return out


def import_tiktok_txt(raw_txt: str, video_id: str) -> list[dict]:
    """Parse a plain-text TikTok comment dump: each non-empty line is either
    ``author: text`` or a bare ``text`` line (author left blank)."""
    pulled_at = datetime.now(timezone.utc).isoformat()
    out: list[dict] = []
    for line in raw_txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            author, text = line.split(":", 1)
            author = author.strip()
            text = text.strip()
        else:
            author, text = "", line
        out.append(
            {
                "platform": "tiktok",
                "comment_id": _tiktok_comment_id(video_id, author, text),
                "video_id": video_id,
                "author": author,
                "text": text,
                "published_at": "",
                "like_count": None,
                "pulled_at": pulled_at,
            }
        )
    return out


def import_tiktok_comments(path: Path, video_id: str | None = None) -> list[dict]:
    """Dispatch a TikTok comment export file to the CSV or plain-text
    parser based on its extension. A ``.txt`` (or any non-CSV) export
    requires ``video_id`` since it carries no per-row video reference."""
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        return import_tiktok_csv(content, video_id)
    if not video_id:
        raise ValueError("--video-id is required for a non-CSV TikTok import")
    return import_tiktok_txt(content, video_id)


KNOWN_PLATFORMS: tuple[str, ...] = ("youtube", "instagram", "facebook", "tiktok")

# Platforms with no existing comment pull/import path (only YouTube comments are
# fetched; TikTok is import-only via an operator-supplied export -- see
# import_tiktok_comments). Listed here so the inbox can say "not connected"
# instead of rendering a possibly-empty count that looks like zero comments.
UNCONNECTED_PLATFORMS: tuple[str, ...] = ("instagram", "facebook")


def build_inbox_rows(
    comments: list[dict],
    *,
    platform: str | None = None,
    unanswered_only: bool = False,
) -> list[dict]:
    """Read-only unified inbox over the shared JSONL ledger: every comment,
    newest ``published_at`` first, regardless of which pull/import wrote it.

    ``replied`` is always ``None`` (unknown) -- the ledger records no reply
    state yet. ``unanswered_only`` is accepted now so the console filter
    wires up without a second migration once that field exists; today it is
    a no-op (nothing is known to be replied)."""
    rows = [
        {
            "platform": c.get("platform", ""),
            "video_id": c.get("video_id", ""),
            "author": c.get("author", ""),
            "text": c.get("text", ""),
            "published_at": c.get("published_at", ""),
            "replied": None,
        }
        for c in comments
    ]
    rows.sort(key=lambda r: r["published_at"] or "", reverse=True)
    if platform:
        rows = [r for r in rows if r["platform"] == platform]
    if unanswered_only:
        rows = [r for r in rows if r["replied"] is not True]
    return rows


def inbox_platform_status(comments: list[dict]) -> list[dict]:
    """One row per known platform: how many ledger comments it has, or --
    for a platform with no existing pull/import path -- ``"not connected"``
    rather than a count that would look like zero comments came in."""
    counts: dict[str, int] = {}
    for c in comments:
        p = c.get("platform", "")
        counts[p] = counts.get(p, 0) + 1
    return [
        {
            "platform": p,
            "status": "not connected" if p in UNCONNECTED_PLATFORMS else "connected",
            "count": 0 if p in UNCONNECTED_PLATFORMS else counts.get(p, 0),
        }
        for p in KNOWN_PLATFORMS
    ]


def fetch_youtube_comments(service, channel_id: str, since: datetime) -> list[dict]:
    """Page ``commentThreads.list`` (order=time, newest-first) until publishedAt < since."""
    pulled_at = datetime.now(timezone.utc).isoformat()
    out: list[dict] = []
    page_token = None
    threads = service.commentThreads()
    while True:
        response = threads.list(
            part="snippet",
            allThreadsRelatedToChannelId=channel_id,
            order="time",
            maxResults=100,
            pageToken=page_token,
        ).execute()
        stop = False
        for item in response.get("items", []):
            top = item["snippet"]["topLevelComment"]
            snippet = top["snippet"]
            published_raw = snippet.get("publishedAt", "")
            published_dt = _parse_rfc3339(published_raw)
            if published_dt is not None and published_dt < since:
                stop = True
                break
            out.append(
                {
                    "platform": "youtube",
                    "comment_id": top.get("id"),
                    "video_id": snippet.get("videoId"),
                    "author": snippet.get("authorDisplayName"),
                    "text": snippet.get("textDisplay"),
                    "published_at": published_raw,
                    "like_count": snippet.get("likeCount"),
                    "pulled_at": pulled_at,
                }
            )
        if stop:
            break
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return out
