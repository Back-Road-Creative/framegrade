"""Day reels from Drive: an owner's phone reel, privacy-blurred, into the queue.

Privacy is never self-attested: a reel is queued only when every required class's
detector resolved, the blur render (cabin check included) finished, and the output
has no location tags. See ``docs/social-publishing-queue.md`` ("Day reels from Drive").
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

from ..pipeline.orchestrator_lock import GpuLockBusyError, acquire_gpu_lock, release_gpu_lock
from .asset_plan import build_asset_sidecar, write_asset_bundle
from .dispatch import PRIVACY_REQUIRED_CLASSES, PRIVACY_VERIFIED
from .enqueue_from_plan import (
    DeliveryPlanConflictError,
    PublishPlanSidecarError,
    enqueue_from_plan_sidecar,
)
from .media_fingerprint import fingerprint_media, verify_media
from .plan_io import load_publish_plan
from .queue import open_queue_db

REQUEST_SCHEMA = "gms.day_reel_request/v1"
PLATFORMS = ("facebook", "instagram", "youtube")
#: Every day reel also goes to YouTube as a Short, whether or not the request
#: lists it (owner, 2026-11-25): only a reel YouTube would not treat as a Short
#: (landscape, or longer than ``MAX_SHORT_SECONDS``) stays off it.
YOUTUBE = "youtube"
#: YouTube Shorts: vertical or square, up to three minutes.
MAX_SHORT_SECONDS = 180.0
#: ``videos.insert`` limits: title 100 characters, tags 500 in total.
MAX_YOUTUBE_TITLE_CHARS = 100
MAX_YOUTUBE_TAG_CHARS = 450
#: The YouTube upload guard (``drive.youtube_upload_filename_pattern``) admits
#: only canonical names, so the one render every platform shares is a short.
RENDER_NAME = "short_01.mp4"
#: Instagram's caption limit; Facebook's is far larger.
MAX_CAPTION_CHARS = 2200
REMOTE = "gmsdayreels"
_FIELDS = {"schema", "video_file_id", "caption", "recording_date", "platforms", "at", "journal_url"}
_JOURNAL_URL = re.compile(r"https://(?:www\.)?gomoveshift\.com/[^\s<>\"]{0,300}")
_HASHTAG = re.compile(r"#(\w+)")
_DRIVE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{9,127}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PASTE_START = "Paste the following into your remote machine --->"
_PASTE_END = "<---End paste"
#: Listing and reading a request are tiny calls: give up inside a minute instead of letting
#: rclone's pacer back off for the cron's whole budget when Google throttles or a TLS
#: handshake stalls. The next hourly run retries.
READ_TIMEOUT_S = 60.0
READ_FLAGS = ("--low-level-retries", "2", "--contimeout", "15s")
_THROTTLED = re.compile(r"rateLimitExceeded|RATE_LIMIT_EXCEEDED|Quota exceeded|Rate limited", re.I)


class DriveReadError(RuntimeError):
    """rclone could not list, read or download from the request folder."""


@dataclass(frozen=True)
class DayReelRequest:
    name: str
    video_file_id: str
    caption: str
    recording_date: str
    platforms: Tuple[str, ...]
    at: Optional[str]
    journal_url: Optional[str] = None


@dataclass(frozen=True)
class Outcome:
    """``state``: enqueued | held (retried next run) | refused (recorded,
    never retried) | done (handled by an earlier run) | would-process |
    would-refuse (dry run)."""

    name: str
    state: str
    reason: str = ""
    delivery_ids: Tuple[int, ...] = ()


def parse_request(name: str, text: str) -> DayReelRequest:
    """Validate one request file. Raises ``ValueError`` naming the bad field."""
    try:
        doc = json.loads(text)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not valid JSON") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{name}: not a JSON object")
    if doc.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"{name}: schema must be {REQUEST_SCHEMA!r}")
    unknown = sorted(set(doc) - _FIELDS)
    if unknown:
        raise ValueError(f"{name}: unknown field(s) {unknown}")

    video_file_id = doc.get("video_file_id")
    if not isinstance(video_file_id, str) or not _DRIVE_ID.fullmatch(video_file_id):
        raise ValueError(f"{name}: video_file_id is not a Drive file id")

    caption = doc.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError(f"{name}: caption is required")
    if len(caption) > MAX_CAPTION_CHARS:
        raise ValueError(f"{name}: caption is over {MAX_CAPTION_CHARS} characters")

    recording_date = doc.get("recording_date")
    try:
        if not isinstance(recording_date, str) or not _DATE.fullmatch(recording_date):
            raise ValueError
        date.fromisoformat(recording_date)
    except ValueError:
        raise ValueError(f"{name}: recording_date must be YYYY-MM-DD") from None

    platforms = doc.get("platforms")
    if (
        not isinstance(platforms, list)
        or not platforms
        or len(set(platforms)) != len(platforms)
        or not set(platforms) <= set(PLATFORMS)
    ):
        raise ValueError(f"{name}: platforms must be a non-empty subset of {list(PLATFORMS)}")

    at = doc.get("at")
    if at is not None:
        try:
            parsed = datetime.fromisoformat(at) if isinstance(at, str) else None
        except ValueError:
            parsed = None
        if parsed is None or parsed.tzinfo is None:
            raise ValueError(f"{name}: at must be an ISO-8601 time with a UTC offset")
        at = parsed.astimezone(timezone.utc).isoformat()

    journal_url = doc.get("journal_url")
    if journal_url is not None and (
        not isinstance(journal_url, str) or not _JOURNAL_URL.fullmatch(journal_url)
    ):
        raise ValueError(f"{name}: journal_url must be an https://gomoveshift.com/ page")

    return DayReelRequest(
        name=name,
        video_file_id=video_file_id,
        caption=caption,
        recording_date=recording_date,
        platforms=tuple(platforms),
        at=at,
        journal_url=journal_url,
    )


def youtube_title(caption: str, recording_date: str) -> str:
    """The Short's title: the caption's first line without hashtags, at most
    100 characters (cut at a word), no angle brackets (``videos.insert``
    refuses them). A caption that is only hashtags names the day."""
    first = caption.strip().splitlines()[0] if caption.strip() else ""
    text = " ".join(_HASHTAG.sub("", first).replace("<", "").replace(">", "").split())
    text = text.strip(" ,;:-")
    if not text:
        text = f"GoMoveShift day reel, {date.fromisoformat(recording_date):%d %b %Y}"
    if len(text) <= MAX_YOUTUBE_TITLE_CHARS:
        return text
    cut = text[: MAX_YOUTUBE_TITLE_CHARS - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut + "\u2026"


def youtube_tags(caption: str) -> Tuple[str, ...]:
    """The caption's hashtags, in order, deduplicated, within the tag limit."""
    tags, total = [], 0
    for tag in dict.fromkeys(_HASHTAG.findall(caption)):
        if total + len(tag) + 1 > MAX_YOUTUBE_TAG_CHARS:
            break
        tags.append(tag)
        total += len(tag) + 1
    return tuple(tags)


def youtube_description(caption: str, journal_url: Optional[str]) -> str:
    """The caption verbatim, the journal page when the request names one,
    and ``#Shorts``. Angle brackets go: ``videos.insert`` refuses them."""
    parts = [caption.strip().replace("<", "").replace(">", "")]
    if journal_url:
        parts.append(f"Read the full day: {journal_url}")
    if "#shorts" not in caption.lower():
        parts.append("#Shorts")
    return "\n\n".join(parts)


JOURNAL_SITE = "https://gomoveshift.com"
JOURNAL_TRIPS_URL = JOURNAL_SITE + "/journal-trips.json"
MAX_JOURNAL_TRIPS_BYTES = 256 * 1024
_TRIP_SLUG = re.compile(r"[a-z0-9-]{1,60}")


def journal_day_link(trips_text: str, recording_date: str) -> Optional[str]:
    """The live journal page for the trip day a reel was filmed on, or None.

    ``trips_text`` is the site's ``/journal-trips.json``: ``{"trips": [{slug,
    start, days, ...}]}``. Day *n* of a trip is ``start + n - 1``, the numbering
    the journal page itself uses for its ``#day-n`` sections. A date that falls
    in no trip's window, a malformed list, or two trips claiming the same date
    give None: a reel without a link is fine, a link to the wrong day is not.
    """
    try:
        filmed = date.fromisoformat(recording_date)
        trips = json.loads(trips_text).get("trips")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(trips, list):
        return None
    links = []
    for trip in trips:
        if not isinstance(trip, dict):
            continue
        slug, start, days = trip.get("slug"), trip.get("start"), trip.get("days")
        if not (isinstance(slug, str) and _TRIP_SLUG.fullmatch(slug)):
            continue
        if not isinstance(days, int) or isinstance(days, bool) or not 1 <= days <= 366:
            continue
        try:
            first = date.fromisoformat(start) if isinstance(start, str) else None
        except ValueError:
            continue
        if first is None:
            continue
        n = (filmed - first).days + 1
        if 1 <= n <= days:
            links.append(f"{JOURNAL_SITE}/journeys/{slug}/#day-{n}")
    return links[0] if len(links) == 1 else None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # a redirect is a refusal: never follow it off the site


def fetch_journal_trips(timeout_s: float = 20.0) -> Optional[str]:
    """GET the site's public trip list (first-party, no data sent), or None."""
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(JOURNAL_TRIPS_URL, headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=timeout_s) as response:
            if response.status != 200:
                return None
            data = response.read(MAX_JOURNAL_TRIPS_BYTES + 1)
    except Exception:  # any network failure only means "no link this time"
        return None
    if len(data) > MAX_JOURNAL_TRIPS_BYTES:
        return None
    return data.decode("utf-8", errors="replace")


def lookup_journal_link(recording_date: str) -> Optional[str]:
    trips = fetch_journal_trips()
    return journal_day_link(trips, recording_date) if trips else None


def _short_platforms(request: DayReelRequest, media) -> Tuple[Tuple[str, ...], str]:
    """The request's platforms plus YouTube, unless the render is not a Short.
    Returns ``(platforms, note)``; ``note`` says why YouTube was left out."""
    base = tuple(p for p in request.platforms if p != YOUTUBE)
    if media.width > media.height:
        return base, "YouTube skipped: landscape is not a Short"
    if float(media.duration_seconds) > MAX_SHORT_SECONDS:
        return base, f"YouTube skipped: over {int(MAX_SHORT_SECONDS)}s is not a Short"
    return base + (YOUTUBE,), ""


def parse_authorize_output(stdout: str) -> str:
    """The token JSON from what ``rclone authorize`` prints between its paste markers.

    Given a base64 config blob, rclone v1.60 answers in kind: base64 (URL-safe,
    unpadded) of ``{"token": "<token JSON>"}``. Plain token JSON (older form) is
    accepted too. Errors never echo the reply: it is a credential."""
    start = stdout.find(_PASTE_START)
    end = stdout.find(_PASTE_END)
    if start < 0 or end <= start:
        raise ValueError("rclone authorize printed no token")
    reply = stdout[start + len(_PASTE_START) : end].strip()
    if not reply.startswith("{"):
        try:
            padded = reply + "=" * (-len(reply) % 4)
            wrapper = json.loads(base64.urlsafe_b64decode(padded))
            reply = wrapper["token"]
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("rclone authorize reply is not a token") from exc
    try:
        token = json.loads(reply)
    except (ValueError, TypeError) as exc:
        raise ValueError("rclone authorize reply is not a token") from exc
    if not isinstance(token, dict) or not token.get("refresh_token"):
        raise ValueError("rclone authorize reply has no refresh token")
    return reply


class RcloneDrive:
    """Read-only view of the request folder through an env-configured remote.

    The token travels only in the child's environment, never in argv or on
    disk, and the remote's scope is ``drive.readonly``: this reader cannot
    change or delete anything in Drive. ``client_id``/``client_secret`` are the
    Google client the token was issued to (the pipeline's own, so Google's
    per-project quota is ours alone); a sign-in saved without one keeps rclone's
    built-in client, which every rclone user shares."""

    def __init__(
        self,
        *,
        folder_id: str,
        token: str,
        client_id: str = "",
        client_secret: str = "",
        rclone: str = "rclone",
        runner: Callable = subprocess.run,
        download_timeout_s: float = 1800.0,
    ) -> None:
        self._folder_id = folder_id
        self._token = token
        self._client = (client_id, client_secret)
        self._rclone = rclone
        self._runner = runner
        self._download_timeout_s = download_timeout_s

    def _env(self) -> dict:
        prefix = f"RCLONE_CONFIG_{REMOTE.upper()}_"
        env = dict(os.environ)
        for key, value in (("TYPE", "drive"), ("SCOPE", "drive.readonly")):
            env[prefix + key] = value
        env[prefix + "TOKEN"] = self._token
        env[prefix + "ROOT_FOLDER_ID"] = self._folder_id
        if self._client[0]:
            env[prefix + "CLIENT_ID"], env[prefix + "CLIENT_SECRET"] = self._client
        return env

    def _run(self, *args: str, timeout: float = READ_TIMEOUT_S) -> str:
        try:
            result = self._runner(
                [self._rclone, *args],
                env=self._env(),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if _THROTTLED.search(_text(getattr(exc, "stderr", None))):
                raise DriveReadError(_throttled_message(args[0])) from None
            raise DriveReadError(f"rclone {args[0]} could not run: {exc}") from None
        if result.returncode != 0:
            if _THROTTLED.search(result.stderr or ""):
                raise DriveReadError(_throttled_message(args[0]))
            tail = (result.stderr or "").strip().splitlines()[-1:] or ["no detail"]
            raise DriveReadError(f"rclone {args[0]} failed (exit {result.returncode}): {tail[0]}")
        return result.stdout

    def list_requests(self) -> list:
        try:
            entries = json.loads(
                self._run("lsjson", f"{REMOTE}:", "--files-only", "--max-depth", "1", *READ_FLAGS)
            )
        except ValueError:
            raise DriveReadError("rclone lsjson returned no JSON") from None
        return sorted(
            entry["Name"]
            for entry in entries
            if not entry.get("IsDir") and str(entry.get("Name", "")).endswith(".json")
        )

    def read_request(self, name: str) -> str:
        if "/" in name:
            raise DriveReadError(f"request name {name!r} is not a top-level file")
        return self._run("cat", f"{REMOTE}:{name}", *READ_FLAGS)

    def download(self, file_id: str, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        argv = ("backend", "copyid", f"{REMOTE}:", file_id, f"{dest_dir}/")
        self._run(*argv, timeout=self._download_timeout_s)
        files = [path for path in dest_dir.iterdir() if path.is_file()]
        if len(files) != 1:
            raise DriveReadError(f"download of {file_id} left {len(files)} files, expected 1")
        return files[0]


def _text(raw) -> str:
    """``TimeoutExpired.stderr`` is bytes even under ``text=True``; None when nothing was read."""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return raw or ""


def _throttled_message(command: str) -> str:
    return (
        f"Google is throttling Drive reads (rclone {command} hit the API rate limit); "
        "the next hourly run retries"
    )


@contextmanager
def no_gpu_lock() -> Iterator[None]:
    yield


@contextmanager
def shared_gpu_lock(wait_s: float = 60.0) -> Iterator[None]:
    """The host's shared GPU lock; raises ``GpuLockBusyError`` if still held
    after ``wait_s`` (the request stays pending for the next run)."""
    fd, path = acquire_gpu_lock(wait_s=wait_s)
    try:
        yield
    finally:
        release_gpu_lock(fd, path)


def _default_privacy_reporter(*, blur: bool) -> dict:
    from ..pipeline.photos.face_filter import attestable_privacy_classes

    return attestable_privacy_classes(blur=blur)


def _default_renderer(source: Path, output: Path, mask: Path) -> dict:
    from ..pipeline.photos.night_private_video import _render_private_video

    return _render_private_video(source, output, mask)


def _default_location_check(path: Path) -> bool:
    from ..pipeline.photos.night_private_video import _location_metadata_absent

    return _location_metadata_absent(path)


def _key(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _unverified(privacy: dict) -> list:
    return [
        name
        for name in PRIVACY_REQUIRED_CLASSES
        if (privacy.get(name) or {}).get("state") != PRIVACY_VERIFIED
    ]


def _default_register_tracking(project: str, index: int, file: str, duration_sec: float) -> None:
    from ..tracking import TrackingStore

    TrackingStore().register_short(project, index, file=file, duration_sec=duration_sec)


def _register_tracking_safely(register: Callable, project: str, short: dict) -> Optional[str]:
    """Give the reel a tracking record, so the publish step can store its
    platform ids and ``metrics collect`` and the growth review can see it.

    ``enqueue-plan --apply`` does the same for drive Shorts
    (``cli.handlers.enqueue_plan._register_enqueued_shorts_in_tracking``);
    this route enqueued straight from its sidecar and so had no record, which
    left every published reel's ids unregistered. Never fails the reel: it is
    already queued. Returns a note on failure."""
    media = short.get("media") or {}
    try:
        register(
            project,
            int(short.get("short_index", 0)),
            str(media.get("path") or ""),
            float(media.get("duration_seconds") or 0.0),
        )
    except Exception as exc:  # noqa: BLE001 -- tracking must never undo a queued reel
        return f"tracking record not written: {exc}"
    return None


def process_requests(
    drive,
    *,
    state_dir: Path,
    storage_path: Path,
    apply: bool,
    now_utc: str,
    privacy_reporter: Callable = _default_privacy_reporter,
    renderer: Callable = _default_renderer,
    location_check: Callable[[Path], bool] = _default_location_check,
    gpu_lock: Callable = shared_gpu_lock,
    journal_link: Callable[[str], Optional[str]] = lookup_journal_link,
    register_tracking: Callable = _default_register_tracking,
) -> list:
    """Handle every request in the folder once. Returns one :class:`Outcome`
    per request. A dry run (``apply=False``) reads and validates requests
    only: no download, no render, no file written."""
    state_dir = Path(state_dir)
    outcomes = []
    privacy: Optional[dict] = None
    for name in drive.list_requests():
        ledger = state_dir / "requests" / f"{_key(name)}.json"
        if ledger.exists():
            prior = json.loads(ledger.read_text(encoding="utf-8"))
            outcomes.append(Outcome(name, "done", f"{prior.get('state')} earlier"))
            continue
        try:
            request = parse_request(name, drive.read_request(name))
        except ValueError as exc:
            if not apply:
                outcomes.append(Outcome(name, "would-refuse", str(exc)))
                continue
            _write_json(ledger, {"name": name, "state": "refused", "reason": str(exc)})
            outcomes.append(Outcome(name, "refused", str(exc)))
            continue
        if not apply:
            outcomes.append(
                Outcome(name, "would-process", f"{request.video_file_id} -> {request.platforms}")
            )
            continue

        if privacy is None:
            privacy = privacy_reporter(blur=True)
        missing = _unverified(privacy)
        if missing:
            outcomes.append(
                Outcome(name, "held", "privacy models not verified for: " + ", ".join(missing))
            )
            continue
        try:
            with gpu_lock():
                outcome = _process_one(
                    drive,
                    request,
                    ledger=ledger,
                    work=state_dir / "reels" / _key(name),
                    storage_path=Path(storage_path),
                    now_utc=now_utc,
                    privacy=privacy,
                    renderer=renderer,
                    location_check=location_check,
                    journal_link=journal_link,
                    register_tracking=register_tracking,
                )
        except GpuLockBusyError as exc:
            outcome = Outcome(name, "held", f"shared GPU lock busy: {exc}")
        outcomes.append(outcome)
    return outcomes


def _refuse(ledger: Path, request: DayReelRequest, reason: str) -> Outcome:
    record = {"name": request.name, "state": "refused", "reason": reason}
    _write_json(ledger, {**record, "video_file_id": request.video_file_id})
    return Outcome(request.name, "refused", reason)


def _process_one(
    drive,
    request: DayReelRequest,
    *,
    ledger: Path,
    work: Path,
    storage_path: Path,
    now_utc: str,
    privacy: dict,
    renderer: Callable,
    location_check: Callable[[Path], bool],
    journal_link: Callable[[str], Optional[str]],
    register_tracking: Callable = _default_register_tracking,
) -> Outcome:
    if work.exists():
        shutil.rmtree(work)
    incoming = work / "incoming"
    output = work / RENDER_NAME
    mask = work / "privacy-mask.mkv"
    try:
        try:
            source = drive.download(request.video_file_id, incoming)
        except DriveReadError as exc:
            return Outcome(request.name, "held", str(exc))
        try:
            evidence = renderer(source, output, mask)
        except Exception as exc:  # any failed privacy pass refuses; nothing is queued
            return _refuse(ledger, request, f"privacy render failed: {exc}")
    finally:
        shutil.rmtree(incoming, ignore_errors=True)
        mask.unlink(missing_ok=True)

    if not output.is_file() or output.stat().st_size == 0:
        return _refuse(ledger, request, "privacy render produced no video")
    if not location_check(output):
        return _refuse(ledger, request, "location metadata is still present in the render")

    platforms, note = _short_platforms(request, fingerprint_media(output))
    link = request.journal_url
    if not link and YOUTUBE in platforms:
        link = journal_link(request.recording_date)
    doc = build_asset_sidecar(
        output,
        title=youtube_title(request.caption, request.recording_date),
        description=youtube_description(request.caption, link),
        tags=youtube_tags(request.caption),
        recording_date=request.recording_date,
        platforms=platforms,
        scheduled_at_utc=request.at or now_utc,
        project_name=f"day-reel-{request.video_file_id}",
        caption=request.caption,
    )
    sidecar_path, _ = write_asset_bundle(work, doc, {"classes": privacy})
    receipt = {"request": request.name, "video_file_id": request.video_file_id}
    receipt.update(output_sha256=doc["shorts"][0]["media"]["sha256"], evidence=evidence)
    _write_json(work / "day_reel_receipt.json", {**receipt, "privacy_classes": privacy})
    verify_media(load_publish_plan(sidecar_path).shorts[0])
    conn = open_queue_db(storage_path)
    try:
        result = enqueue_from_plan_sidecar(conn, sidecar_path)
    except (DeliveryPlanConflictError, PublishPlanSidecarError) as exc:
        # Recorded so this one request is not retried every hour, and never
        # stops the requests after it in the folder.
        return _refuse(ledger, request, f"queue refused the reel: {exc}")
    finally:
        conn.close()
    delivery_ids = tuple(int(entry[3]) for entry in result.enqueued)
    present = result.already_present_count
    tracking_note = _register_tracking_safely(
        register_tracking, doc["shorts"][0]["project_name"], doc["shorts"][0]
    )
    if tracking_note:
        note = f"{note}; {tracking_note}" if note else tracking_note
    record = {"name": request.name, "state": "enqueued", "sidecar": str(sidecar_path)}
    record.update(video_file_id=request.video_file_id, delivery_ids=list(delivery_ids))
    record.update(platforms=list(platforms))
    _write_json(ledger, record)
    reason = f"{len(delivery_ids)} new, {present} already queued ({', '.join(platforms)})"
    return Outcome(request.name, "enqueued", reason + (f"; {note}" if note else ""), delivery_ids)
