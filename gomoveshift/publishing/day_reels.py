"""Day reels from Drive: an owner's phone reel, privacy-blurred, into the queue.

Privacy is never self-attested: a reel is queued only when every required class's
detector resolved, the blur render (cabin check included) finished, and the output
has no location tags. See ``docs/social-publishing-queue.md`` ("Day reels from Drive").
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

from ..pipeline.orchestrator_lock import GpuLockBusyError, acquire_gpu_lock, release_gpu_lock
from .asset_plan import build_asset_sidecar, write_asset_bundle
from .dispatch import PRIVACY_REQUIRED_CLASSES, PRIVACY_VERIFIED
from .enqueue_from_plan import enqueue_from_plan_sidecar
from .media_fingerprint import verify_media
from .plan_io import load_publish_plan
from .queue import open_queue_db

REQUEST_SCHEMA = "gms.day_reel_request/v1"
PLATFORMS = ("facebook", "instagram")
#: Instagram's caption limit; Facebook's is far larger.
MAX_CAPTION_CHARS = 2200
REMOTE = "gmsdayreels"
_FIELDS = {"schema", "video_file_id", "caption", "recording_date", "platforms", "at"}
_DRIVE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{9,127}")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PASTE_START = "Paste the following into your remote machine --->"
_PASTE_END = "<---End paste"


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

    return DayReelRequest(
        name=name,
        video_file_id=video_file_id,
        caption=caption,
        recording_date=recording_date,
        platforms=tuple(platforms),
        at=at,
    )


def parse_authorize_output(stdout: str) -> str:
    """The token JSON ``rclone authorize`` prints between its paste markers."""
    start = stdout.find(_PASTE_START)
    end = stdout.find(_PASTE_END)
    if start < 0 or end <= start:
        raise ValueError("rclone authorize printed no token")
    token = stdout[start + len(_PASTE_START) : end].strip()
    json.loads(token)
    return token


class RcloneDrive:
    """Read-only view of the request folder through an env-configured remote.

    The token travels only in the child's environment, never in argv or on
    disk, and the remote's scope is ``drive.readonly``: this reader cannot
    change or delete anything in Drive."""

    def __init__(
        self,
        *,
        folder_id: str,
        token: str,
        rclone: str = "rclone",
        runner: Callable = subprocess.run,
        download_timeout_s: float = 1800.0,
    ) -> None:
        self._folder_id = folder_id
        self._token = token
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
        return env

    def _run(self, *args: str, timeout: float = 120.0) -> str:
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
            raise DriveReadError(f"rclone {args[0]} could not run: {exc}") from None
        if result.returncode != 0:
            tail = (result.stderr or "").strip().splitlines()[-1:] or ["no detail"]
            raise DriveReadError(f"rclone {args[0]} failed (exit {result.returncode}): {tail[0]}")
        return result.stdout

    def list_requests(self) -> list:
        try:
            entries = json.loads(
                self._run("lsjson", f"{REMOTE}:", "--files-only", "--max-depth", "1")
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
        return self._run("cat", f"{REMOTE}:{name}")

    def download(self, file_id: str, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        argv = ("backend", "copyid", f"{REMOTE}:", file_id, f"{dest_dir}/")
        self._run(*argv, timeout=self._download_timeout_s)
        files = [path for path in dest_dir.iterdir() if path.is_file()]
        if len(files) != 1:
            raise DriveReadError(f"download of {file_id} left {len(files)} files, expected 1")
        return files[0]


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
) -> Outcome:
    if work.exists():
        shutil.rmtree(work)
    incoming = work / "incoming"
    output = work / "day_reel.mp4"
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

    doc = build_asset_sidecar(
        output,
        title=request.caption,
        description=request.caption,
        tags=(),
        recording_date=request.recording_date,
        platforms=request.platforms,
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
    finally:
        conn.close()
    delivery_ids = tuple(int(entry[3]) for entry in result.enqueued)
    present = result.already_present_count
    record = {"name": request.name, "state": "enqueued", "sidecar": str(sidecar_path)}
    record.update(video_file_id=request.video_file_id, delivery_ids=list(delivery_ids))
    _write_json(ledger, record)
    return Outcome(
        request.name, "enqueued", f"{len(delivery_ids)} new, {present} already queued", delivery_ids
    )
