"""
YouTubeUploader — resumable YouTube Data API v3 upload for long-form videos.

A first-class, tested module for uploading and re-privating videos.
Credentials load from the shared `_yt_creds.load_youtube_credentials`
helper so analytics sync and upload stay in lock-step.

Key behaviors:
- Resumable chunked upload (MediaFileUpload, resumable=True).
- Transient network errors retry with backoff, capped at 5 attempts.
- `RefreshError` (hard auth failure) is never retried.
- Pipeline-state idempotency via `upload_if_needed(state, lf_idx, ...)`:
  if `state["youtube_uploads"][lf_idx]["video_id"]` exists, returns that record
  and does not re-upload.
- Filename guard (admits long-form, superlong parts and shorts):
  `upload_video` rejects any `video_path` whose basename fails to match
  `drive.youtube_upload_filename_pattern` (default
  `^(?:lf|sl|short)_\\d{2}\\.mp4$`). This is defense-in-depth against the legacy
  "episode NN FINAL" drafts that ended up on the user's channel from an
  ancient pipeline version that uploaded raw episode concats. Current code
  paths do not exhibit that bug, but the guard blocks any future path that
  bypasses LF/superlong/short assembly. Override via config for dev/testing.
- Upload media backstop: see `_enforce_youtube_upload_media_limits`.
- Quota disambiguation: HTTP
  403s carrying a quota-class ``reason`` (``quotaExceeded``,
  ``userRateLimitExceeded``, ``dailyLimitExceeded``) raise
  ``YouTubeQuotaError`` rather than the generic permanent-403 HttpError.
  Phase D's outer upload loop traps this distinctly from ``RefreshError``
  (auth) and from other 403s (permanent auth failure), banners the
  operator recovery steps, and exits cleanly so the upload batch
  is resumable after the quota window resets. See ``YouTubeQuotaError``
  docstring and ``QUOTA_REASONS`` for the reason set.

Never writes credentials to disk — the decrypted dict is held in memory
and passed directly to `google.oauth2.credentials.Credentials`.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from ...config import get_config
from .._yt_creds import DEFAULT_CREDENTIALS_DIR, load_youtube_credentials
from ..endscreen_sidecar import format_endscreen_checklist_line
from ..exceptions import YouTubeQuotaError
from ..playlist_resolver import resolve_playlist

logger = logging.getLogger(__name__)

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
# Needed to post/manage comments (first comment) — the shared upload token
# does not carry this by default; `gomoveshift youtube-auth` mints a token
# with both scopes. See `cli/handlers/youtube_auth.py`.
YOUTUBE_COMMENT_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
# Needed by tracking/collector.py for YouTube Analytics reads (fetch_analytics_*);
# without it every Analytics call 403s insufficientPermissions.
YOUTUBE_ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"

MAX_TITLE_LEN = 100
# YouTube Data API v3 hard-rejection thresholds (not SEO guidance):
MAX_DESCRIPTION_BYTES = 5000
MAX_TAGS_TOTAL_LEN = 500
FORBIDDEN_SNIPPET_CHARS = "<>"
# YouTube's published upload limits ("256 GB or 12 hours, whichever is
# less") — backstop only, see ``_enforce_youtube_upload_media_limits``.
MAX_UPLOAD_DURATION_SEC = 12 * 60 * 60
MAX_UPLOAD_FILE_SIZE_BYTES = 256 * 1024 * 1024 * 1024
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2.0
CHUNK_TIMEOUT_SEC = 600

# HTTP status codes that should trigger a retry
RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# HTTP status codes that are permanent — never retry
PERMANENT_STATUS_CODES = {401, 403, 404}

# HTTP 403 ``reason`` strings that indicate quota / rate-limit exhaustion
# (recoverable after a time-based backoff) rather than permanent auth
# failure. YouTube Data API v3 surfaces both classes as 403 but carries
# the distinguishing hint in ``HttpError.error_details[*].reason``. See
# ``YouTubeQuotaError`` docstring for recovery semantics.
QUOTA_REASONS = frozenset(
    {
        "quotaExceeded",
        "dailyLimitExceeded",
        "userRateLimitExceeded",
    }
)

# Recognised inputs for ``upload_video(recording_date=...)``: a date-only
# ``YYYY-MM-DD`` (most common — the pipeline's ``per_lf_recording_date``
# state) or a pre-normalized ISO 8601 UTC timestamp ending in ``Z``.
_RECORDING_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RECORDING_DATE_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def _normalize_recording_date(value: str) -> str:
    """Coerce ``value`` to the ISO 8601 UTC form YouTube Data API v3 wants.

    ``YYYY-MM-DD`` → ``YYYY-MM-DDT00:00:00.000Z``. A full
    ``YYYY-MM-DDTHH:MM:SS[.fff]Z`` is returned verbatim. Anything else
    raises ``ValueError`` at the boundary so the operator sees the bad
    input immediately, not as an opaque 400 hours into the batch.
    """
    if _RECORDING_DATE_ONLY_RE.match(value):
        return f"{value}T00:00:00.000Z"
    if _RECORDING_DATE_ISO_RE.match(value):
        return value
    raise ValueError(
        f"recording_date must be 'YYYY-MM-DD' or ISO 8601 UTC "
        f"('YYYY-MM-DDTHH:MM:SSZ'); got {value!r}"
    )


def _extract_quota_reason(http_error) -> Optional[str]:
    """Return the first quota-class ``reason`` carried on ``http_error``.

    The googleapiclient ``HttpError`` populates ``error_details`` as a
    list of error dicts (mirrored from the JSON response's
    ``error.errors`` array). Each may contain a ``reason`` key such as
    ``quotaExceeded`` / ``dailyLimitExceeded`` / ``userRateLimitExceeded``.
    Any other 403 reason (``forbidden``, ``insufficientPermissions``,
    etc.) returns ``None`` so the caller falls through to the permanent-
    403 path unchanged.

    Hardened against edge cases: missing ``error_details`` attribute,
    non-iterable value, dict entries without ``reason`` — all return
    ``None`` (treat as non-quota; preserves existing permanent-403
    semantics).
    """
    details = getattr(http_error, "error_details", None)
    if not details:
        return None
    try:
        iterator = iter(details)
    except TypeError:
        return None
    for entry in iterator:
        if not isinstance(entry, dict):
            continue
        reason = entry.get("reason")
        if isinstance(reason, str) and reason in QUOTA_REASONS:
            return reason
    return None


def _enforce_youtube_api_limits(title: str, description: str, tags: list[str]) -> None:
    """Raise ``ValueError`` if metadata violates a hard YouTube Data API v3 limit.

    These are the API's *rejection* thresholds — a 400 mid-upload — not
    best-practice guidance. Raising here (before ``videos.insert``) is an
    upstream gate: it replaces the old silent ``title[:100]`` slice that
    desynced the state record (truncated) from the metadata files / CSV
    (full), and it catches over-cap descriptions and the ``<``/``>`` the API
    rejects before a batch 400s after every gate already passed.

    Limits enforced:
      - title: <= 100 chars, no ``<``/``>``
      - description: <= 5000 UTF-8 bytes, no ``<``/``>``
      - tags: combined length <= 500 chars
    """
    if len(title) > MAX_TITLE_LEN:
        raise ValueError(
            f"YouTube title too long: {len(title)} chars > {MAX_TITLE_LEN} "
            f"(API hard limit). Fix upstream metadata rather than truncating. "
            f"Title: {title!r}"
        )
    for field_name, value in (("title", title), ("description", description)):
        bad = [c for c in FORBIDDEN_SNIPPET_CHARS if c in value]
        if bad:
            raise ValueError(
                f"YouTube {field_name} contains characters the API rejects "
                f"({', '.join(bad)!r}). Remove '<' and '>' before upload."
            )
    desc_bytes = len(description.encode("utf-8"))
    if desc_bytes > MAX_DESCRIPTION_BYTES:
        raise ValueError(
            f"YouTube description too long: {desc_bytes} bytes > "
            f"{MAX_DESCRIPTION_BYTES} (API hard limit)."
        )
    tags_len = sum(len(t) for t in tags)
    if tags_len > MAX_TAGS_TOTAL_LEN:
        raise ValueError(
            f"YouTube tags too long: combined {tags_len} chars > "
            f"{MAX_TAGS_TOTAL_LEN} (API hard limit)."
        )


def _enforce_youtube_upload_media_limits(video_path: Path) -> None:
    """Backstop: refuse anything at/over YouTube's published hard caps —
    "256 GB or 12 hours, whichever is less". Unreachable given the
    superlong planner's 4h-per-part cap — a firing check means an
    upstream invariant broke, not a routine gate to relax.
    """
    size = video_path.stat().st_size
    if size >= MAX_UPLOAD_FILE_SIZE_BYTES:
        raise ValueError(
            f"YouTube upload refused: {video_path.name} is {size} bytes >= "
            f"{MAX_UPLOAD_FILE_SIZE_BYTES} (256 GB cap) — unreachable given "
            f"the superlong 4h part cap; treat as an upstream bug."
        )
    import subprocess as _subprocess  # lazy: matches phase_bc_day_first's pattern

    from ..validation.ffprobe_client import FFprobeClient

    duration = FFprobeClient(_subprocess).probe_duration(video_path)
    if duration >= MAX_UPLOAD_DURATION_SEC:
        raise ValueError(
            f"YouTube upload refused: {video_path.name} duration="
            f"{duration:.1f}s >= {MAX_UPLOAD_DURATION_SEC} (12h cap) — "
            f"unreachable given the superlong 4h part cap; treat as an "
            f"upstream bug."
        )


def _build_credentials_obj(creds_dict: dict) -> Credentials:
    """Construct a google Credentials object from a loaded creds dict."""
    return Credentials(
        token=creds_dict.get("token"),
        refresh_token=creds_dict.get("refresh_token"),
        token_uri=creds_dict.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=creds_dict.get("client_id"),
        client_secret=creds_dict.get("client_secret"),
        scopes=creds_dict.get("scopes", [YOUTUBE_UPLOAD_SCOPE]),
    )


class YouTubeUploader:
    """Upload long-form videos to YouTube via the Data API v3."""

    def __init__(self, credentials_dir: Optional[Path] = None):
        self.credentials_dir = credentials_dir or DEFAULT_CREDENTIALS_DIR
        self._creds_dict: Optional[dict] = None
        self._credentials: Optional[Credentials] = None
        self._service: Any = None
        # Per-run cache: playlist title → playlist id. Populated on first
        # ``_load_playlist_cache`` call by a single ``playlists.list``;
        # subsequent uploads in the same uploader instance hit this map
        # so ``playlists.list`` runs at most once per run.
        self._playlist_cache: Optional[dict[str, str]] = None
        # Operator-checklist sidecar path. Populated by
        # ``set_operator_checklist_path`` from Phase D once
        # ``output_base`` is known; ``upload_video_with_playlist``
        # appends one line per upload there. ``None`` means callers that
        # haven't opted in (older test mocks) skip the file write — the
        # logger line still fires.
        self._operator_checklist_path: Optional[Path] = None

    def _load_credentials(self) -> dict:
        """Load credentials dict (in-memory only; never written to disk)."""
        if self._creds_dict is None:
            self._creds_dict = load_youtube_credentials(self.credentials_dir)
        return self._creds_dict

    def _build_service(self) -> Any:
        """Build (or return cached) YouTube API service."""
        if self._service is not None:
            return self._service
        creds_dict = self._load_credentials()
        self._credentials = _build_credentials_obj(creds_dict)
        self._service = build(
            YOUTUBE_API_SERVICE_NAME,
            YOUTUBE_API_VERSION,
            credentials=self._credentials,
            cache_discovery=False,
        )
        return self._service

    # ------------------------------------------------------------------ upload

    def upload_video(
        self,
        video_path: Path,
        *,
        title: str,
        description: str,
        tags: list[str],
        category_id: str = "2",
        publish_at: Optional[datetime] = None,
        privacy: str = "private",
        recording_date: Optional[str] = None,
        contains_synthetic_media: Optional[bool] = None,
    ) -> str:
        """
        Upload a video file and return the YouTube video ID.

        Raises ``ValueError`` (via ``_enforce_youtube_api_limits``) if the
        title exceeds 100 chars, the description exceeds 5000 bytes, the tags
        exceed 500 combined chars, or title/description contain ``<``/``>`` —
        the API's hard-rejection limits — rather than silently truncating.
        If `publish_at` is provided, schedules the video (privacyStatus=private
        + publishAt=ISO8601). If omitted, uses `privacy` as-is.

        ``recording_date`` (optional) populates the ``recordingDetails.recordingDate``
        API field — the filmed-on date YouTube surfaces under "More info" on the
        watch page. Accepts ``"YYYY-MM-DD"`` (normalized to ISO 8601 UTC midnight
        ``"YYYY-MM-DDT00:00:00.000Z"`` per Data API v3 requirements) or a
        full ISO 8601 UTC timestamp (passed through verbatim). Garbage strings
        raise ``ValueError`` at the boundary rather than silently uploading
        and 400-ing hours later.

        Rejects any `video_path` whose basename does not match
        ``config.drive.youtube_upload_filename_pattern`` (default
        ``^(?:lf|sl|short)_\\d{2}\\.mp4$`` — canonical LF, superlong-part, and
        short output). The pattern is read lazily via ``get_config`` so dev
        overrides take effect per-invocation. See the module docstring
        for rationale (legacy "episode NN FINAL" draft prevention).

        Every upload also sends ``snippet.defaultLanguage``, ``status.license``,
        and ``status.embeddable`` (config-backed via ``config.drive.youtube_default_language``,
        ``youtube_license``, ``youtube_embeddable`` — see the YouTube Data API v3
        ``videos.insert`` reference).
        ``contains_synthetic_media`` (``status.containsSyntheticMedia``) is sent
        only when explicitly ``True``/``False`` (caller kwarg, falling back to
        ``config.drive.youtube_contains_synthetic_media``); ``None`` omits the
        field entirely rather than sending a guessed value.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Filename guard — read pattern at call time so dev overrides
        # via get_config take effect per-invocation, not just at import time.
        pattern = get_config().drive.youtube_upload_filename_pattern
        if not re.match(pattern, video_path.name):
            raise ValueError(
                f"YouTube upload refused: filename {video_path.name!r} "
                f"does not match YOUTUBE_UPLOAD_FILENAME_PATTERN "
                f"(pattern={pattern!r}). Only canonical LF-assembly, superlong-part, "
                f"and short outputs (e.g. 'lf_01.mp4', 'sl_01.mp4', 'short_01.mp4') "
                f"may be uploaded. Adjust drive.youtube_upload_filename_pattern "
                f"in config.yaml to override for dev/testing."
            )

        # Backstop: duration/size hard caps (see docstring above).
        _enforce_youtube_upload_media_limits(video_path)

        # Upstream API-limits gate: raise on over-cap title/description/tags
        # or forbidden ``<``/``>`` rather than silently truncating (which
        # desyncs the state record from the metadata files) or 400-ing
        # mid-batch after every gate already passed.
        _enforce_youtube_api_limits(title, description, list(tags))

        service = self._build_service()
        drive_config = get_config().drive

        snippet = {
            "title": title,
            "description": description,
            "tags": list(tags),
            "categoryId": category_id,
            "defaultLanguage": drive_config.youtube_default_language,
        }
        status: dict[str, Any] = {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
            "license": drive_config.youtube_license,
            "embeddable": drive_config.youtube_embeddable,
        }
        if publish_at is not None:
            if publish_at.tzinfo is None:
                publish_at = publish_at.replace(tzinfo=timezone.utc)
            # Scheduled uploads MUST be private prior to publishAt
            status["privacyStatus"] = "private"
            status["publishAt"] = publish_at.isoformat().replace("+00:00", "Z")

        # contains_synthetic_media: caller kwarg wins, else config default;
        # None (either level) omits the field rather than guessing a value.
        synthetic_media = (
            contains_synthetic_media
            if contains_synthetic_media is not None
            else drive_config.youtube_contains_synthetic_media
        )
        if synthetic_media is not None:
            status["containsSyntheticMedia"] = synthetic_media

        body: dict[str, Any] = {"snippet": snippet, "status": status}
        part = "snippet,status"

        if recording_date is not None:
            body["recordingDetails"] = {"recordingDate": _normalize_recording_date(recording_date)}
            part = "snippet,status,recordingDetails"

        media = MediaFileUpload(
            str(video_path),
            chunksize=256 * 1024 * 1024,
            resumable=True,
            mimetype="video/*",
        )

        request = service.videos().insert(
            part=part,
            body=body,
            media_body=media,
        )

        video_id = self._execute_resumable(request)
        logger.info(f"YouTube upload complete: {video_id} ({video_path.name})")
        return video_id

    def _refresh_oauth_token(self) -> None:
        """Refresh the in-memory OAuth credentials once.

        Called from the 401 path in ``_execute_resumable``. Uses
        ``google.auth.transport.requests.Request`` to drive
        ``Credentials.refresh`` — the standard google-auth refresh path.
        Token state is held in memory only; we never persist a
        half-refreshed creds dict to disk on either success or failure.

        Refresh-side failures (refresh raises) are
        distinguishable from a post-refresh 401 (genuine revocation) in
        the caller's log line. The caller logs the loud failure message
        — this method just propagates whatever ``refresh`` raises.
        """
        if self._credentials is None:
            # Service was never built (shouldn't happen — _execute_resumable
            # only runs after _build_service populates this). Defensive.
            raise RuntimeError("Cannot refresh OAuth token: credentials not initialised")
        self._credentials.refresh(GoogleAuthRequest())

    def _execute_resumable(self, request: Any) -> str:
        """Drive a resumable upload to completion with bounded retry.

        Uses exponential backoff with jitter: delay * (0.5 + random()).
        Permanent errors (403/404) fail immediately without retry.

        HTTP 401: attempt ONE OAuth-token refresh via
        ``_refresh_oauth_token`` and retry the chunk once. If the retry
        also returns 401, fail-permanent (genuine revocation). If the
        refresh itself raises, log loudly (refresh-side failure) and
        fail-permanent. Refresh-and-retry must NOT loop — guarded by the
        ``did_401_refresh`` flag.
        """
        response: Optional[dict] = None
        attempt = 0
        backoff = INITIAL_BACKOFF_SEC
        did_401_refresh = False

        while response is None:
            attempt += 1
            try:
                status, response = self._next_chunk_with_timeout(request)
                if status is not None:
                    logger.debug(f"YouTube upload progress: {status.progress() * 100:.1f}%")
            except RefreshError:
                # Hard auth failure — do not retry
                logger.error("YouTube credentials refresh failed — re-auth required")
                raise
            except HttpError as e:
                code = getattr(getattr(e, "resp", None), "status", None)
                # Quota / rate-limit disambiguation. HTTP 403 is used by the
                # YouTube Data API v3 for BOTH quota exhaustion (recover
                # by waiting for the window to reset) and true auth
                # failures (require credential rotation). Inspect
                # ``error_details`` for a quota-class reason; if present,
                # raise ``YouTubeQuotaError`` so Phase D's outer loop can
                # banner the recovery steps and exit cleanly for resume.
                # Non-quota 403s fall through to the original permanent
                # path unchanged — keeps legit auth-failure semantics.
                if code == 403:
                    quota_reason = _extract_quota_reason(e)
                    if quota_reason is not None:
                        logger.error(
                            "YouTube quota/rate-limit error (reason=%s) — "
                            "raising YouTubeQuotaError for clean Phase D "
                            "exit; upload is resumable after the quota "
                            "window resets",
                            quota_reason,
                        )
                        raise YouTubeQuotaError(
                            reason=quota_reason,
                            status_code=403,
                            http_error=e,
                        ) from e
                # HTTP 401 — refresh-and-retry once.
                # Mid-flight token expiry is the dominant 401 cause for
                # long-running resumable uploads. Refresh the creds, retry
                # the chunk exactly once. A second 401 means the token is
                # truly revoked; a refresh that itself raises means the
                # refresh path is broken (logged distinctly so operators
                # can tell them apart).
                if code == 401 and not did_401_refresh:
                    did_401_refresh = True
                    logger.warning(
                        "YouTube HTTP 401 — attempting OAuth token refresh and one retry"
                    )
                    try:
                        self._refresh_oauth_token()
                    except Exception as refresh_exc:  # noqa: BLE001
                        # Refresh-side failure — distinct log shape so it
                        # is grep-distinguishable from a post-refresh 401.
                        logger.error(
                            "YouTube OAuth token refresh FAILED (%s: %s) — "
                            "cannot retry upload; failing permanent. This "
                            "is a refresh-side failure, NOT a revoked-token "
                            "case (which would surface as a second 401 "
                            "after a successful refresh).",
                            type(refresh_exc).__name__,
                            refresh_exc,
                        )
                        raise e from refresh_exc
                    logger.info(
                        "YouTube OAuth token refresh succeeded — retrying upload chunk once"
                    )
                    continue
                # Permanent errors — fail immediately
                if code in PERMANENT_STATUS_CODES:
                    logger.error(f"YouTube permanent error HTTP {code} — not retrying")
                    raise
                if code in RETRIABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    jitter = 0.5 + random.random()  # [0.5, 1.5)
                    wait = backoff * jitter
                    logger.warning(
                        f"YouTube HTTP {code} — retry {attempt}/{MAX_RETRIES} after {wait:.1f}s"
                    )
                    time.sleep(wait)
                    backoff *= 2
                    continue
                raise
            except TimeoutError:
                if attempt >= MAX_RETRIES:
                    logger.error("YouTube chunk upload timed out after %d attempts", attempt)
                    raise
                jitter = 0.5 + random.random()
                wait = backoff * jitter
                logger.warning(
                    "YouTube chunk timed out (%ds) — retry %d/%d after %.1fs",
                    CHUNK_TIMEOUT_SEC,
                    attempt,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                backoff *= 2
            except (OSError, ConnectionError) as e:
                if attempt >= MAX_RETRIES:
                    logger.error(f"YouTube upload failed after {attempt} attempts: {e}")
                    raise
                jitter = 0.5 + random.random()
                wait = backoff * jitter
                logger.warning(
                    f"YouTube network error ({e}) — retry {attempt}/{MAX_RETRIES} after {wait:.1f}s"
                )
                time.sleep(wait)
                backoff *= 2

        if not isinstance(response, dict) or "id" not in response:
            raise RuntimeError(f"YouTube upload returned no video id: {response!r}")
        return response["id"]

    @staticmethod
    def _next_chunk_with_timeout(request: Any) -> tuple:
        """Call request.next_chunk() with a per-chunk timeout.

        Uses an explicit (not ``with``-managed) pool: the ``with`` form's
        ``__exit__`` calls ``shutdown(wait=True)``, which on TimeoutError
        blocks the caller until the orphaned ``next_chunk()`` worker
        returns — defeating the timeout entirely. On timeout we
        ``shutdown(wait=False, cancel_futures=True)`` so the caller is
        freed immediately; the daemon worker (if it had not started) is
        cancelled, and any in-flight socket is abandoned. The success path
        shuts the pool down normally so threads are not leaked.
        """
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(request.next_chunk)
        try:
            result = future.result(timeout=CHUNK_TIMEOUT_SEC)
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(
                f"YouTube chunk upload exceeded {CHUNK_TIMEOUT_SEC}s timeout"
            ) from None
        pool.shutdown(wait=True)
        return result

    def _execute_with_retry(self, request: Any) -> Any:
        """Run ``request.execute()`` with the chunk-path 5xx retry policy.

        Ancillary calls (``set_thumbnail`` / ``update_privacy``) fire AFTER
        the video is published, so a transient 5xx there strands the video
        with no thumbnail / wrong privacy. Reuses the same backoff as
        the resumable-upload path: retriable status codes back off with
        jitter up to ``MAX_RETRIES``; permanent codes (and any non-retriable
        status) raise immediately.
        """
        attempt = 0
        backoff = INITIAL_BACKOFF_SEC
        while True:
            attempt += 1
            try:
                return request.execute()
            except HttpError as e:
                code = getattr(getattr(e, "resp", None), "status", None)
                if code in RETRIABLE_STATUS_CODES and attempt < MAX_RETRIES:
                    wait = backoff * (0.5 + random.random())  # jitter [0.5, 1.5)
                    logger.warning(
                        "YouTube ancillary HTTP %s — retry %d/%d after %.1fs",
                        code,
                        attempt,
                        MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    backoff *= 2
                    continue
                raise

    # --------------------------------------------------------------- ancillary

    def set_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        """Attach a custom thumbnail to a video."""
        thumbnail_path = Path(thumbnail_path)
        if not thumbnail_path.exists():
            raise FileNotFoundError(f"Thumbnail not found: {thumbnail_path}")
        service = self._build_service()
        media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
        self._execute_with_retry(service.thumbnails().set(videoId=video_id, media_body=media))
        logger.info(f"Thumbnail set for {video_id}")

    def update_privacy(
        self,
        video_id: str,
        *,
        privacy: str,
        publish_at: Optional[datetime] = None,
    ) -> None:
        """
        Update privacy / schedule of an existing video.

        Pass privacy="unlisted" to drop a scheduled video off the publish queue.
        """
        service = self._build_service()
        status: dict[str, Any] = {"privacyStatus": privacy}
        if publish_at is not None:
            if publish_at.tzinfo is None:
                publish_at = publish_at.replace(tzinfo=timezone.utc)
            status["privacyStatus"] = "private"
            status["publishAt"] = publish_at.isoformat().replace("+00:00", "Z")

        self._execute_with_retry(
            service.videos().update(
                part="status",
                body={"id": video_id, "status": status},
            )
        )
        logger.info(f"Updated privacy for {video_id} -> {status['privacyStatus']}")

    # ----------------------------------------------------------- playlist op

    # ------------------------------------------------------------- end-screen
    def set_operator_checklist_path(self, path: Path) -> None:
        """Configure the per-batch operator checklist sidecar.

        Phase D calls this once after resolving ``output_base`` so
        every subsequent ``upload_video_with_playlist`` appends an
        end-screen instruction line to ``<output_base>/operator_checklist.txt``.
        """
        self._operator_checklist_path = Path(path)

    def _emit_endscreen_checklist(
        self, *, video_id: str, reference_video_id: Optional[str]
    ) -> None:
        """Format and emit the end-screen operator checklist line.

        Mirrors the playlist assignment's "total by construction" pattern in
        upload_video_with_playlist: every playlist-assigning upload path
        (upload_video_with_playlist and the playlist branch of
        upload_if_needed) runs through here. The line is logged at INFO and (when
        ``set_operator_checklist_path`` was called) appended to the
        per-batch sidecar file. File-write failures are logged but
        never fail the upload — the logger line is the durable record.
        """
        line = format_endscreen_checklist_line(
            video_id=video_id, reference_video_id=reference_video_id
        )
        logger.info(line)
        path = self._operator_checklist_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            logger.warning(
                "operator_checklist.txt append failed (path=%s): %s — "
                "the INFO log line above is the durable record",
                path,
                exc,
            )

    def _load_playlist_cache(self) -> dict[str, str]:
        """Populate (once) and return the title→id playlist cache.

        Calls ``playlists.list(mine=True, maxResults=50)`` exactly once
        per uploader instance. Subsequent calls return the cached map.
        Construction guarantee for "one playlists.list per run".
        """
        if self._playlist_cache is not None:
            return self._playlist_cache
        service = self._build_service()
        resp = service.playlists().list(part="snippet", mine=True, maxResults=50).execute()
        cache: dict[str, str] = {}
        for item in resp.get("items", []) or []:
            title = (item.get("snippet") or {}).get("title")
            pid = item.get("id")
            if isinstance(title, str) and isinstance(pid, str):
                cache[title] = pid
        self._playlist_cache = cache
        return cache

    def _get_or_create_playlist(self, name: str) -> str:
        """Return the id of the playlist titled ``name``, creating if absent.

        Idempotent: ``playlists.list`` runs at most once per uploader
        instance (cache); ``playlists.insert`` only fires when no
        existing playlist matches by exact title. Default privacy
        ``public`` mirrors the channel-default for LF uploads (LFs are
        scheduled-private until publish_at fires; the playlist itself is
        public so embeds and direct links resolve cleanly).
        """
        cache = self._load_playlist_cache()
        if name in cache:
            return cache[name]
        service = self._build_service()
        resp = (
            service.playlists()
            .insert(
                part="snippet,status",
                body={
                    "snippet": {"title": name},
                    "status": {"privacyStatus": "public"},
                },
            )
            .execute()
        )
        pid = resp.get("id")
        if not isinstance(pid, str):
            raise RuntimeError(f"playlists.insert returned no id: {resp!r}")
        cache[name] = pid
        return pid

    def _add_video_to_playlist(self, video_id: str, playlist_id: str) -> None:
        """Insert ``video_id`` into ``playlist_id`` exactly once."""
        service = self._build_service()
        service.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()

    def assign_video_to_resolved_playlist(
        self,
        video_id: str,
        *,
        metadata: dict,
        config: Any,
    ) -> str:
        """Resolve and assign ``video_id`` to its playlist; return playlist_id.

        Extracted from ``upload_video_with_playlist`` so resume
        can recover an orphaned upload — a video whose ``videos.insert``
        succeeded but whose ``playlistItems.insert`` failed — without
        re-uploading. Idempotency of ``playlistItems.insert`` itself is
        managed by the caller via the persisted ``playlist_id`` on the
        upload manifest record.
        """
        playlist_name = resolve_playlist(metadata, config)
        playlist_id = self._get_or_create_playlist(playlist_name)
        self._add_video_to_playlist(video_id, playlist_id)
        logger.info("Added %s to playlist %r (id=%s)", video_id, playlist_name, playlist_id)
        return playlist_id

    def upload_video_with_playlist(
        self,
        video_path: Path,
        *,
        title: str,
        description: str,
        tags: list[str],
        metadata: dict,
        config: Any,
        category_id: str = "2",
        publish_at: Optional[datetime] = None,
        privacy: str = "private",
        recording_date: Optional[str] = None,
    ) -> str:
        """Upload + assign to the resolved playlist.

        Total by construction: ``resolve_playlist`` returns the
        config-defined ``default`` for any input that no rule matches,
        so every uploaded video gets exactly one
        ``playlistItems.insert``. There is no path through this method
        that uploads without playlist assignment.

        Returns the ``video_id``. If ``playlistItems.insert`` fails, the
        exception propagates with ``video_id`` lost from this call site
        — callers should prefer ``upload_if_needed`` which persists
        ``video_id`` before the playlist insert so resume can recover.
        For callers that need both ids in-process, use the lower-level
        ``upload_video`` + ``assign_video_to_resolved_playlist`` pair
        directly.
        """
        video_id = self.upload_video(
            video_path,
            title=title,
            description=description,
            tags=tags,
            category_id=category_id,
            publish_at=publish_at,
            privacy=privacy,
            recording_date=recording_date,
        )
        self.assign_video_to_resolved_playlist(video_id, metadata=metadata, config=config)
        # Emit the operator end-screen checklist line for this
        # video. No API call — Studio end screens have no documented
        # public API.
        endscreen_cfg = getattr(getattr(config, "drive", None), "endscreen", None)
        ref_id = getattr(endscreen_cfg, "reference_video_id", None)
        self._emit_endscreen_checklist(video_id=video_id, reference_video_id=ref_id)
        return video_id

    # ---------------------------------------------------------- state-aware op

    def upload_if_needed(
        self,
        *,
        state: dict,
        lf_idx: int,
        video_path: Path,
        title: str,
        description: str,
        tags: list[str],
        category_id: str = "2",
        publish_at: Optional[datetime] = None,
        privacy: str = "private",
        metadata: Optional[dict] = None,
        config: Any = None,
        recording_date: Optional[str] = None,
        persist: Optional[Callable[[], None]] = None,
    ) -> dict:
        """
        Upload a long-form video unless state shows it is already uploaded.

        Reads/writes `state["youtube_uploads"][lf_idx]`. Returns the record
        (either existing or freshly created).

        Playlist auto-assignment: when both
        ``metadata`` and ``config`` are provided, the upload routes
        through ``upload_video_with_playlist`` so the resolved playlist
        gets a ``playlistItems.insert``. When either is omitted (older
        callers / tests), falls back to plain ``upload_video`` — the
        contract is opt-in at this layer because Phase D is the canonical
        opt-in site.

        ``persist`` (optional) is a caller-supplied checkpoint flush,
        invoked the moment ``videos.insert`` returns and BEFORE the
        playlist assignment — see the call site below for why.
        """
        uploads = state.youtube_uploads
        if uploads is None:
            uploads = {}
            state.youtube_uploads = uploads
        existing = uploads.get(lf_idx)
        playlist_expected = metadata is not None and config is not None
        if existing and existing.get("video_id"):
            # Detect the orphan-upload case — a record where
            # ``videos.insert`` succeeded but ``playlistItems.insert``
            # never persisted a ``playlist_id``. Re-run only the playlist
            # assignment; do NOT re-upload the video. Preserves all other
            # fields on the record so resume is non-destructive.
            if playlist_expected and not existing.get("playlist_id"):
                logger.warning(
                    "LF %s has video_id=%s but no playlist_id — re-running "
                    "playlistItems.insert only (orphan recovery)",
                    lf_idx,
                    existing["video_id"],
                )
                playlist_id = self.assign_video_to_resolved_playlist(
                    existing["video_id"], metadata=metadata, config=config
                )
                existing["playlist_id"] = playlist_id
                uploads[lf_idx] = existing
                return existing
            logger.info(f"LF {lf_idx} already uploaded as {existing['video_id']} — skipping")
            return existing

        if playlist_expected:
            # Split upload from playlist assignment so we can
            # persist ``video_id`` BEFORE attempting the playlist insert.
            # If the playlist insert then fails, resume sees video_id
            # present + playlist_id missing and recovers cleanly above.
            video_id = self.upload_video(
                video_path,
                title=title,
                description=description,
                tags=tags,
                category_id=category_id,
                publish_at=publish_at,
                privacy=privacy,
                recording_date=recording_date,
            )
            record = {
                "video_id": video_id,
                "url": f"https://youtu.be/{video_id}",
                "publish_at": publish_at.isoformat() if publish_at else None,
                "title": title[:MAX_TITLE_LEN],
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            uploads[lf_idx] = record
            # ``videos.insert`` above is IRREVERSIBLE and the video is now
            # live on YouTube, while everything after this line is fallible
            # and recoverable (the playlist insert can fail on its own, for
            # example with an HTTP 409). Flush the checkpoint HERE so no crash,
            # SIGTERM or swallowed error between the two API calls can lose the
            # video_id: losing it makes resume re-run ``videos.insert`` for
            # content already published. The record deliberately carries no
            # ``playlist_id`` yet — that orphan signature is exactly what
            # resume's playlist-only recovery branch above keys on.
            if persist is not None:
                persist()
            playlist_id = self.assign_video_to_resolved_playlist(
                video_id, metadata=metadata, config=config
            )
            record["playlist_id"] = playlist_id
            # Emit the operator end-screen checklist line.
            endscreen_cfg = getattr(getattr(config, "drive", None), "endscreen", None)
            ref_id = getattr(endscreen_cfg, "reference_video_id", None)
            self._emit_endscreen_checklist(video_id=video_id, reference_video_id=ref_id)
        else:
            video_id = self.upload_video(
                video_path,
                title=title,
                description=description,
                tags=tags,
                category_id=category_id,
                publish_at=publish_at,
                privacy=privacy,
                recording_date=recording_date,
            )
            record = {
                "video_id": video_id,
                "url": f"https://youtu.be/{video_id}",
                "publish_at": publish_at.isoformat() if publish_at else None,
                "title": title[:MAX_TITLE_LEN],
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
            }
            uploads[lf_idx] = record
        return record
