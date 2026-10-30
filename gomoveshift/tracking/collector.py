"""
Metrics Collector — pulls performance data from social media APIs.

Implements four collectors: YouTube (Data API v3 + Analytics), TikTok
(Display API ``video/query``), Instagram and Facebook (Graph API). The
"YouTube only, extensible to the rest" note this line used to carry was
left behind by those additions.

A missing metric stays ``None`` ("not reported by this platform") and is
never written as a measured 0 -- see ``schema.PlatformMetrics``. That
distinction only holds if callers preserve it: reporting an absence as a
collection, a failure, or a zero all destroy it.

YouTube credentials are loaded from the SEO service's credential store by default.
If credentials are missing the collector degrades gracefully — you can
still register shorts and collect metrics manually or when creds are set up.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional
from zoneinfo import ZoneInfo

from gomoveshift.publishing.meta_auth import DEFAULT_GRAPH_VERSION, load_meta_credentials
from gomoveshift.publishing.tiktok_auth import TikTokAuthError, get_valid_tiktok_credentials

from .day7 import DAY7_DAYS, has_settled_series, posted_date, window_settled
from .observation import _UPLOAD_HANDLE_PREFIX, MetricObservation, ObservationStatus, has_numbers
from .schema import (
    LONG_FORM_DURATION_SOURCE_YOUTUBE,
    DailyMetricPoint,
    PlatformMetrics,
    ShortRecord,
    TrackingRecord,
)
from .store import append_observation

logger = logging.getLogger(__name__)


def _graph_failure_status(status_code: int, message: str) -> str:
    """Which kind of "no numbers" a Graph API refusal is.

    A scope denial, a throttle and a deleted post are three different facts
    about the world; collapsing them into one empty result is what makes a
    permissions problem read downstream as a post nobody watched.
    """
    text = (message or "").lower()
    if status_code == 429 or "rate limit" in text or "too many calls" in text:
        return ObservationStatus.RATE_LIMITED
    if status_code in (401, 403) or "scope" in text or "permission" in text:
        return ObservationStatus.PERMISSION_DENIED
    if status_code == 404 or "does not exist" in text or "cannot be loaded" in text:
        return ObservationStatus.NOT_RETURNED
    return ObservationStatus.PROVIDER_ERROR


def _answered_observation(
    base: Dict[str, Any], metrics: PlatformMetrics, failure: Optional[tuple]
) -> MetricObservation:
    """Turn a completed Graph request into an observation: the recorded
    refusal when Graph declined, a measurement when it returned numbers, and
    NOT_RETURNED when it answered about nothing -- three outcomes the old
    all-``None`` :class:`PlatformMetrics` could not tell apart."""
    if failure:
        status, detail = failure
        return MetricObservation(status=status, detail=detail, **base)
    if not has_numbers(metrics):
        return MetricObservation(
            status=ObservationStatus.NOT_RETURNED,
            detail="queried; the platform reported no values for this post",
            **base,
        )
    return MetricObservation(status=ObservationStatus.MEASURED, cumulative=metrics, **base)


#: IG renamed the Reels play-count metric from "plays" to "views" at v20.0.
_IG_VIEWS_METRIC_CUTOVER_MAJOR = 20.0

_TIKTOK_QUERY_URL = "https://open.tiktokapis.com/v2/video/query/"
_TIKTOK_QUERY_FIELDS = "id,view_count,like_count,comment_count,share_count"
_TIKTOK_MAX_IDS_PER_REQUEST = 20  # per-request id cap; larger batches are chunked

#: TikTok ``video/query`` error codes mapped to WHY an observation carries no
#: numbers. Anything unlisted is a provider error -- never silence.
_TIKTOK_ERROR_STATUS = {
    "rate_limit_exceeded": ObservationStatus.RATE_LIMITED,
    "scope_not_authorized": ObservationStatus.PERMISSION_DENIED,
    "access_token_invalid": ObservationStatus.PERMISSION_DENIED,
    "access_token_expired": ObservationStatus.PERMISSION_DENIED,
}


def _reported_metric(value: Optional[Any]) -> Optional[int]:
    """``int(value)`` when the platform actually returned this key; ``None``
    when it didn't. ``dict.get(key, 0)`` can't distinguish "absent" from "a
    reported 0" — see the ``Optional[...]`` contract on ``PlatformMetrics``
    in schema.py, which this exists to honor."""
    return None if value is None else int(value)


def _looks_like_upload_handle(video_id: str) -> bool:
    """``v_inbox_file~v2.<digits>`` upload handles (``platform_upload_id``
    for `awaiting_manual_publish` rows) aren't queryable video ids — a
    real one is a plain numeric ``platform_post_id`` string."""
    return not video_id.isdigit()


# Default credential paths — derived from config.yaml seo section when available
def _resolve_default_cred_paths():
    """Resolve credential paths from the LOADED config, not SEOConfig defaults.

    `SEOConfig` defaults `manager_path` to "." deliberately -- every concrete
    deployment overrides it in config.yaml, and the default exists only so the
    model constructs. Reading the default instead of the loaded value made both
    paths cwd-relative, so the token resolved to whatever directory the process
    happened to start in and was never found.
    """
    try:
        from gomoveshift.config import get_config

        manager_dir = Path(get_config().seo.manager_path).resolve()
    except Exception:  # config unreadable in a bare import context
        manager_dir = Path(".").resolve()
    return manager_dir / "client_secret.json", manager_dir / "youtube_credentials.json"


def _load_shared_youtube_credentials(credentials_dir=None, brand=None) -> dict:
    """The one loader every other YouTube consumer uses (`_yt_creds`), imported
    lazily so importing this module never shells out to SOPS."""
    from gomoveshift.pipeline._yt_creds import load_youtube_credentials

    return load_youtube_credentials(credentials_dir, brand=brand)


_DEFAULT_CLIENT_SECRET, _DEFAULT_YOUTUBE_TOKEN = _resolve_default_cred_paths()


ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"


def daily_points(rows: List[Dict[str, Any]]) -> List[DailyMetricPoint]:
    """Analytics daily rows -> ``DailyMetricPoint``s."""
    return [
        DailyMetricPoint(
            day=row["day"],
            views=row.get("views"),
            # Absent (not 0-defaulted) means this day's column
            # didn't come back from the API — unreported, the
            # same distinction PlatformMetrics makes above.
            likes=row.get("likes"),
            shares=row.get("shares"),
            avg_view_duration_seconds=row.get("avg_view_duration_seconds"),
            watch_time_seconds=(
                row["watch_time_minutes"] * 60 if "watch_time_minutes" in row else None
            ),
        )
        for row in rows
    ]


#: 403 reasons that mean "out of budget", not "no permission".
_QUOTA_REASONS = (
    "quotaExceeded",
    "rateLimitExceeded",
    "dailyLimitExceeded",
    "userRateLimitExceeded",
)


class AnalyticsStatus(NamedTuple):
    """Whether the YouTube Analytics API is usable for this collector."""

    status: Literal["ok", "scope_missing", "no_token", "error"]
    detail: str = ""

    def line(self) -> str:
        """The single operator-facing reason line (``ANALYTICS_OK`` when fine)."""
        if self.status == "ok":
            return "ANALYTICS_OK"
        if self.status == "scope_missing":
            return (
                "ANALYTICS_SCOPE_MISSING scope=yt-analytics.readonly "
                "action=run 'gomoveshift youtube-auth'"
            )
        if self.status == "no_token":
            return "ANALYTICS_NO_TOKEN action=run 'gomoveshift youtube-auth'"
        return f"ANALYTICS_UNAVAILABLE reason={self.detail}"


class YouTubeCollector:
    """
    Collects metrics for YouTube Shorts via the YouTube Data API v3.

    Uses OAuth2 credentials from the SEO service's credential store. Falls back to
    API-key mode if available, or raises a clear error if neither is set up.
    """

    SCOPES = [
        "https://www.googleapis.com/auth/youtube.readonly",
        "https://www.googleapis.com/auth/yt-analytics.readonly",
    ]

    def __init__(
        self,
        client_secret_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
        api_key: Optional[str] = None,
        brand: Optional[str] = None,
    ):
        self._client_secret = (
            Path(client_secret_path) if client_secret_path else _DEFAULT_CLIENT_SECRET
        )
        self._token_path = Path(token_path) if token_path else _DEFAULT_YOUTUBE_TOKEN
        self._api_key = api_key
        self._brand = brand
        self._youtube = None  # Lazy-initialised API client
        self._shared_creds = None  # SOPS store, decrypted at most once
        self._shared_tried = False
        self._analytics_status: Optional[AnalyticsStatus] = None
        self._quota_exhausted = False
        self._analytics_line_logged = False

    # ── Auth ─────────────────────────────────────────────────────────────

    def _shared_credentials(self):
        """The SOPS store, decrypted at most once per collector.

        `is_available()` runs on every collection pass and SOPS decrypt is a
        subprocess, so the miss is cached too -- an install with no store must
        not re-shell on each check.
        """
        if not self._shared_tried:
            self._shared_tried = True
            try:
                self._shared_creds = _load_shared_youtube_credentials(brand=self._brand)
            except Exception as exc:  # absent store, failed decrypt, no sops
                logger.info("YouTube credentials not configured: %s", exc)
                self._shared_creds = None
        return self._shared_creds

    def _get_service(self):
        """Build or return the cached YouTube API client."""
        if self._youtube is not None:
            return self._youtube

        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
        except ImportError:
            raise RuntimeError(
                "google-api-python-client, google-auth-oauthlib required. "
                "Install with: pip install google-api-python-client google-auth-oauthlib"
            )

        creds = None

        # Try loading saved token; fall back to the shared SOPS store, which is
        # what a real deployment actually has (youtube_upload_token.json.enc).
        token_data = None
        if self._token_path.exists():
            try:
                token_data = json.loads(self._token_path.read_text())
            except Exception as exc:
                logger.warning("Failed to load YouTube token: %s", exc)
                token_data = None
        if token_data is None:
            token_data = self._shared_credentials()
        if token_data is not None:
            try:
                creds = Credentials(
                    token=token_data.get("token"),
                    refresh_token=token_data.get("refresh_token"),
                    token_uri=token_data.get("token_uri"),
                    client_id=token_data.get("client_id"),
                    client_secret=token_data.get("client_secret"),
                    scopes=token_data.get("scopes", self.SCOPES),
                )
            except Exception as exc:
                logger.warning("Failed to load YouTube token: %s", exc)

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Persist refreshed token
                self._save_token(creds)
            except Exception as exc:
                logger.warning("Token refresh failed: %s — will try re-auth", exc)
                creds = None

        # Interactive auth as last resort
        if not creds or not creds.valid:
            if not self._client_secret.exists():
                if self._api_key:
                    self._youtube = build("youtube", "v3", developerKey=self._api_key)
                    return self._youtube
                raise RuntimeError(
                    f"No YouTube credentials found.\n"
                    f"  Expected client_secret at: {self._client_secret}\n"
                    f"  Or provide api_key= for read-only access.\n"
                    f"  See: https://console.cloud.google.com/apis/credentials"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(self._client_secret), self.SCOPES)
            creds = flow.run_local_server(port=0)
            self._save_token(creds)

        self._youtube = build("youtube", "v3", credentials=creds)
        return self._youtube

    def _save_token(self, creds) -> None:
        """Persist OAuth token for future runs."""
        token_data = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": list(creds.scopes or self.SCOPES),
        }
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._token_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(token_data, indent=2).encode())
        finally:
            os.close(fd)
        logger.info("Saved YouTube token to %s", self._token_path)

    # ── Data Collection ──────────────────────────────────────────────────

    def fetch_video_metrics(self, video_id: str) -> PlatformMetrics:
        """
        Fetch metrics for a single YouTube video/short.

        Args:
            video_id: YouTube video ID (the 11-char string, not full URL)

        Returns:
            PlatformMetrics with views, likes, comments, etc.
        """
        return self.fetch_video_metrics_and_duration(video_id)[0]

    def fetch_video_metrics_and_duration(
        self, video_id: str
    ) -> tuple[PlatformMetrics, Optional[float]]:
        """``fetch_video_metrics`` plus the video's duration in seconds, which
        the same ``contentDetails`` response already carries. The duration is
        None when the response omits it (or reports a non-positive one) --
        an unknown length is not a zero-length video.
        """
        yt = self._get_service()

        try:
            response = (
                yt.videos()
                .list(
                    part="statistics,contentDetails",
                    id=video_id,
                )
                .execute()
            )
        except Exception as exc:
            logger.error("YouTube API error for %s: %s", video_id, exc)
            return PlatformMetrics(), None

        items = response.get("items", [])
        if not items:
            logger.warning("Video %s not found on YouTube", video_id)
            return PlatformMetrics(), None

        stats = items[0].get("statistics", {})
        content = items[0].get("contentDetails", {})

        # Parse ISO 8601 duration (PT1M30S → 90 seconds)
        duration_iso = content.get("duration")
        duration_sec = self._parse_iso_duration(duration_iso) if duration_iso else 0.0

        return PlatformMetrics(
            views=_reported_metric(stats.get("viewCount")),
            # YouTube omits likeCount when a creator hides likes, and
            # commentCount when comments are disabled — an absent key here
            # is "unreported", not a measured zero.
            likes=_reported_metric(stats.get("likeCount")),
            comments=_reported_metric(stats.get("commentCount")),
            shares=None,  # Not available via Data API — unreported, not zero
        ), (duration_sec if duration_sec > 0 else None)

    def collect_observation(
        self,
        video_id: str,
        published_at: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> MetricObservation:
        """One video's metrics WITH the reason when there are none.

        ``fetch_video_metrics`` returns an all-``None`` ``PlatformMetrics``
        for "no credentials", "video not found" AND "the API call raised" --
        a caller cannot tell those apart. This is the same sibling the other
        three collectors already carry (``TikTokCollector.
        collect_observations_batch``, ``InstagramCollector.
        collect_observation``, ``FacebookCollector.collect_observation``);
        ``fetch_video_metrics``/``collect_short_metrics`` are unchanged for
        callers that only want the bare numbers.
        """
        base: Dict[str, Any] = {
            "platform": "youtube",
            "post_id": video_id,
            "account_id": account_id,
            "post_format": "short",
            "event_at": published_at,
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
        # A real YouTube video id is an 11-char alphanumeric string, not the
        # all-digit shape TikTok's ``_looks_like_upload_handle`` checks for --
        # that helper would misclassify every genuine YouTube id here. The
        # actual upload-handle shape is the ``v_inbox_file~`` prefix
        # ``observation.py`` already validates against.
        if str(video_id).startswith(_UPLOAD_HANDLE_PREFIX):
            return MetricObservation(
                status=ObservationStatus.UNQUERYABLE_ID,
                detail="upload handle, not a queryable public post id",
                **base,
            )
        if not self.is_available():
            return MetricObservation(
                status=ObservationStatus.NOT_CONFIGURED,
                detail="no YouTube credentials",
                **base,
            )
        yt = self._get_service()
        try:
            response = (
                yt.videos()
                .list(
                    part="statistics,contentDetails",
                    id=video_id,
                )
                .execute()
            )
        except Exception as exc:
            logger.error("YouTube API error for %s: %s", video_id, exc)
            return MetricObservation(
                status=ObservationStatus.PROVIDER_ERROR, detail=str(exc), **base
            )
        items = response.get("items", [])
        if not items:
            return MetricObservation(
                status=ObservationStatus.NOT_RETURNED,
                detail="queried; the platform reported no values for this post",
                **base,
            )
        stats = items[0].get("statistics", {})
        metrics = PlatformMetrics(
            views=_reported_metric(stats.get("viewCount")),
            likes=_reported_metric(stats.get("likeCount")),
            comments=_reported_metric(stats.get("commentCount")),
            shares=None,  # Not available via Data API -- unreported, not zero
        )
        if not has_numbers(metrics):
            return MetricObservation(
                status=ObservationStatus.NOT_RETURNED,
                detail="queried; the platform reported no values for this post",
                **base,
            )
        return MetricObservation(status=ObservationStatus.MEASURED, cumulative=metrics, **base)

    def _analytics_token_data(self):
        token_data = None
        if self._token_path.exists():
            token_data = json.loads(self._token_path.read_text())
        if not token_data:
            token_data = self._shared_credentials()
        return token_data

    def analytics_status(self) -> AnalyticsStatus:
        """Typed Analytics availability, computed once and cached.

        Decided from the credentials alone (no network): no token anywhere is
        ``no_token``; a scopes list lacking ``yt-analytics.readonly`` is
        ``scope_missing``. A later 403 insufficientPermissions during a fetch
        downgrades the cached value to ``scope_missing``.
        """
        if self._analytics_status is None:
            try:
                data = self._analytics_token_data()
            except Exception as exc:
                self._analytics_status = AnalyticsStatus("error", str(exc))
            else:
                if not data:
                    self._analytics_status = AnalyticsStatus(
                        "no_token", "no token file or shared credentials"
                    )
                elif ANALYTICS_SCOPE not in (data.get("scopes") or self.SCOPES):
                    self._analytics_status = AnalyticsStatus(
                        "scope_missing", "token lacks yt-analytics.readonly"
                    )
                else:
                    self._analytics_status = AnalyticsStatus("ok")
        return self._analytics_status

    def _log_analytics_status_once(self) -> None:
        if not self._analytics_line_logged:
            self._analytics_line_logged = True
            logger.warning(self.analytics_status().line())

    @property
    def quota_exhausted(self) -> bool:
        """True once an Analytics call hit a quota / rate limit (429, or 403
        with a quota reason) -- the rest of the run must stop asking."""
        return self._quota_exhausted

    def _note_analytics_exception(self, exc: Exception) -> None:
        """A 403 insufficientPermissions means the token lacks the scope;
        a 429 or a 403 quota reason means the budget is spent."""
        from googleapiclient.errors import HttpError

        if isinstance(exc, HttpError):
            status = getattr(exc.resp, "status", None)
            body = (exc.content or b"").decode("utf-8", "replace")
            if status == 429 or (status == 403 and any(r in body for r in _QUOTA_REASONS)):
                self._quota_exhausted = True

        if (
            isinstance(exc, HttpError)
            and getattr(exc.resp, "status", None) == 403
            and "insufficientPermissions" in (exc.content or b"").decode("utf-8", "replace")
        ):
            self._analytics_status = AnalyticsStatus("scope_missing", str(exc))
            self._log_analytics_status_once()

    def probe_analytics(self) -> AnalyticsStatus:
        """One tiny live query (last 2 days, views); returns the final status."""
        if self.analytics_status().status != "ok":
            return self.analytics_status()
        try:
            end = datetime.now()
            analytics = self._analytics_client()
            analytics.reports().query(
                ids="channel==MINE",
                startDate=(end - timedelta(days=2)).strftime("%Y-%m-%d"),
                endDate=end.strftime("%Y-%m-%d"),
                metrics="views",
            ).execute()
        except Exception as exc:
            self._note_analytics_exception(exc)
            if self.analytics_status().status == "ok":
                self._analytics_status = AnalyticsStatus("error", str(exc))
        return self.analytics_status()

    def _analytics_client(self):
        """Build a youtubeAnalytics v2 client, or None when status is not ok.

        Credential precedence mirrors `_get_service`: an explicit token file
        if it exists, else the shared SOPS store (decrypted at most once,
        in memory only). A non-ok status logs one reason line per collector.
        """
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        if self.analytics_status().status != "ok":
            self._log_analytics_status_once()
            return None
        token_data = self._analytics_token_data()

        creds = Credentials(
            token=token_data.get("token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=token_data.get("token_uri"),
            client_id=token_data.get("client_id"),
            client_secret=token_data.get("client_secret"),
            scopes=token_data.get("scopes", self.SCOPES),
        )
        return build("youtubeAnalytics", "v2", credentials=creds)

    def fetch_analytics_metrics(
        self, video_id: str, include_avg_view_percentage: bool = False
    ) -> Dict[str, Any]:
        """
        Fetch detailed analytics (watch time, avg view duration) via
        YouTube Analytics API. Requires yt-analytics.readonly scope.

        Returns raw analytics dict or empty dict on failure.
        """
        try:
            analytics = self._analytics_client()
            if analytics is None:
                return {}

            # Query last 90 days for this video
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = "2020-01-01"  # Far back enough to cover anything

            response = (
                analytics.reports()
                .query(
                    ids="channel==MINE",
                    startDate=start_date,
                    endDate=end_date,
                    metrics=(
                        "views,estimatedMinutesWatched,averageViewDuration,likes,shares"
                        + (",averageViewPercentage" if include_avg_view_percentage else "")
                    ),
                    filters=f"video=={video_id}",
                )
                .execute()
            )

            rows = response.get("rows", [])
            if rows:
                row = rows[0]
                return {
                    "views": row[0],
                    "watch_time_minutes": row[1],
                    "avg_view_duration_seconds": row[2],
                    "likes": row[3],
                    "shares": row[4],
                    # Absent column = unreported, never 0.
                    "avg_view_percentage": (
                        row[5] if include_avg_view_percentage and len(row) > 5 else None
                    ),
                }
        except Exception as exc:
            self._note_analytics_exception(exc)
            logger.debug("Analytics API not available for %s: %s", video_id, exc)

        return {}

    def fetch_analytics_daily_metrics(
        self, video_id: str, start_date: str, max_days: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Fetch per-day analytics rows (dimensions=day), capped to the first
        `max_days` days after `start_date` (YYYY-MM-DD). Feeds first-hours
        velocity. Returns a list of row dicts (empty on failure/no data).
        """
        try:
            analytics = self._analytics_client()
            if analytics is None:
                return []

            start = datetime.strptime(start_date[:10], "%Y-%m-%d")
            end = min(start + timedelta(days=max_days - 1), datetime.now())
            if end < start:
                return []

            response = (
                analytics.reports()
                .query(
                    ids="channel==MINE",
                    startDate=start_date[:10],
                    endDate=end.strftime("%Y-%m-%d"),
                    dimensions="day",
                    metrics="views,estimatedMinutesWatched,averageViewDuration,likes,shares",
                    filters=f"video=={video_id}",
                )
                .execute()
            )

            keys = [
                "day",
                "views",
                "watch_time_minutes",
                "avg_view_duration_seconds",
                "likes",
                "shares",
            ]
            return [dict(zip(keys, row)) for row in response.get("rows", [])[:max_days]]
        except Exception as exc:
            self._note_analytics_exception(exc)
            logger.debug("Daily analytics not available for %s: %s", video_id, exc)

        return []

    def fetch_channel_daily_dimension(
        self, dimension: str, start_date: str, end_date: str
    ) -> List[tuple]:
        """Channel-level daily views by ``dimension`` (e.g.
        ``insightTrafficSourceType``, ``creatorContentType``) between
        ``start_date``/``end_date`` (``YYYY-MM-DD``, inclusive).

        Feeds ``tracking.channel_daily.pull_channel_daily`` -- reuses this
        collector's own Analytics v2 client (`_analytics_client`), the same
        one `fetch_analytics_metrics` builds, rather than a second auth
        path. Returns ``(day, dimension_value, views)`` tuples; empty on no
        client / no data / an API error, the same degrade-gracefully
        contract as `fetch_analytics_metrics`.
        """
        try:
            analytics = self._analytics_client()
            if analytics is None:
                return []

            response = (
                analytics.reports()
                .query(
                    ids="channel==MINE",
                    startDate=start_date,
                    endDate=end_date,
                    dimensions=f"day,{dimension}",
                    metrics="views",
                )
                .execute()
            )
            return [(row[0], row[1], int(row[2])) for row in response.get("rows", [])]
        except Exception as exc:
            self._note_analytics_exception(exc)
            logger.debug("Channel daily %s not available: %s", dimension, exc)

        return []

    def collect_short_metrics(
        self, video_id: str, published_at: Optional[str] = None
    ) -> PlatformMetrics:
        """
        Full collection: Data API stats + Analytics API detail.

        Args:
            video_id: YouTube video ID
            published_at: ISO 8601 publish timestamp for the daily time
                series; skipped when the aggregate analytics query fails.

        Returns:
            PlatformMetrics with all available data merged
        """
        metrics = self.fetch_video_metrics(video_id)
        return self._enrich_with_analytics(metrics, video_id, published_at)

    def collect_long_form_metrics(
        self, video_id: str, published_at: Optional[str] = None, max_days: int = 28
    ) -> tuple[PlatformMetrics, Optional[float]]:
        """Like ``collect_short_metrics`` for a long-form, and also returns the
        video duration (seconds, or None when the API did not report one) --
        the same Data API call already made, which used to be thrown away.
        """
        metrics, duration_sec = self.fetch_video_metrics_and_duration(video_id)
        return (
            self._enrich_with_analytics(
                metrics, video_id, published_at, max_days=max_days, long_form=True
            ),
            duration_sec,
        )

    def _enrich_with_analytics(
        self,
        metrics: PlatformMetrics,
        video_id: str,
        published_at: Optional[str],
        max_days: int = 14,
        long_form: bool = False,
    ) -> PlatformMetrics:
        # Try to enrich with analytics data
        analytics = self.fetch_analytics_metrics(video_id, include_avg_view_percentage=long_form)
        if analytics:
            # Only overwrite the unreported (None) default when the row
            # actually carries the key — a fabricated 0 here would look
            # identical to a genuine zero once stored (schema.py policy).
            watch_time_minutes = analytics.get("watch_time_minutes")
            if watch_time_minutes is not None:
                metrics.watch_time_seconds = watch_time_minutes * 60
            avg_view_duration = analytics.get("avg_view_duration_seconds")
            if avg_view_duration is not None:
                metrics.avg_view_duration_seconds = avg_view_duration
            metrics.avg_view_percentage = analytics.get("avg_view_percentage")
            if analytics.get("shares", 0) > 0:
                metrics.shares = analytics["shares"]

            if published_at:
                daily_rows = self.fetch_analytics_daily_metrics(
                    video_id, published_at, max_days=max_days
                )
                metrics.daily = daily_points(daily_rows)

        return metrics

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_iso_duration(iso_str: str) -> float:
        """Parse ISO 8601 duration (e.g. PT1M30S) to seconds."""
        import re

        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso_str)
        if not match:
            return 0.0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    def is_available(self) -> bool:
        """Check if YouTube API access is configured (without making API calls)."""
        if self._token_path.exists() or self._api_key is not None:
            return True
        return self._shared_credentials() is not None


class TikTokCollector:
    """Shorts metrics via the TikTok Display API `video/query`. Lazy
    credentials, degrades like the other collectors on missing/revoked
    creds or an API error. Ids are batched (never one request per short)
    and paged via `cursor`/`has_more`; upload handles aren't queryable
    and are skipped."""

    def __init__(
        self,
        access_token: Optional[str] = None,
        session: Optional[Any] = None,
        brand: Optional[str] = None,
    ):
        self._token = access_token
        self._token_loaded = access_token is not None
        self._session = session
        self._brand = brand

    def _access_token(self) -> Optional[str]:
        if not self._token_loaded:
            try:
                creds = get_valid_tiktok_credentials(brand=self._brand)
                self._token = creds.get("access_token")
            except (FileNotFoundError, TikTokAuthError) as exc:
                logger.info("TikTok credentials not configured: %s", exc)
            self._token_loaded = True
        return self._token

    def is_available(self) -> bool:
        return self._access_token() is not None

    def _query_batch(
        self, token: str, video_ids: List[str]
    ) -> tuple[Dict[str, PlatformMetrics], str, str]:
        """``(metrics_by_id, failure_status, detail)``. ``failure_status`` is
        empty when the query itself succeeded; an id missing from the dict is
        then genuinely one TikTok declined to return, not a hidden error."""
        if self._session is None:
            import requests

            self._session = requests.Session()
        session = self._session
        collected: Dict[str, PlatformMetrics] = {}
        cursor: Optional[int] = None
        while True:
            body: Dict[str, Any] = {"filters": {"video_ids": video_ids}}
            if cursor:
                body["cursor"] = cursor
            try:
                response = session.post(
                    _TIKTOK_QUERY_URL,
                    params={"fields": _TIKTOK_QUERY_FIELDS},
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
                payload = response.json()
            except Exception as exc:
                logger.error("TikTok video/query request failed: %s", exc)
                return collected, ObservationStatus.PROVIDER_ERROR, str(exc)
            error = (payload or {}).get("error") or {}
            if error.get("code") not in (None, "ok"):
                code = str(error.get("code"))
                detail = str(error.get("message") or code)
                logger.error("TikTok video/query error: %s", detail)
                return (
                    collected,
                    _TIKTOK_ERROR_STATUS.get(code, ObservationStatus.PROVIDER_ERROR),
                    detail,
                )
            data = payload.get("data") or {}
            for video in data.get("videos", []):
                vid = video.get("id")
                if vid:
                    # A count TikTok did not return is unknown, not a measured
                    # zero -- the same contract _reported_metric carries for the
                    # YouTube and Instagram collectors, and the contract used
                    # elsewhere in this service. A genuine 0 in the payload
                    # still reads as 0.
                    collected[vid] = PlatformMetrics(
                        views=_reported_metric(video.get("view_count")),
                        likes=_reported_metric(video.get("like_count")),
                        comments=_reported_metric(video.get("comment_count")),
                        shares=_reported_metric(video.get("share_count")),
                    )
            cursor = data.get("cursor") if data.get("has_more") else None
            if not cursor:
                return collected, "", ""

    def collect_observations_batch(
        self,
        video_ids: List[str],
        account_id: Optional[str] = None,
        event_at_by_id: Optional[Dict[str, str]] = None,
    ) -> Dict[str, MetricObservation]:
        """One :class:`MetricObservation` for EVERY requested id.

        The old ``collect_metrics_batch`` returned a bare ``{}`` for three
        different worlds -- every id an unqueryable upload handle, no access
        token, and a query that ran and returned nothing -- which a caller
        could not tell apart. Here an id is never simply absent: it carries
        the reason, so "no numbers" can be acted on.
        """
        fetched_at = datetime.now().astimezone().isoformat()
        event_at_by_id = event_at_by_id or {}

        def _observe(vid: str, status: str, detail: str = "", metrics=None):
            return MetricObservation(
                platform="tiktok",
                status=status,
                post_id=vid,
                account_id=account_id,
                post_format="short",
                cumulative=metrics or PlatformMetrics(),
                event_at=event_at_by_id.get(vid),
                fetched_at=fetched_at,
                detail=detail,
            )

        observations: Dict[str, MetricObservation] = {}
        queryable: List[str] = []
        for vid in dict.fromkeys(video_ids):
            if not vid:
                continue
            if _looks_like_upload_handle(vid):
                observations[vid] = _observe(
                    vid,
                    ObservationStatus.UNQUERYABLE_ID,
                    "upload handle, not a queryable public post id",
                )
            else:
                queryable.append(vid)
        if not queryable:
            return observations
        token = self._access_token()
        if not token:
            logger.info("TikTok collector not configured, skipping %d video id(s)", len(queryable))
            for vid in queryable:
                observations[vid] = _observe(
                    vid, ObservationStatus.NOT_CONFIGURED, "no TikTok access token"
                )
            return observations
        for i in range(0, len(queryable), _TIKTOK_MAX_IDS_PER_REQUEST):
            batch_ids = queryable[i : i + _TIKTOK_MAX_IDS_PER_REQUEST]
            collected, failure, detail = self._query_batch(token, batch_ids)
            for vid in batch_ids:
                metrics = collected.get(vid)
                if metrics is not None:
                    observations[vid] = _observe(vid, ObservationStatus.MEASURED, metrics=metrics)
                elif failure:
                    observations[vid] = _observe(vid, failure, detail)
                else:
                    observations[vid] = _observe(
                        vid,
                        ObservationStatus.NOT_RETURNED,
                        "queried; TikTok did not return this id (deleted, private, "
                        "or not on this account)",
                    )
        return observations

    def collect_metrics_batch(self, video_ids: List[str]) -> Dict[str, PlatformMetrics]:
        """Metrics for the ids TikTok actually measured.

        A thin projection of :meth:`collect_observations_batch`, kept for
        callers that only want numbers. Prefer the observations: this shape
        still cannot say WHY an id is missing."""
        return {
            vid: obs.cumulative
            for vid, obs in self.collect_observations_batch(video_ids).items()
            if obs.is_measurement
        }

    def collect_short_metrics(
        self, video_id: str, published_at: Optional[str] = None
    ) -> PlatformMetrics:
        """Single-video convenience wrapper over ``collect_metrics_batch``."""
        return self.collect_metrics_batch([video_id]).get(video_id, PlatformMetrics())


class InstagramCollector:
    """Collects Reel metrics via the Graph API. Credentials load lazily on
    first use (never at construction, so building this never touches SOPS)
    and degrade like ``YouTubeCollector`` when missing. A Graph 403/scope
    error is caught, logged, and treated as empty rather than raised —
    live-scope coverage is unverified."""

    def __init__(
        self,
        credentials: Optional[Dict[str, Any]] = None,
        session: Optional[Any] = None,
        brand: Optional[str] = None,
    ):
        self._credentials = credentials
        self._credentials_loaded = credentials is not None
        self._session = session
        self._brand = brand
        #: ``(status, detail)`` of the last refusal, so an observation can
        #: report WHICH refusal it was rather than an undifferentiated empty.
        self._last_failure: Optional[tuple[str, str]] = None

    def _creds(self) -> Dict[str, Any]:
        if not self._credentials_loaded:
            try:
                self._credentials = load_meta_credentials(brand=self._brand)
            except FileNotFoundError as exc:
                logger.info("Instagram credentials not configured: %s", exc)
            self._credentials_loaded = True
        return self._credentials or {}

    def is_available(self) -> bool:
        creds = self._creds()
        return bool(creds.get("page_access_token") and creds.get("ig_user_id"))

    def _play_metric_name(self) -> str:
        graph_version = self._creds().get("graph_version") or DEFAULT_GRAPH_VERSION
        try:
            major = float(str(graph_version).lstrip("v"))
        except ValueError:
            major = 0.0
        return "views" if major >= _IG_VIEWS_METRIC_CUTOVER_MAJOR else "plays"

    def _get(self, path: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        graph_version = self._creds().get("graph_version") or DEFAULT_GRAPH_VERSION
        if self._session is None:
            import requests

            self._session = requests.Session()
        try:
            url = f"https://graph.facebook.com/{graph_version}/{path}"
            response = self._session.get(url, params=params, timeout=30)
            body = response.json()
        except Exception as exc:  # requests errors + non-JSON bodies
            logger.error("Instagram Graph API request failed for %s: %s", path, exc)
            self._last_failure = (ObservationStatus.PROVIDER_ERROR, str(exc))
            return None
        error = (body or {}).get("error")
        if not error and response.status_code < 400:
            return body
        message = str((error or {}).get("message") or f"HTTP {response.status_code}")
        kind = (
            "scope error" if response.status_code == 403 or "scope" in message.lower() else "error"
        )
        logger.error("Instagram Graph API %s: %s", kind, message)
        self._last_failure = (_graph_failure_status(response.status_code, message), message)
        return None

    def collect_short_metrics(
        self, media_id: str, published_at: Optional[str] = None
    ) -> PlatformMetrics:
        """views/reach/likes/comments/shares for one Reel; never raises.

        "Empty" on any failure or missing creds means every metric ``None``
        -- unknown -- not zero. A Graph scope denial is the platform
        declining to answer, and recording that as a measured 0 would let a
        permissions problem read downstream as a Reel that flopped."""
        if not self.is_available():
            return PlatformMetrics()
        play_metric = self._play_metric_name()
        metric_names = f"{play_metric},reach,likes,comments,shares"
        params = {"metric": metric_names, "access_token": self._creds().get("page_access_token")}
        body = self._get(f"{media_id}/insights", params) or {}
        values = {
            # No ``, 0`` default: a metric Graph names but returns no value
            # for is unreported, and a zero here would be indistinguishable
            # from a Reel nobody watched.
            e["name"]: e["values"][0].get("value")
            for e in body.get("data", [])
            if e.get("name") and e.get("values")
        }
        # A refused request (`_get` returned None, so `body` is `{}`) leaves
        # every name absent and therefore every metric ``None``. That is the
        # point: a scope denial means Graph would not SAY, which is not the
        # same answer as zero, and this collector's results are read as
        # measurements downstream.
        return PlatformMetrics(
            views=_reported_metric(values.get(play_metric)),
            likes=_reported_metric(values.get("likes")),
            comments=_reported_metric(values.get("comments")),
            shares=_reported_metric(values.get("shares")),
            impressions=_reported_metric(values.get("reach")),
        )

    def collect_observation(
        self, media_id: str, published_at: Optional[str] = None
    ) -> MetricObservation:
        """One Reel's metrics WITH the reason when there are none.

        ``collect_short_metrics`` returns an all-``None`` result for a scope
        denial, a deleted Reel and a Reel Graph simply had nothing for. Same
        request, but the answer says which."""
        base: Dict[str, Any] = {
            "platform": "instagram",
            "post_id": media_id,
            "account_id": self._creds().get("ig_user_id"),
            "post_format": "reel",
            "event_at": published_at,
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
        if not self.is_available():
            return MetricObservation(
                status=ObservationStatus.NOT_CONFIGURED,
                detail="no Instagram Graph credentials",
                **base,
            )
        self._last_failure = None
        metrics = self.collect_short_metrics(media_id, published_at=published_at)
        return _answered_observation(base, metrics, self._last_failure)

    def fetch_audience_online(self) -> Dict[int, int]:
        """Hour-of-day follower-online histogram from the account's own
        ``online_followers`` insight; empty on any failure/missing creds."""
        if not self.is_available():
            return {}
        creds = self._creds()
        params = {
            "metric": "online_followers",
            "period": "lifetime",
            "access_token": creds.get("page_access_token"),
        }
        body = self._get(f"{creds.get('ig_user_id')}/insights", params) or {}
        data = body.get("data") or []
        values = data[0].get("values") if data else None
        histogram = values[-1].get("value") if values else None
        return {int(h): int(n) for h, n in histogram.items()} if histogram else {}


#: Video-edge metrics are not Post insights aliases. Meta retired video
#: impressions for every Graph version in June 2026; requesting one retired
#: metric rejects the entire bundled lookup. Keep impressions unknown rather
#: than replacing them with a differently defined media-view count.
#: Contract and current-version notes: docs/facebook-metrics.md.
_FB_VIDEO_METRICS: Dict[str, str] = {
    "views": "total_video_views",
    "watch_time_seconds": "total_video_view_total_time",
    "avg_view_duration_seconds": "total_video_avg_time_watched",
}

#: Reel play definitions differ from the Video edge's three-second views.
_FB_REEL_METRICS: Dict[str, str] = {
    "reel_initial_plays": "blue_reels_play_count",
    "reel_replays": "fb_reels_replay_count",
    "reel_total_plays": "fb_reels_total_plays",
    "watch_time_seconds": "post_video_view_time",
    "avg_view_duration_seconds": "post_video_avg_time_watched",
}

#: Reel reach candidates, each its own lifetime request so a retired name
#: cannot reject the play-count bundle; the first numeric answer wins.
_FB_REEL_REACH_METRICS = ("post_impressions_unique", "post_total_media_view_unique")

#: Page insights report a day closed at midnight Pacific (``end_time``).
_FB_PAGE_DAY_TZ = ZoneInfo("America/Los_Angeles")

#: Scope gating both insight edges; a token may lack it,
#: so every read degrades instead of raising. Page cap guards a bad cursor chain.
_FB_INSIGHTS_SCOPE = "read_insights"
_FB_MAX_INSIGHT_PAGES = 10


class FacebookCollector:
    """Facebook video metrics plus the Page's hour-of-day fan-online curve.
    Shaped like :class:`InstagramCollector`: credentials load lazily on first
    use (never at construction, so building this never touches SOPS) and a
    Graph error is caught+logged, never raised. A token without
    ``read_insights`` logs the scope NAME once and returns empty un-requested."""

    def __init__(
        self,
        credentials: Optional[Dict[str, Any]] = None,
        session: Optional[Any] = None,
        brand: Optional[str] = None,
    ):
        self._credentials = credentials
        self._credentials_loaded = credentials is not None
        self._session = session
        self._brand = brand
        self._scope_warned = False
        self._last_failure: Optional[tuple[str, str]] = None
        self._last_post_format = ""
        #: Reach candidates Graph refused this run; never asked again.
        self._refused_reach_metrics: set[str] = set()

    @property
    def last_failure_detail(self) -> str:
        """Graph's message for the last refused request ('' when none)."""
        return self._last_failure[1] if self._last_failure else ""

    def _creds(self) -> Dict[str, Any]:
        if not self._credentials_loaded:
            try:
                self._credentials = load_meta_credentials(brand=self._brand)
            except FileNotFoundError as exc:
                logger.info("Facebook credentials not configured: %s", exc)
            self._credentials_loaded = True
        return self._credentials or {}

    def is_available(self) -> bool:
        creds = self._creds()
        return bool(creds.get("page_access_token") and creds.get("page_id"))

    def _graph_version(self) -> str:
        return self._creds().get("graph_version") or DEFAULT_GRAPH_VERSION

    def _has_insights_scope(self) -> bool:
        """True when the token carries ``read_insights``; else logs the scope
        NAME (never a credential value) once per collector."""
        if _FB_INSIGHTS_SCOPE in (self._creds().get("scopes") or []):
            return True
        if not self._scope_warned:
            self._scope_warned = True
            logger.warning(
                "Facebook insights unavailable: token is missing the %s scope — "
                "reconsent to collect them; returning empty until then",
                _FB_INSIGHTS_SCOPE,
            )
        return False

    def _get(self, url: str, params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._session is None:
            import requests

            self._session = requests.Session()
        try:
            response = self._session.get(url, params=params, timeout=30)
            body = response.json()
        except Exception as exc:  # requests errors + non-JSON bodies
            logger.error("Facebook Graph API request failed for %s: %s", url, exc)
            self._last_failure = (ObservationStatus.PROVIDER_ERROR, str(exc))
            return None
        error = (body or {}).get("error")
        if not error and response.status_code < 400:
            return body
        message = str((error or {}).get("message") or f"HTTP {response.status_code}")
        logger.error("Facebook Graph API error: %s", message)
        self._last_failure = (_graph_failure_status(response.status_code, message), message)
        return None

    def _get_data(self, path: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Every ``data`` entry for an insights edge (they page), following
        ``paging.next`` up to :data:`_FB_MAX_INSIGHT_PAGES`."""
        entries: List[Dict[str, Any]] = []
        url: Optional[str] = f"https://graph.facebook.com/{self._graph_version()}/{path}"
        for _ in range(_FB_MAX_INSIGHT_PAGES):
            body = self._get(url, params) if url else None
            if not body:
                break
            entries.extend(body.get("data") or [])
            params = None  # the cursor URL already carries token + metric
            url = (body.get("paging") or {}).get("next")
        return entries

    def collect_short_metrics(
        self, video_id: str, published_at: Optional[str] = None
    ) -> PlatformMetrics:
        """Video views or distinct Reel plays/watch-time for one public ID.

        A successful empty Video answer gets one Reel-metric lookup. Refused
        requests never fan out; an empty response never becomes a zero.
        """
        self._last_failure = None
        self._last_post_format = ""
        if not self.is_available() or not self._has_insights_scope():
            return PlatformMetrics()

        def read_metrics(names: Dict[str, str]) -> Dict[str, float]:
            params = {
                "metric": ",".join(names.values()),
                "period": "lifetime",
                "access_token": self._creds().get("page_access_token"),
            }
            entries = self._get_data(f"{video_id}/video_insights", params)
            return {
                e["name"]: float(e["values"][0]["value"])
                for e in entries
                if e.get("name") in names.values()
                and isinstance((e.get("values") or [{}])[0].get("value"), (int, float))
            }

        names = _FB_VIDEO_METRICS
        values = read_metrics(names)
        reach: Optional[float] = None
        reach_metric: Optional[str] = None
        if values:
            self._last_post_format = "video"
        elif self._last_failure is None:
            names = _FB_REEL_METRICS
            values = read_metrics(names)
            if values:
                self._last_post_format = "reel"
                reach, reach_metric = self._read_reel_reach(read_metrics)

        # A metric Graph did not report is unknown, not zero. `values` is built
        # only from entries Graph actually returned, so an absent name means it
        # would not say -- which is a different answer from a measured 0, and
        # these results are read as measurements downstream.
        def _ms_to_seconds(name: str) -> Optional[float]:
            # Graph reports video watch times in milliseconds. The conversion
            # has to survive the absent case: 0/1000.0 would manufacture a
            # measured 0.0 out of silence.
            raw = values.get(name)
            return None if raw is None else raw / 1000.0

        return PlatformMetrics(
            views=_reported_metric(values.get(names.get("views", ""))),
            watch_time_seconds=_ms_to_seconds(names["watch_time_seconds"]),
            avg_view_duration_seconds=_ms_to_seconds(names["avg_view_duration_seconds"]),
            reel_initial_plays=_reported_metric(values.get("blue_reels_play_count")),
            reel_replays=_reported_metric(values.get("fb_reels_replay_count")),
            reel_total_plays=_reported_metric(values.get("fb_reels_total_plays")),
            reach=_reported_metric(reach),
            reach_metric=reach_metric,
        )

    def _read_reel_reach(
        self, read_metrics: Callable[[Dict[str, str]], Dict[str, float]]
    ) -> tuple[Optional[float], Optional[str]]:
        """First numeric reach candidate for a measured Reel. A refusal is
        remembered for the run and never becomes the observation's failure:
        the play counts were already measured."""
        measured_failure = self._last_failure
        try:
            for metric in _FB_REEL_REACH_METRICS:
                if metric in self._refused_reach_metrics:
                    continue
                self._last_failure = None
                value = read_metrics({"reach": metric}).get(metric)
                if self._last_failure is not None:
                    self._refused_reach_metrics.add(metric)
                    continue
                if value is not None:
                    return value, metric
            return None, None
        finally:
            self._last_failure = measured_failure

    def fetch_page_daily(
        self, metric: str, since: str, until: str
    ) -> Optional[List[tuple[str, int]]]:
        """One Page insight's daily values for ``[since, until)`` as
        ``(pacific_day, value)`` pairs; ``None`` when Graph refused or the
        token cannot ask. One request: a time-ranged answer's ``paging.next``
        is the NEXT date window, so it is never followed."""
        self._last_failure = None
        if not self.is_available() or not self._has_insights_scope():
            return None
        creds = self._creds()
        url = f"https://graph.facebook.com/{self._graph_version()}/{creds.get('page_id')}/insights"
        params = {
            "metric": metric,
            "period": "day",
            "since": since,
            "until": until,
            "access_token": creds.get("page_access_token"),
        }
        body = self._get(url, params)
        if body is None:
            return None
        points: List[tuple[str, int]] = []
        for entry in body.get("data") or []:
            if entry.get("name") != metric or entry.get("period") != "day":
                continue
            for point in entry.get("values") or []:
                value, end_time = point.get("value"), point.get("end_time")
                if not isinstance(value, (int, float)) or not end_time:
                    continue
                closed = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S%z")
                day = (closed.astimezone(_FB_PAGE_DAY_TZ) - timedelta(seconds=1)).date()
                points.append((day.isoformat(), int(value)))
        return points

    def collect_observation(
        self, video_id: str, published_at: Optional[str] = None
    ) -> MetricObservation:
        """One video's metrics WITH the reason when there are none -- most
        often a token without ``read_insights``, which must never read
        downstream as a video nobody watched."""
        base: Dict[str, Any] = {
            "platform": "facebook",
            "post_id": video_id,
            "account_id": self._creds().get("page_id"),
            "post_format": "",
            "event_at": published_at,
            "fetched_at": datetime.now().astimezone().isoformat(),
        }
        if not self.is_available():
            return MetricObservation(
                status=ObservationStatus.NOT_CONFIGURED,
                detail="no Facebook Page credentials",
                **base,
            )
        if not self._has_insights_scope():
            return MetricObservation(
                status=ObservationStatus.PERMISSION_DENIED,
                detail=f"token is missing the {_FB_INSIGHTS_SCOPE} scope",
                **base,
            )
        self._last_failure = None
        metrics = self.collect_short_metrics(video_id, published_at=published_at)
        base["post_format"] = self._last_post_format
        return _answered_observation(base, metrics, self._last_failure)

    def fetch_audience_online(self) -> Dict[int, int]:
        """Hour-of-day fan-online histogram from the Page's own
        ``page_fans_online`` insight; the latest snapshot of an hour wins.
        Empty on missing creds, a missing scope, or any Graph failure."""
        if not self.is_available() or not self._has_insights_scope():
            return {}
        creds = self._creds()
        params = {"access_token": creds.get("page_access_token")}
        histogram: Dict[int, int] = {}
        edge = f"{creds.get('page_id')}/insights/page_fans_online"
        for entry in self._get_data(edge, params):
            for point in entry.get("values") or []:
                for hour, n_online in (point.get("value") or {}).items():
                    histogram[int(hour)] = int(n_online)
        return histogram


class MetricsCollector:
    """
    Unified collector that delegates to platform-specific collectors.

    Usage:
        collector = MetricsCollector()
        record = store.load("20250101Example")
        collector.collect_all(record)
        store.save(record)
    """

    def __init__(
        self,
        youtube_client_secret: Optional[Path] = None,
        youtube_token: Optional[Path] = None,
        youtube_api_key: Optional[str] = None,
        tiktok_token: Optional[str] = None,
        brand: Optional[str] = None,
    ):
        self.youtube = YouTubeCollector(
            client_secret_path=youtube_client_secret,
            token_path=youtube_token,
            api_key=youtube_api_key,
            brand=brand,
        )
        # Wired to the Display API `video/query`.
        self.tiktok = TikTokCollector(access_token=tiktok_token, brand=brand)
        # Credentials load lazily on first use (SOPS decrypt), never here —
        # constructing MetricsCollector must never touch SOPS.
        self.instagram = InstagramCollector(brand=brand)
        self.facebook = FacebookCollector(brand=brand)

        self._collectors: Dict[str, Any] = {
            "youtube": self.youtube,
            "tiktok": self.tiktok,
            "instagram": self.instagram,
            "facebook": self.facebook,
        }

    def collect_observations_for_short(self, short: ShortRecord) -> Dict[str, MetricObservation]:
        """One :class:`MetricObservation` per platform this short has an id
        for -- a measurement OR a refusal, never a silent skip.

        This is the ONE place production collection calls a platform's own
        observation method (``collect_observation`` /
        ``collect_observations_batch``); ``collect_for_short`` (bare numbers
        for legacy callers) and ``collect_all`` (appends to
        ``observation_log``) both project from this result rather than
        calling the platform APIs a second time, so there is exactly one
        collection path, not two that can drift.
        """
        observations: Dict[str, MetricObservation] = {}
        for platform, video_id in short.platform_ids.items():
            collector = self._collectors.get(platform)
            if collector is None:
                logger.warning("No collector for platform %s", platform)
                continue
            try:
                if hasattr(collector, "collect_observations_batch"):
                    event_at_by_id = {video_id: short.posted_at} if short.posted_at else None
                    batch = collector.collect_observations_batch(
                        [video_id], event_at_by_id=event_at_by_id
                    )
                    obs = batch.get(video_id)
                elif hasattr(collector, "collect_observation"):
                    obs = collector.collect_observation(video_id, published_at=short.posted_at)
                else:
                    logger.warning("Collector for %s has no observation method, skipping", platform)
                    continue
            except Exception as exc:
                logger.error(
                    "Failed collecting %s observation for short #%d: %s",
                    platform,
                    short.index,
                    exc,
                )
                continue
            if obs is not None:
                observations[platform] = obs
        return observations

    def collect_for_short(self, short: ShortRecord) -> Dict[str, PlatformMetrics]:
        """
        Collect metrics for a single short from all platforms where it has IDs.

        A thin projection of :meth:`collect_observations_for_short` for
        callers that only want the bare numbers -- a platform that refused
        is simply absent from the result, same as before.

        Args:
            short: ShortRecord with platform_ids populated

        Returns:
            Dict of platform → PlatformMetrics
        """
        results: Dict[str, PlatformMetrics] = {}

        for platform, obs in self.collect_observations_for_short(short).items():
            if not obs.is_measurement:
                # Not a measurement -- an unqueryable id, a deleted or
                # private post, a permission the token lacks, a throttle.
                # The old line formatted ``views`` with %d, which raised
                # TypeError inside logging when it was None; silence is not
                # a failure and it is not a zero.
                logger.info(
                    "%s answered for short #%d but reported no measurement (%s%s)",
                    platform,
                    short.index,
                    obs.status,
                    f": {obs.detail}" if obs.detail else "",
                )
                continue
            results[platform] = obs.cumulative
            if obs.cumulative.views is None:
                logger.info(
                    "%s answered for short #%d but no views reported "
                    "(nothing measured — not a zero)",
                    platform,
                    short.index,
                )
            else:
                logger.info(
                    "Collected %s metrics for short #%d: %d views",
                    platform,
                    short.index,
                    obs.cumulative.views,
                )

        return results

    def collect_all(self, record: TrackingRecord, today: Optional[date] = None) -> int:
        """
        Collect observations for all shorts in a tracking record.

        Appends every observation -- measurement or refusal -- to
        ``ShortRecord.observation_log`` via ``store.append_observation``, so
        the series that ``interval.latest_interval`` reads always has this
        run's answer. ``ShortRecord.metrics`` is updated only from a
        MEASURED observation's cumulative numbers; a refusal carries no
        numbers and must never overwrite a prior measurement.

        After the observations, fetches the YouTube Analytics daily series of
        every short whose day-7 window has settled and that does not already
        hold a settled series (``tracking/day7.py``); a settled series is never
        refetched. A quota 403/429 stops those fetches and records
        ``budget_exhausted_line``.

        Args:
            record: TrackingRecord to update
            today: the date settling is judged against (default: today)

        Returns:
            Number of shorts at least one platform MEASURED this run. A
            short every platform refused about is not counted -- its
            refusals are recorded in ``observation_log`` instead.
        """
        updated = 0
        health: Dict[str, List[int]] = {}  # attempted, measured this run
        now = datetime.now().astimezone().isoformat()

        for short in record.shorts:
            if not short.platform_ids:
                logger.debug("Short #%d has no platform IDs, skipping", short.index)
                continue

            for platform in short.platform_ids:
                health.setdefault(platform, [0, 0])[0] += 1
            observations = self.collect_observations_for_short(short)
            if not observations:
                continue

            measured_any = False
            for platform, obs in observations.items():
                counts = health[platform]
                try:
                    append_observation(record, short.index, obs)
                except ValueError as exc:
                    # Best-effort, matching collect_for_short: one platform's
                    # bookkeeping failure must never abort the others.
                    logger.error(
                        "Failed to append %s observation for short #%d: %s",
                        platform,
                        short.index,
                        exc,
                    )
                    continue
                if obs.is_measurement:
                    counts[1] += 1
                    prior = short.metrics.get(platform)
                    short.metrics[platform] = obs.cumulative
                    # A cumulative read carries no daily series; keep the one
                    # already collected rather than wipe it every run.
                    if prior is not None and not obs.cumulative.daily:
                        obs.cumulative.daily = prior.daily
                    measured_any = True

            # ``collected_at`` is "a collection was ATTEMPTED", so a run every
            # platform refused still stamps it -- that attempt happened.
            short.collected_at = now
            # The COUNT is a different claim. It feeds the CLI's "Collected
            # metrics for N short(s)" line, and a short every platform refused
            # about measured nothing; counting it restates for the count what
            # ``test_tracking_collected_nothing_is_not_collected`` refuses for
            # the log line. The refusals are not lost -- they are in
            # ``observation_log``, which is what makes them reportable without
            # inflating a measurement count.
            if measured_any:
                updated += 1

        self._collect_settled_series(record, today or date.today())

        for platform, (attempted, measured) in sorted(health.items()):
            log = logger.warning if measured < attempted else logger.info
            log(
                "metrics health platform=%s measured=%d attempted=%d unanswered=%d",
                platform,
                measured,
                attempted,
                attempted - measured,
            )

        logger.info(
            "Updated metrics for %d/%d shorts in project %s",
            updated,
            len(record.shorts),
            record.project,
        )
        return updated

    #: Set when a run's daily-series fetches were stopped by the quota.
    budget_exhausted_line: Optional[str] = None

    def _collect_settled_series(self, record: TrackingRecord, today: date) -> None:
        pending = [
            s
            for s in record.shorts
            if s.platform_ids.get("youtube")
            and s.posted_at
            and (posted := posted_date(s.posted_at)) is not None
            and window_settled(posted, today)
            and not has_settled_series(s, today)
        ]
        for i, short in enumerate(pending):
            if self.youtube.quota_exhausted:
                break
            rows = self.youtube.fetch_analytics_daily_metrics(
                short.platform_ids["youtube"], short.posted_at, max_days=DAY7_DAYS
            )
            if self.youtube.quota_exhausted:
                break
            if rows:
                metrics = short.metrics.setdefault("youtube", PlatformMetrics())
                metrics.daily = daily_points(rows)
        else:
            return
        self.budget_exhausted_line = (
            f"ANALYTICS_BUDGET_EXHAUSTED remaining={len(pending) - i} reason=quota"
        )
        logger.warning(self.budget_exhausted_line)

    def collect_long_forms(self, record: TrackingRecord) -> int:
        """Collect YouTube metrics + duration for every long-form in ``record``.

        Returns the number of long-forms something was MEASURED for. A
        collection that returned no numbers never overwrites a prior
        measurement, and a missing duration never overwrites a known one.
        """
        if not self.youtube.is_available():
            return 0
        updated = 0
        now = datetime.now().astimezone().isoformat()
        for lf in record.long_forms:
            if not lf.video_id:
                continue
            metrics, duration_sec = self.youtube.collect_long_form_metrics(
                lf.video_id, published_at=lf.published_date
            )
            lf.collected_at = now
            if duration_sec is not None:
                lf.duration_sec = duration_sec
                lf.duration_source = LONG_FORM_DURATION_SOURCE_YOUTUBE
            # has_numbers() predates avg_view_percentage; a lone one is a measurement.
            if has_numbers(metrics) or metrics.avg_view_percentage is not None:
                lf.metrics["youtube"] = metrics
                updated += 1
        return updated

    def available_platforms(self) -> List[str]:
        """Return list of platforms with configured collectors."""
        return [name for name, c in self._collectors.items() if c.is_available()]
