"""Recording adapters and builders for the acceptance lane (tests/e2e).

Adapters are injected through ``handlers.build_handlers`` exactly as
production does; nothing here touches the network, credentials, or a real
drive. The queue is the real sqlite queue in a temp dir."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

from gomoveshift.publishing.models import (
    FacebookPayload,
    InstagramPayload,
    MediaFingerprint,
    PlatformPayloads,
    RouteSelection,
    ScheduleWindow,
    ShortPublishRecord,
    TikTokPayload,
    YouTubePayload,
)
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_CONFIRMED, transition

NOW = "2026-10-03T12:00:00+00:00"
ACCOUNT = "acct-gomoveshift-ig"
PROJECT = "example-journey"


def make_record(
    asset: Path,
    *,
    caption: str = "Cresting the ridge at dawn.",
    short_index: int = 0,
    platforms: tuple = ("instagram", "youtube", "tiktok"),
) -> ShortPublishRecord:
    return ShortPublishRecord(
        project_name=PROJECT,
        lf_index=1,
        short_index=short_index,
        media=MediaFingerprint(
            path=str(asset),
            size_bytes=11,
            duration_seconds=30.0,
            width=1080,
            height=1920,
            codec="h264",
            sha256="a" * 64,
        ),
        schedule=ScheduleWindow(
            scheduled_at_utc=datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc),
            display_timezone="America/New_York",
        ),
        recording_date="2026-09-20",
        sampled_regions=("Ridge",),
        parent_lf_video_id="yt_parent",
        parent_lf_publish_at=datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc),
        payloads=PlatformPayloads(
            youtube=YouTubePayload(title="Ridge", description=caption, category="19"),
            instagram=InstagramPayload(caption=caption),
            facebook=FacebookPayload(description=caption),
            tiktok=TikTokPayload(title="Ridge", privacy_level="SELF_ONLY", caption=caption),
        ),
        consent_hash="c" * 64,
        reviewed_plan_hash="d" * 64,
        routes=RouteSelection(selected=platforms),
        format="short",
    )


class RecordingInstagramAdapter:
    """Records every ``publish`` call and plays a script, one step per call:
    ``"ok"``, ``"transient"``, ``"auth"``, ``"accept_then_lose"``.
    ``remote_posts`` is the platform's side: one entry per accepted post, so
    a duplicate publish is visible as ``len(remote_posts) == 2``."""

    def __init__(self, script: Optional[List[str]] = None, account: str = ACCOUNT):
        from gomoveshift.publishing.instagram_adapter import (
            InstagramAuthBlockedError,
            InstagramRetryableError,
        )

        self._auth_exc = InstagramAuthBlockedError
        self._transient_exc = InstagramRetryableError
        self.script = list(script or [])
        self.account = account
        self.calls: List[dict] = []
        self.remote_posts: List[dict] = []

    def _step(self) -> str:
        return self.script.pop(0) if self.script else "ok"

    def publish(self, record, *, conn, delivery_id, video_bytes, **_kw):
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        self.calls.append(
            {
                "delivery_id": delivery_id,
                "asset_path": record.media.path,
                "caption": record.payloads.instagram.caption,
                "account": self.account,
                "video_bytes": video_bytes,
            }
        )
        step = self._step()
        if step == "transient":
            raise self._transient_exc("graph api 5xx")
        if step == "auth":
            raise self._auth_exc("access token expired")
        # Idempotent adoption, as the real adapters do: a persisted container
        # id means the remote post exists already -- adopt it, never re-post.
        post_id = row["platform_upload_id"]
        if not post_id:
            post_id = f"ig-post-{len(self.remote_posts) + 1}"
            self.remote_posts.append({"id": post_id, "caption": self.calls[-1]["caption"]})
            transition(conn, delivery_id, row["status"], "uploaded", platform_upload_id=post_id)
            row = conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone()
        if step == "accept_then_lose":
            raise self._transient_exc("connection reset after remote accept")
        transition(
            conn,
            delivery_id,
            row["status"],
            "published",
            platform_post_id=post_id,
            published_at_source=PUBLISHED_AT_PLATFORM_CONFIRMED,
            destination_account=self.account,
        )


def asset_loader(record) -> bytes:
    """Mirrors a production loader: reads the real file, so a missing asset
    raises before the adapter is ever called."""
    return Path(record.media.path).read_bytes()


def row_of(conn, delivery_id) -> dict:
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


def loader_for(records: dict) -> Callable[[str, int], Any]:
    return lambda project, idx: records[(project, idx)]
