"""Unified read-only comment inbox over the shared JSONL ledger (list-only
half of the comment inbox; replying is out of scope)."""

from __future__ import annotations

from gomoveshift.pipeline.comments import (
    UNCONNECTED_PLATFORMS,
    build_inbox_rows,
    inbox_platform_status,
)


def _comment(platform, video_id, author, text, published_at):
    return {
        "platform": platform,
        "video_id": video_id,
        "author": author,
        "text": text,
        "published_at": published_at,
        "comment_id": f"{platform}-{author}-{published_at}",
    }


def test_build_inbox_rows_sorts_newest_first_across_platforms():
    comments = [
        _comment("youtube", "vid1", "Alice", "great drive", "2026-09-01T00:00:00Z"),
        _comment("tiktok", "vid2", "Bob", "nice", "2026-09-10T00:00:00Z"),
        _comment("youtube", "vid1", "Carl", "where is this", "2026-09-05T00:00:00Z"),
    ]
    rows = build_inbox_rows(comments)
    assert [r["author"] for r in rows] == ["Bob", "Carl", "Alice"]


def test_build_inbox_rows_carries_platform_video_author_text_and_replied():
    comments = [_comment("youtube", "vid1", "Alice", "great drive", "2026-09-01T00:00:00Z")]
    rows = build_inbox_rows(comments)
    assert rows[0]["platform"] == "youtube"
    assert rows[0]["video_id"] == "vid1"
    assert rows[0]["author"] == "Alice"
    assert rows[0]["text"] == "great drive"
    # The ledger records no reply state yet -- "unknown", not a fabricated bool.
    assert rows[0]["replied"] is None


def test_build_inbox_rows_filters_by_platform():
    comments = [
        _comment("youtube", "vid1", "Alice", "hi", "2026-09-01T00:00:00Z"),
        _comment("tiktok", "vid2", "Bob", "yo", "2026-09-02T00:00:00Z"),
    ]
    rows = build_inbox_rows(comments, platform="tiktok")
    assert [r["author"] for r in rows] == ["Bob"]


def test_build_inbox_rows_unanswered_only_keeps_everything_when_replied_is_unknown():
    comments = [_comment("youtube", "vid1", "Alice", "hi", "2026-09-01T00:00:00Z")]
    rows = build_inbox_rows(comments, unanswered_only=True)
    assert len(rows) == 1


def test_inbox_platform_status_flags_unconnected_platforms_without_fetching():
    comments = [_comment("youtube", "vid1", "Alice", "hi", "2026-09-01T00:00:00Z")]
    status = {row["platform"]: row for row in inbox_platform_status(comments)}
    assert status["youtube"]["status"] == "connected"
    assert status["youtube"]["count"] == 1
    for platform in UNCONNECTED_PLATFORMS:
        assert status[platform]["status"] == "not connected"
        assert status[platform]["count"] == 0
