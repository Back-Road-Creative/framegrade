"""A refused access token pauses its platform instead of failing every queued
post. Before, ``handlers.py`` turned each platform's auth-blocked error into
``PermanentDeliveryError``, so one expired Facebook, Instagram or TikTok login
failed row 1, then row 2, and so on, with no pause and no alert.

The guard: an auth error on row 1 leaves rows 2..n queued and untouched,
pauses the platform with ``AUTH_BLOCKED_PAUSE_REASON``, and ``health check``
rule 16 raises exactly one incident naming the platform."""

from __future__ import annotations

import dataclasses

import pytest
from test_publishing_models import _record

from gomoveshift.health.check import auth_blocked_incidents
from gomoveshift.publishing.dispatch import drain_queue
from gomoveshift.publishing.facebook_adapter import FacebookAuthBlockedError
from gomoveshift.publishing.handlers import (
    PlatformAuthPausedError,
    TikTokCredentials,
    build_handlers,
)
from gomoveshift.publishing.instagram_adapter import InstagramAuthBlockedError
from gomoveshift.publishing.queue import (
    AUTH_BLOCKED_PAUSE_REASON,
    AUTH_PAUSED_ERROR_CLASS,
    enqueue,
    open_queue_db,
    resume_platform,
    transition,
)
from gomoveshift.publishing.tiktok_adapter import TikTokAuthBlockedError, _throttle_type

NOW = "2026-11-26T12:00:00+00:00"
ROWS = 3


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _loader(project, index):
    return dataclasses.replace(_record(), project_name=project, short_index=index)


class _AuthFailsAdapter:
    """Moves its row to ``submitting`` as the real adapters do, then has the
    platform refuse the token."""

    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.calls = []

    def publish(self, record, *, conn, delivery_id, **_kw):
        self.calls.append(delivery_id)
        transition(conn, delivery_id, "queued", "submitting")
        raise self.exc_type("[graph http=400 code=190] Meta access token expired or revoked")


def _handlers(conn, platform, monkeypatch):
    if platform == "tiktok":
        import gomoveshift.publishing.handlers as handlers_mod

        calls = []

        def refuse(token):
            calls.append(token)
            raise TikTokAuthBlockedError(
                "creator_info failed: HTTP 401 code='access_token_invalid'"
            )

        monkeypatch.setenv("GMS_TIKTOK_MODE", "direct")
        monkeypatch.setattr(handlers_mod, "query_creator_info", refuse)
        creds = TikTokCredentials(access_token="act.fake", audit_approved=True)  # noqa: S106
        return build_handlers(conn, _loader, tiktok=(creds, lambda r: b"v")), calls
    exc = FacebookAuthBlockedError if platform == "facebook" else InstagramAuthBlockedError
    adapter = _AuthFailsAdapter(exc)
    return build_handlers(conn, _loader, **{platform: (adapter, lambda r: b"v")}), adapter.calls


@pytest.mark.parametrize("platform", ["facebook", "instagram", "tiktok"])
def test_an_auth_error_pauses_the_platform_and_leaves_every_row_queued(
    conn, tmp_path, platform, monkeypatch
):
    ids = [enqueue(conn, "example-2026-08", i, platform, "hash-a") for i in range(ROWS)]
    handlers, calls = _handlers(conn, platform, monkeypatch)

    result = drain_queue(conn, handlers, now=NOW, incidents_dir=tmp_path / "incidents")

    assert len(calls) == 1  # row 1 only: the pause held rows 2..n in the same pass
    assert result.failed_permanent == 0
    rows = [dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (i,)).fetchone()) for i in ids]
    assert [r["status"] for r in rows] == ["queued"] * ROWS
    assert rows[0]["error_class"] == AUTH_PAUSED_ERROR_CLASS
    for untouched in rows[1:]:
        assert untouched["attempt_count"] == 0 and untouched["error_class"] is None
    pause = conn.execute(
        "SELECT reason FROM platform_pauses WHERE platform=?", (platform,)
    ).fetchone()
    assert pause is not None and pause["reason"].startswith(AUTH_BLOCKED_PAUSE_REASON + " ")

    # A later pass touches nothing while the platform stays paused.
    assert drain_queue(conn, handlers, now="2026-11-27T12:00:00+00:00").attempted == 0

    conn.commit()
    incidents = auth_blocked_incidents(tmp_path / "seo" / "social_queue.sqlite3", NOW)
    assert [i["key"] for i in incidents] == [f"platform_auth_blocked:{platform}"]
    assert incidents[0]["platform"] == platform

    # Resuming the platform (after re-authorizing) resolves the incident.
    resume_platform(conn, platform)
    assert auth_blocked_incidents(tmp_path / "seo" / "social_queue.sqlite3", NOW) == []


def test_the_paused_error_class_name_is_the_one_resume_looks_for():
    assert PlatformAuthPausedError.__name__ == AUTH_PAUSED_ERROR_CLASS


@pytest.mark.parametrize("platform", ["facebook", "instagram", "tiktok"])
def test_resuming_makes_the_row_that_hit_the_refusal_due_on_the_next_pass(
    conn, tmp_path, platform, monkeypatch
):
    """Row 1 got the 6 h throttle delay when the token was refused. Once the
    platform is resumed it goes on the next pass with the rest, not 6 h later."""
    ids = [enqueue(conn, "example-2026-08", i, platform, "hash-a") for i in range(ROWS)]
    handlers, _calls = _handlers(conn, platform, monkeypatch)
    drain_queue(conn, handlers, now=NOW, incidents_dir=tmp_path / "incidents")

    assert resume_platform(conn, platform) is True
    seen = []
    after = drain_queue(
        conn,
        {platform: lambda row: seen.append(row["id"])},
        now="2026-11-26T13:00:00+00:00",  # 1 h later, well inside the 6 h delay
        incidents_dir=tmp_path / "incidents",
    )

    assert sorted(seen) == sorted(ids)
    assert after.attempted == ROWS


def test_resume_leaves_other_throttled_rows_on_their_delay(conn):
    held = enqueue(conn, "example-2026-08", 0, "facebook", "hash-a")
    with conn:
        conn.execute(
            "UPDATE deliveries SET next_attempt_at=?, next_attempt_reason='platform_throttle', "
            "error_class='ThrottledDeliveryError' WHERE id=?",
            ("2026-11-26T18:00:00+00:00", held),
        )
    resume_platform(conn, "facebook")
    row = conn.execute("SELECT next_attempt_at FROM deliveries WHERE id=?", (held,)).fetchone()
    assert row["next_attempt_at"] == "2026-11-26T18:00:00+00:00"


def test_a_manual_pause_raises_no_auth_incident(conn, tmp_path):
    from gomoveshift.publishing.queue import pause_platform

    pause_platform(conn, "facebook", now=NOW)
    pause_platform(conn, "instagram", now=NOW, reason="auto:suppressed")
    assert auth_blocked_incidents(tmp_path / "seo" / "social_queue.sqlite3", NOW) == []


def test_tiktok_names_a_refused_token_by_its_own_error_codes():
    for code in ("access_token_invalid", "scope_not_authorized"):
        detail = f"HTTP 401 code={code!r} message='x' log_id='y'"
        assert _throttle_type(detail, RuntimeError) is TikTokAuthBlockedError
    assert _throttle_type("HTTP 401 code='unauthorized'", RuntimeError) is RuntimeError
