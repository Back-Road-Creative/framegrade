"""Tests for the social publishing queue drain loop (``dispatch.py``).

Exercised against a real temporary SQLite queue via ``queue.py`` — cheap
enough that mocking it would only hide bugs in the SQL."""

from datetime import datetime, timedelta, timezone

import pytest

from gomoveshift.publishing.dispatch import (
    PermanentDeliveryError,
    ThrottledDeliveryError,
    TransientMediaError,
    _select_due,
    apply_suppression_response,
    reconcile_awaiting_manual,
    compute_next_attempt_at,
    drain_queue,
)
from gomoveshift.publishing.queue import (
    acquire_lease,
    enqueue,
    get_platform_daily_cap,
    is_platform_paused,
    log_response,
    open_queue_db,
    pause_platform,
    set_platform_daily_cap,
    transition,
)
from gomoveshift.publishing.queue import PUBLISHED_AT_PLATFORM_CONFIRMED as CONFIRMED

NOW = "2026-08-23T00:00:00+00:00"
LATER = "2026-08-23T01:00:00+00:00"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _status(conn, delivery_id):
    row = conn.execute("SELECT status, attempt_count FROM deliveries WHERE id=?", (delivery_id,))
    return row.fetchone()


def _row(conn, delivery_id):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


def test_empty_queue_is_success_not_error(conn):
    result = drain_queue(conn, {}, now=NOW)
    assert result.attempted == 0
    assert result.errors == []


def test_successful_handler_transitions_to_published(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    calls = []
    result = drain_queue(conn, {"instagram": calls.append}, now=NOW)
    assert result.succeeded == 1
    assert len(calls) == 1
    status, _ = _status(conn, delivery_id)
    assert status == "published"
    # Synchronous handler success is direct confirmation, not reconciled.
    assert _row(conn, delivery_id)["published_at_source"] == "platform_confirmed"


def test_lease_already_held_is_skipped_not_retried_same_pass(conn):
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    # Another worker holds a live lease past `NOW`.
    assert acquire_lease(conn, delivery_id, "other-worker", LATER, now=NOW)

    def boom(_row):
        raise AssertionError("handler must not run for a leased delivery")

    result = drain_queue(conn, {"tiktok": boom}, now=NOW)
    assert result.skipped_leased == 1
    assert result.attempted == 0
    status, _ = _status(conn, delivery_id)
    assert status == "queued"


def test_handler_exception_default_is_retryable_not_permanent(conn):
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")

    def flaky(_row):
        raise RuntimeError("transient network error")

    result = drain_queue(conn, {"facebook": flaky}, now=NOW)
    assert result.retried == 1
    assert result.failed_permanent == 0
    status, attempt_count = _status(conn, delivery_id)
    assert status == "queued"
    assert attempt_count == 1


def test_handler_permanent_error_marks_failed_permanent(conn):
    delivery_id = enqueue(conn, "proj", 0, "youtube", "hash-a")

    def rejected(_row):
        raise PermanentDeliveryError("platform rejected this content permanently")

    result = drain_queue(conn, {"youtube": rejected}, now=NOW)
    assert result.failed_permanent == 1
    assert result.retried == 0
    status, _ = _status(conn, delivery_id)
    assert status == "failed_permanent"


def test_one_handler_exception_does_not_abort_the_pass(conn):
    boom_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    ok_id = enqueue(conn, "proj", 1, "instagram", "hash-b")

    def boom(_row):
        raise RuntimeError("boom")

    def ok(_row):
        pass

    result = drain_queue(conn, {"facebook": boom, "instagram": ok}, now=NOW)
    assert result.attempted == 2
    assert result.retried == 1
    assert result.succeeded == 1
    assert _status(conn, boom_id)[0] == "queued"
    assert _status(conn, ok_id)[0] == "published"


def test_unregistered_platform_is_skipped_and_left_retryable(conn):
    delivery_id = enqueue(conn, "proj", 0, "snapchat", "hash-a")
    result = drain_queue(conn, {}, now=NOW)
    assert result.skipped_no_handler == 1
    assert result.attempted == 0
    status, _ = _status(conn, delivery_id)
    assert status == "queued"


def test_blocked_no_route_true_when_a_due_delivery_had_no_handler(conn):
    """attempted=0 succeeded=0 failed_permanent=0 is also what a genuinely
    idle pass reports -- ``blocked_no_route`` is how a caller tells the two
    apart."""
    enqueue(conn, "proj", 0, "snapchat", "hash-a")
    result = drain_queue(conn, {}, now=NOW)
    assert result.blocked_no_route is True


def test_blocked_no_route_false_on_a_genuinely_idle_pass(conn):
    result = drain_queue(conn, {}, now=NOW)
    assert result.blocked_no_route is False


def test_unregistered_platform_does_not_hold_the_lease(conn):
    delivery_id = enqueue(conn, "proj", 0, "snapchat", "hash-a")
    drain_queue(conn, {}, now=NOW)
    row = _row(conn, delivery_id)
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None


def test_retryable_failure_sets_a_future_next_attempt_at(conn):
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")

    def flaky(_row):
        raise RuntimeError("transient network error")

    drain_queue(conn, {"facebook": flaky}, now=NOW)
    row = _row(conn, delivery_id)
    assert row["next_attempt_at"] is not None
    assert row["next_attempt_at"] > NOW
    assert row["next_attempt_reason"] == "retry_backoff"


def test_delivery_with_future_next_attempt_at_is_not_selected(conn):
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")

    def flaky(_row):
        raise RuntimeError("transient network error")

    drain_queue(conn, {"facebook": flaky}, now=NOW)
    row = _row(conn, delivery_id)
    assert row["next_attempt_at"] > NOW

    due = _select_due(conn, row["next_attempt_at"][:-1])  # strictly before its due time
    assert delivery_id not in [r["id"] for r in due]


def test_delivery_is_selected_once_next_attempt_at_has_passed(conn):
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")

    def flaky(_row):
        raise RuntimeError("transient network error")

    drain_queue(conn, {"facebook": flaky}, now=NOW)
    row = _row(conn, delivery_id)
    next_attempt_at = row["next_attempt_at"]

    due = _select_due(conn, next_attempt_at)
    assert delivery_id in [r["id"] for r in due]


def test_backoff_grows_and_is_capped():
    base = compute_next_attempt_at(NOW, 1, base_seconds=60, cap_seconds=3600)
    grown = compute_next_attempt_at(NOW, 4, base_seconds=60, cap_seconds=3600)
    capped = compute_next_attempt_at(NOW, 20, base_seconds=60, cap_seconds=3600)
    assert base < grown < capped
    # capped delay is exactly the cap
    from datetime import datetime

    delta = (datetime.fromisoformat(capped) - datetime.fromisoformat(NOW)).total_seconds()
    assert delta == 3600


def test_exceeding_max_attempts_marks_failed_permanent_visibly(conn):
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")

    def flaky(_row):
        raise RuntimeError("transient network error")

    # Drive attempt_count up to max_attempts - 1 by hand, then one more pass
    # through drain_queue should push it over and terminate the delivery.
    from gomoveshift.publishing.queue import transition

    transition(conn, delivery_id, "queued", "queued", attempt_count=2)

    result = drain_queue(conn, {"facebook": flaky}, now=NOW, max_attempts=3)
    assert result.failed_permanent == 1
    row = _row(conn, delivery_id)
    assert row["status"] == "failed_permanent"
    assert "MaxAttempts" in row["error_class"]


def test_retryable_release_lets_backoff_govern_not_the_lease(tmp_path):
    """A retryable failure releases its lease. Holding it meant the 300s lease
    swallowed every backoff shorter than itself — the first three by default —
    so tightening backoff_base_seconds for a canary would do nothing."""
    conn = open_queue_db(tmp_path / "q.db")
    enqueue(conn, project_name="p", short_index=0, platform="youtube", plan_hash="h")

    def boom(row):
        raise RuntimeError("transient")

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    drain_queue(conn, {"youtube": boom}, now=start.isoformat())
    row = dict(next(iter(conn.execute("SELECT * FROM deliveries"))))
    assert row["lease_owner"] is None and row["lease_expires_at"] is None

    # 61s > the 60s first backoff but well inside the 300s lease that used to
    # be held: the delivery must be attempted again, not skipped as leased.
    result = drain_queue(conn, {"youtube": boom}, now=(start + timedelta(seconds=61)).isoformat())
    assert result.attempted == 1 and result.skipped_leased == 0


def test_transient_media_missing_error_never_becomes_failed_permanent_past_max_attempts(conn):
    """The defect guarded against: a handler that can't confirm its media
    still matches disk (unmounted drive, deleted short) must not have that
    treated like ordinary API flakiness — driving attempt_count past
    max_attempts must not terminate the row."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")

    def media_missing(_row):
        raise TransientMediaError("no such file: /drive-d/proj/shorts/lf_01/short_00.mp4")

    from gomoveshift.publishing.queue import transition

    # Already at/over what would be a terminal attempt_count for the
    # ordinary retryable path.
    transition(conn, delivery_id, "queued", "queued", attempt_count=9)

    result = drain_queue(conn, {"facebook": media_missing}, now=NOW, max_attempts=3)
    assert result.failed_permanent == 0
    assert result.retried == 1
    row = _row(conn, delivery_id)
    assert row["status"] == "queued"
    assert row["next_attempt_at"] is not None
    assert row["next_attempt_reason"] == "retry_backoff"
    assert row["lease_owner"] is None and row["lease_expires_at"] is None


def test_shadow_deliveries_are_never_due(conn):
    live = enqueue(conn, "proj", 1, "youtube", "hash")
    enqueue(conn, "proj", 2, "youtube", "hash", shadow=True)
    due = _select_due(conn, NOW)
    assert [row["id"] for row in due] == [live]


def test_drain_never_publishes_a_shadow_delivery(conn):
    # Robust by construction: shadow rows are not selectable, so no drain
    # path -- CLI, cron, or a future worker -- can fire them. A shadow run
    # must be safe even if `dispatch-queue --apply` runs by accident.
    enqueue(conn, "proj", 1, "youtube", "hash", shadow=True)
    published = []
    result = drain_queue(conn, {"youtube": lambda row: published.append(row["id"])}, now=NOW)
    assert published == []
    assert result.attempted == 0
    assert _status(conn, 1)["status"] == "queued"


# --- a delivery is not due before its planned publish time ------------------
#
# `_select_due` gated only on `next_attempt_at`, which `enqueue` leaves NULL,
# so every queued row was due the instant it was spooled. The plan's
# `schedule.scheduled_at_utc` lived only in the sidecar and no drain path
# consulted it. Only the YouTube adapter passes `publish_at` (a server-side
# scheduled upload); Instagram, TikTok and Facebook (`video_state="PUBLISHED"`)
# publish IMMEDIATELY -- so one `dispatch-queue --apply` would have dumped a
# whole week of shorts at once. The schedule is now recorded on the row and
# the SQL selection enforces it, so no caller -- CLI, cron or future worker --
# can publish early by taking a different path.


def test_a_row_scheduled_in_the_future_is_not_due(tmp_path):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H", scheduled_at="2030-01-01T00:00:00+00:00")
    try:
        assert _select_due(conn, "2026-09-05T00:00:00+00:00") == []
    finally:
        conn.close()


def test_a_row_whose_time_has_arrived_is_due(tmp_path):
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H", scheduled_at="2026-09-05T00:00:00+00:00")
    try:
        due = _select_due(conn, "2026-09-05T00:00:01+00:00")
        assert [r["short_index"] for r in due] == [1]
    finally:
        conn.close()


def test_a_row_with_no_recorded_schedule_stays_due(tmp_path):
    """NULL means a row enqueued before this column existed. Treating it as
    never-due would strand it silently -- the same class of failure this
    guard exists to prevent."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H")
    try:
        assert [r["short_index"] for r in _select_due(conn, "2026-09-05T00:00:00+00:00")] == [1]
    finally:
        conn.close()


# --- Blackout dates hold `_select_due` ---------------------------------


def _patch_blackout(monkeypatch, *dates):
    import datetime

    from gomoveshift.publishing import dispatch as dispatch_mod
    from gomoveshift.publishing.brand_policy import BrandPolicy

    parsed = tuple(datetime.date.fromisoformat(d) for d in dates)
    monkeypatch.setattr(
        dispatch_mod,
        "load_brand_policy",
        lambda slug: BrandPolicy(blackout=parsed),
    )


def test_a_row_scheduled_on_a_blackout_date_is_not_selected(tmp_path, monkeypatch):
    _patch_blackout(monkeypatch, "2026-09-05")
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H", scheduled_at="2026-09-05T12:00:00+00:00")
    try:
        assert _select_due(conn, "2026-09-05T12:00:01+00:00") == []
    finally:
        conn.close()


def test_a_row_scheduled_off_a_blackout_date_is_still_selected(tmp_path, monkeypatch):
    _patch_blackout(monkeypatch, "2026-09-05")
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H", scheduled_at="2026-09-06T12:00:00+00:00")
    try:
        due = _select_due(conn, "2026-09-06T12:00:01+00:00")
        assert [r["short_index"] for r in due] == [1]
    finally:
        conn.close()


def test_a_row_with_no_schedule_uses_nows_date_for_the_blackout_check(tmp_path, monkeypatch):
    _patch_blackout(monkeypatch, "2026-09-05")
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H")
    try:
        assert _select_due(conn, "2026-09-05T00:00:00+00:00") == []
        assert [r["short_index"] for r in _select_due(conn, "2026-09-06T00:00:00+00:00")] == [1]
    finally:
        conn.close()


def test_no_blackout_configured_is_unaffected(tmp_path, monkeypatch):
    _patch_blackout(monkeypatch)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H", scheduled_at="2026-09-05T00:00:00+00:00")
    try:
        due = _select_due(conn, "2026-09-05T00:00:01+00:00")
        assert [r["short_index"] for r in due] == [1]
    finally:
        conn.close()


# --- quiet_after_alert_hours holds `_select_due` -----------------------


def _patch_quiet(monkeypatch, hours):
    from gomoveshift.publishing import dispatch as dispatch_mod
    from gomoveshift.publishing.brand_policy import BrandPolicy

    monkeypatch.setattr(
        dispatch_mod,
        "load_brand_policy",
        lambda slug: BrandPolicy(quiet_after_alert_hours=hours),
    )


def test_platform_is_held_inside_the_quiet_window_after_its_alert(tmp_path, monkeypatch):
    _patch_quiet(monkeypatch, 6)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H")
    try:
        log_response(
            conn,
            "instagram",
            state="HEALTHY",
            cap=5,
            action="cap_set",
            mode="live",
            now="2026-09-05T00:00:00+00:00",
        )
        log_response(
            conn,
            "instagram",
            state="SUPPRESSED",
            cap=0,
            action="cap_set",
            mode="live",
            now="2026-09-05T01:00:00+00:00",
        )
        assert _select_due(conn, "2026-09-05T06:59:00+00:00") == []
        due = _select_due(conn, "2026-09-05T07:00:01+00:00")
        assert [r["short_index"] for r in due] == [1]
    finally:
        conn.close()


def test_a_repeated_suppressed_relog_does_not_restart_the_quiet_clock(tmp_path, monkeypatch):
    _patch_quiet(monkeypatch, 6)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H")
    try:
        log_response(
            conn,
            "instagram",
            state="SUPPRESSED",
            cap=0,
            action="cap_set",
            mode="live",
            now="2026-09-05T01:00:00+00:00",
        )
        # A nightly re-log while still SUPPRESSED -- the clock stays
        # anchored to the original transition, not this later row.
        log_response(
            conn,
            "instagram",
            state="SUPPRESSED",
            cap=0,
            action="cap_set",
            mode="live",
            now="2026-09-05T05:00:00+00:00",
        )
        due = _select_due(conn, "2026-09-05T07:00:01+00:00")
        assert [r["short_index"] for r in due] == [1]
    finally:
        conn.close()


def test_a_shadow_mode_suppressed_row_never_holds(tmp_path, monkeypatch):
    _patch_quiet(monkeypatch, 6)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H")
    try:
        log_response(
            conn,
            "instagram",
            state="SUPPRESSED",
            cap=0,
            action="cap_set",
            mode="shadow",
            now="2026-09-05T01:00:00+00:00",
        )
        due = _select_due(conn, "2026-09-05T01:30:00+00:00")
        assert [r["short_index"] for r in due] == [1]
    finally:
        conn.close()


def test_other_platforms_are_unaffected_by_one_platforms_quiet_window(tmp_path, monkeypatch):
    _patch_quiet(monkeypatch, 6)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H")
    enqueue(conn, "proj", 2, "tiktok", "H")
    try:
        log_response(
            conn,
            "instagram",
            state="SUPPRESSED",
            cap=0,
            action="cap_set",
            mode="live",
            now="2026-09-05T01:00:00+00:00",
        )
        due = _select_due(conn, "2026-09-05T01:30:00+00:00")
        assert sorted(r["platform"] for r in due) == ["tiktok"]
    finally:
        conn.close()


class _SpyingConn:
    """Delegates everything to a real connection except ``execute``, which
    asserts no query ever touches ``response_log`` -- proves the ``None``
    case costs zero queries, not just zero holds."""

    def __init__(self, real):
        self._real = real

    def execute(self, sql, *args, **kwargs):
        assert "response_log" not in sql
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_quiet_after_alert_hours_none_makes_no_response_log_query(tmp_path, monkeypatch):
    _patch_quiet(monkeypatch, None)
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H")
    try:
        due = _select_due(_SpyingConn(conn), "2026-09-05T01:30:00+00:00")
        assert [r["short_index"] for r in due] == [1]
    finally:
        conn.close()


def test_drain_publishes_nothing_before_the_scheduled_time(tmp_path):
    """The property that matters: the handler is never called early."""
    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 1, "instagram", "H", scheduled_at="2030-01-01T00:00:00+00:00")
    called = []
    try:
        result = drain_queue(
            conn,
            {"instagram": lambda row: called.append(row["id"])},
            now="2026-09-05T00:00:00+00:00",
        )
    finally:
        conn.close()
    assert called == []
    assert result.attempted == 0


def test_handler_can_reacquire_the_lease_drain_queue_already_holds(conn):
    """Every real adapter re-acquires the lease before its irreversible API
    call, while ``drain_queue`` is already holding it for the same pass. That
    second acquire must succeed, or nothing can ever publish.

    Failure mode: a first real drain attempts all four
    due deliveries (youtube, instagram, facebook, tiktok) and every one raised
    ``RuntimeError("delivery N is leased by another worker")`` -- against its
    own pass, with no other worker running.
    """
    delivery_id = enqueue(conn, "proj", 2, "youtube", "hash-a")
    conn.execute(
        "UPDATE deliveries SET scheduled_at=? WHERE id=?",
        ("2026-08-22T00:00:00+00:00", delivery_id),
    )
    conn.commit()
    owner = "worker-under-test"
    seen = {}

    def handler(row):
        # exactly what youtube/instagram/facebook/tiktok adapters do
        # now=NOW keeps this inside the fixture's clock: the lease drain_queue
        # just took is LIVE here. Defaulting to the real clock would read the
        # fixed-clock expiry as long past and pass for the wrong reason.
        seen["reacquired"] = acquire_lease(conn, row["id"], owner, LATER, now=NOW)

    result = drain_queue(conn, {"youtube": handler}, owner=owner, now=NOW)

    assert result.attempted == 1
    assert seen.get("reacquired") is True, (
        "the adapter could not re-take the lease its own drain pass holds"
    )


def test_a_failed_delivery_records_the_message_not_just_the_class(conn):
    """The row must be enough to diagnose the failure on its own.

    Four deliveries once recorded ``error_class=RuntimeError`` and
    nothing else; the message that explained them ("delivery N is leased by
    another worker") existed only in the drain's stdout, of which the cron
    wrapper keeps one line. Three of the four were unrecoverable.
    """
    delivery_id = enqueue(conn, "proj", 9, "youtube", "hash-a")
    conn.execute(
        "UPDATE deliveries SET scheduled_at=? WHERE id=?",
        ("2026-08-22T00:00:00+00:00", delivery_id),
    )
    conn.commit()

    def handler(row):
        raise RuntimeError("delivery 8 is leased by another worker")

    result = drain_queue(conn, {"youtube": handler}, owner="w", now=NOW)

    assert result.retried == 1
    row = conn.execute(
        "SELECT error_class, error_detail FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()
    assert row[0] == "RuntimeError"
    assert row[1] == "delivery 8 is leased by another worker"


# --- Reconciling a draft a human published from the app ---------------------
#
# An inbox draft parks at `awaiting_manual_publish` with
# next_attempt_at=9999-01-01, so `_select_due` deliberately never returns it
# and the drain cannot close it. Until now only the console's mark-published
# button could, and only if someone clicked. `status/fetch` answers exactly
# once the human taps publish:
#
#     {'status': 'PUBLISH_COMPLETE',
#      'publicaly_available_post_id': [7000000000001]}
#
# (`publicaly_` is TikTok's spelling, not a typo here.)

AWAITING = "awaiting_manual_publish"
PARKED_FOREVER = "9999-01-01T00:00:00+00:00"


def _parked(conn, short_index=0, publish_id="v_inbox_file~v2.1"):
    """A TikTok row in the state the inbox handler leaves behind."""
    delivery_id = enqueue(conn, "proj", short_index, "tiktok", "hash-a")
    conn.execute(
        "UPDATE deliveries SET status=?, next_attempt_at=?, platform_upload_id=? WHERE id=?",
        (AWAITING, PARKED_FOREVER, publish_id, delivery_id),
    )
    conn.commit()
    return delivery_id


def _permalink_for(post_id):
    return f"https://www.tiktok.com/@example_account/video/{post_id}"


def test_publish_complete_closes_the_row_with_post_id_and_permalink(conn):
    delivery_id = _parked(conn)

    def poll(publish_id):
        return {
            "status": "PUBLISH_COMPLETE",
            "publicaly_available_post_id": [7000000000001],
        }

    result = reconcile_awaiting_manual(conn, poll=poll, permalink_for=_permalink_for)

    assert result.published == 1
    row = _row(conn, delivery_id)
    assert row["status"] == "published"
    assert row["platform_post_id"] == "7000000000001"
    assert row["permalink"] == "https://www.tiktok.com/@example_account/video/7000000000001"
    # Delayed/automated confirmation via the TikTok poll, never masquerading
    # as an operator's own console click.
    assert row["published_at_source"] == "platform_reconciled"


def test_publish_complete_without_a_post_id_yet_leaves_the_row_parked(conn):
    """Moderation lags the tap by a minute or two. Publishing the row now
    would store a NULL post id -- the exact ambiguity the YouTube rows have."""
    delivery_id = _parked(conn)

    result = reconcile_awaiting_manual(
        conn, poll=lambda _p: {"status": "PUBLISH_COMPLETE"}, permalink_for=_permalink_for
    )

    assert result.published == 0
    assert _row(conn, delivery_id)["status"] == AWAITING


def test_failed_status_fails_the_row_permanently_with_the_reason(conn):
    delivery_id = _parked(conn)

    def poll(publish_id):
        return {"status": "FAILED", "fail_reason": "picture_size_check_failed"}

    result = reconcile_awaiting_manual(conn, poll=poll, permalink_for=_permalink_for)

    assert result.failed == 1
    row = _row(conn, delivery_id)
    assert row["status"] == "failed_permanent"
    assert "picture_size_check_failed" in row["error_detail"]


def test_a_draft_still_in_the_inbox_is_left_alone(conn):
    delivery_id = _parked(conn)

    result = reconcile_awaiting_manual(
        conn, poll=lambda _p: {"status": "SEND_TO_USER_INBOX"}, permalink_for=_permalink_for
    )

    assert (result.published, result.failed) == (0, 0)
    assert _row(conn, delivery_id)["status"] == AWAITING


def test_a_poll_that_raises_leaves_the_row_parked_and_does_not_propagate(conn):
    """Reconcile is best effort: it runs before the drain, and an API blip
    must never stop the shorts that are due from publishing."""
    delivery_id = _parked(conn)

    def poll(publish_id):
        raise RuntimeError("connection reset")

    result = reconcile_awaiting_manual(conn, poll=poll, permalink_for=_permalink_for)

    assert result.errors and "connection reset" in result.errors[0]
    assert _row(conn, delivery_id)["status"] == AWAITING


def test_a_row_with_no_publish_id_is_skipped_without_calling_the_api(conn):
    delivery_id = _parked(conn, publish_id=None)
    calls = []

    reconcile_awaiting_manual(conn, poll=lambda p: calls.append(p), permalink_for=_permalink_for)

    assert calls == []
    assert _row(conn, delivery_id)["status"] == AWAITING


def test_only_awaiting_manual_publish_rows_are_polled(conn):
    """A queued or published row is none of reconcile's business."""
    _parked(conn, short_index=0, publish_id="v_inbox_file~v2.1")
    enqueue(conn, "proj", 1, "tiktok", "hash-a")  # plain queued row
    calls = []

    reconcile_awaiting_manual(
        conn,
        poll=lambda p: calls.append(p) or {"status": "SEND_TO_USER_INBOX"},
        permalink_for=_permalink_for,
    )

    assert calls == ["v_inbox_file~v2.1"]


def test_the_permalink_is_written_in_the_same_transition_as_the_status(conn):
    """The earlier defect, in a new place: `transition` refuses every write once a
    row is terminal, so a permalink fetched after the publish transition is
    unstorable. One row must never reach `published` without it."""
    delivery_id = _parked(conn)
    seen = []

    def permalink_for(post_id):
        seen.append(_row(conn, delivery_id)["status"])
        return _permalink_for(post_id)

    reconcile_awaiting_manual(
        conn,
        poll=lambda _p: {
            "status": "PUBLISH_COMPLETE",
            "publicaly_available_post_id": [7000000000001],
        },
        permalink_for=permalink_for,
    )

    assert seen == [AWAITING]  # built BEFORE the row went terminal
    assert _row(conn, delivery_id)["permalink"]


# ─── throttled deliveries (a human, not time, clears these) ──────────────


def test_throttled_retries_without_counting_against_max_attempts(conn):
    """A throttle is cleared by a human publishing drafts, never by retrying.

    Burning the 8-attempt budget on it would declare `failed_permanent` a
    delivery whose media and credentials are both fine -- the same reasoning
    that exempts TransientMediaError."""
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    conn.execute("UPDATE deliveries SET attempt_count=99 WHERE id=?", (delivery_id,))
    conn.commit()

    def handler(row):
        raise ThrottledDeliveryError("5 drafts pending")

    result = drain_queue(conn, {"tiktok": handler}, now=NOW, max_attempts=8)

    assert result.retried == 1
    assert result.failed_permanent == 0
    status, attempts = _status(conn, delivery_id)
    assert status == "queued"
    assert attempts == 100


def test_throttled_backs_off_far_longer_than_the_api_curve(conn):
    """Fixed long delay, not `compute_next_attempt_at`: the backoff curve is
    sized for API flakiness and caps at an hour, which would re-ask TikTok
    hourly for a state only a human can change."""
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")

    def handler(row):
        raise ThrottledDeliveryError("5 drafts pending")

    drain_queue(conn, {"tiktok": handler}, now=NOW)

    row = _row(conn, delivery_id)
    next_at = datetime.fromisoformat(row["next_attempt_at"])
    delay = (next_at - datetime.fromisoformat(NOW)).total_seconds()
    assert delay >= 6 * 3600
    assert row["error_class"] == "ThrottledDeliveryError"
    assert "5 drafts pending" in row["error_detail"]
    assert row["next_attempt_reason"] == "platform_throttle"


def test_throttled_releases_the_lease(conn):
    """Same as every other non-terminal exit: a held lease would make the
    next pass report `skipped_leased` instead of retrying."""
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")

    def handler(row):
        raise ThrottledDeliveryError("throttled")

    drain_queue(conn, {"tiktok": handler}, now=NOW)

    row = _row(conn, delivery_id)
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None


def test_throttled_is_reported_distinctly_from_a_generic_retry(conn):
    """The digest keys off this wording to tell the operator that publishing the
    parked drafts is the fix -- a generic 'retryable' line does not."""
    enqueue(conn, "proj", 0, "tiktok", "hash-a")

    def handler(row):
        raise ThrottledDeliveryError("3 drafts pending")

    result = drain_queue(conn, {"tiktok": handler}, now=NOW)

    assert result.throttled == 1
    assert result.retried == 1
    assert any("throttled" in e and "3 drafts pending" in e for e in result.errors)


def test_select_due_excludes_a_paused_platform(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    assert pause_platform(conn, "instagram") is True
    ids = [r["id"] for r in _select_due(conn, NOW)]
    assert delivery_id not in ids


def test_select_due_still_returns_an_unpaused_platform(conn):
    paused_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    live_id = enqueue(conn, "proj", 1, "youtube", "hash-b")
    assert pause_platform(conn, "instagram") is True
    ids = [r["id"] for r in _select_due(conn, NOW)]
    assert paused_id not in ids
    assert live_id in ids


def test_drain_pass_does_not_mutate_a_paused_platforms_row(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    assert pause_platform(conn, "instagram") is True
    before = _row(conn, delivery_id)

    calls = []
    result = drain_queue(conn, {"instagram": calls.append}, now=NOW)

    assert result.attempted == 0
    assert calls == []
    after = _row(conn, delivery_id)
    assert after == before


def test_resume_restores_selection(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    assert pause_platform(conn, "instagram") is True
    assert delivery_id not in [r["id"] for r in _select_due(conn, NOW)]

    from gomoveshift.publishing.queue import resume_platform

    assert resume_platform(conn, "instagram") is True
    assert delivery_id in [r["id"] for r in _select_due(conn, NOW)]


def _expired_lease(conn, delivery_id, *, owner="worker-a:1:dead", now=NOW, seconds=300):
    """A worker that acquired this row and then died: the lease is set, the
    row never transitioned, and the lease has since expired."""
    acquired_at = datetime.fromisoformat(now) - timedelta(seconds=seconds + 1)
    acquire_lease(
        conn,
        delivery_id,
        owner,
        (acquired_at + timedelta(seconds=seconds)).isoformat(),
        now=acquired_at.isoformat(),
    )


def test_an_interrupted_attempt_is_counted_not_forgotten(conn):
    """A worker killed between acquire_lease and any transition leaves
    attempt_count untouched. The lease expires, the row is retried, and if
    whatever killed the worker kills it again the loop never ends: every
    max_attempts check lives in an `except` clause that a SIGKILL never
    reaches. To an operator the row reads 'queued, 0 attempts' forever --
    indistinguishable from one that was never tried."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)

    drain_queue(conn, {"facebook": lambda _row: None}, now=NOW)

    assert _row(conn, delivery_id)["attempt_count"] >= 1


def test_a_repeatedly_interrupted_delivery_eventually_stops(conn):
    """The point of counting: a delivery that kills its worker every time
    must reach failed_permanent like any other repeated failure, instead of
    retrying until someone notices by hand."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    moment = datetime.fromisoformat(NOW)
    for _ in range(3):
        _expired_lease(conn, delivery_id, now=moment.isoformat())
        drain_queue(conn, {}, now=moment.isoformat(), max_attempts=3)
        moment += timedelta(hours=6)

    row = _row(conn, delivery_id)
    assert row["status"] == "failed_permanent"
    assert "Interrupted" in (row["error_class"] or "")


def test_an_interrupted_attempt_says_what_happened(conn):
    """An error_class 'RuntimeError' with no detail is not diagnosable.
    An interrupted attempt has no exception at all, so the row has to say so
    itself or it looks like an attempt that simply never happened."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)

    drain_queue(conn, {}, now=NOW)

    row = _row(conn, delivery_id)
    assert "Interrupted" in (row["error_class"] or "")
    assert row["error_detail"]


def test_reconciling_an_interrupted_attempt_clears_the_dead_workers_lease(conn):
    """The console shows 'held by worker-a' for a worker that no longer
    exists (publishing-status --validate calls this a stale_lease). Counting
    the attempt without clearing the lease would leave that intact."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)

    drain_queue(conn, {}, now=NOW)

    row = _row(conn, delivery_id)
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None


def test_a_live_lease_is_never_reconciled(conn):
    """The accept-case that matters most: a live lease is a worker that is
    running RIGHT NOW, possibly mid platform-API call. Counting its attempt
    or clearing its lease would let a second worker pick up the same
    delivery and publish it twice."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    acquire_lease(conn, delivery_id, "worker-live:2:x", LATER, now=NOW)

    drain_queue(conn, {}, now=NOW)

    row = _row(conn, delivery_id)
    assert row["lease_owner"] == "worker-live:2:x"
    assert row["attempt_count"] == 0


def test_an_unleased_delivery_is_never_reconciled(conn):
    """The other accept-case: a row nobody ever leased was not interrupted,
    it is simply waiting. Charging it an attempt would burn the retry budget
    of every ordinary queued delivery."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")

    drain_queue(conn, {}, now=NOW)

    assert _row(conn, delivery_id)["attempt_count"] == 0


def test_an_interrupted_attempt_backs_off_before_being_retried(conn):
    """Whatever killed the worker is likely still true a second later. The
    reconciled row gets the same backoff an ordinary retryable failure gets,
    so a crash loop does not spin at drain frequency."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)

    drain_queue(conn, {}, now=NOW)

    row = _row(conn, delivery_id)
    assert row["next_attempt_at"] is not None and row["next_attempt_at"] > NOW
    assert row["next_attempt_reason"] == "retry_backoff"


def test_an_interrupted_terminal_delivery_is_left_alone(conn):
    """A row that reached published/cancelled after its lease was taken is
    finished. Reopening it to record an attempt would contradict the receipt
    it already carries."""
    delivery_id = enqueue(conn, "proj", 0, "facebook", "hash-a")
    _expired_lease(conn, delivery_id)
    from gomoveshift.publishing.queue import transition

    transition(conn, delivery_id, "queued", "published", published_at_source="platform_confirmed")

    drain_queue(conn, {}, now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == "published"
    assert row["attempt_count"] == 0


# A handler that returns normally has not always published. The TikTok inbox
# handler parks its row as a draft for a human and returns; the YouTube
# adapter defers a row past its daily upload budget and returns. A TikTok
# delivery once read "published" without ever being posted: the park
# kept the drain's lease, the lease expired, the interrupted-attempt
# reconcile charged an attempt and set a retry time that un-parked it, and
# the next pass took the handler's early return for a parked row as a
# publish.


def test_a_parked_drafts_expired_lease_is_released_without_unparking_it(conn):
    """A parked draft is waiting on a person, not on a worker that died. Only
    the stale lease goes: no attempt charged, the park and its reason kept."""
    delivery_id = _parked(conn)
    conn.execute(
        "UPDATE deliveries SET next_attempt_reason='awaiting_manual' WHERE id=?", (delivery_id,)
    )
    conn.commit()
    _expired_lease(conn, delivery_id)

    result = drain_queue(conn, {}, now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == AWAITING
    assert row["next_attempt_at"] == PARKED_FOREVER
    assert row["next_attempt_reason"] == "awaiting_manual"
    assert row["attempt_count"] == 0
    assert row["lease_owner"] is None
    assert row["lease_expires_at"] is None
    assert result.reconciled_interrupted == 0


def test_select_due_never_returns_a_parked_draft_even_with_a_past_retry_time(conn):
    """The 9999 park time was the only thing keeping a parked draft out of the
    drain, and any write that moves next_attempt_at un-parks it. Parked is a
    status, so the selection excludes the status itself."""
    delivery_id = _parked(conn)
    conn.execute(
        "UPDATE deliveries SET next_attempt_at=?, next_attempt_reason='retry_backoff' WHERE id=?",
        ("2026-08-22T00:00:00+00:00", delivery_id),
    )
    conn.commit()

    assert delivery_id not in [r["id"] for r in _select_due(conn, NOW)]


def test_drain_never_marks_a_parked_draft_published(conn):
    """End to end: the pass that parks the draft, a pass after its
    lease would have expired, and a pass after any retry backoff."""
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")

    def inbox_handler(row):
        if _row(conn, row["id"])["status"] == AWAITING:
            return  # nothing to do until a human publishes it from the app
        transition(conn, row["id"], "queued", "submitting", platform_upload_id="v_inbox_file~v2.1")
        transition(conn, row["id"], "submitting", "uploaded")
        transition(
            conn,
            row["id"],
            "uploaded",
            AWAITING,
            next_attempt_at=PARKED_FOREVER,
            next_attempt_reason="awaiting_manual",
        )

    start = datetime.fromisoformat(NOW)
    results = [
        drain_queue(conn, {"tiktok": inbox_handler}, owner="worker-under-test", now=at.isoformat())
        for at in (start, start + timedelta(hours=1, minutes=1), start + timedelta(hours=3))
    ]

    row = _row(conn, delivery_id)
    assert row["status"] == AWAITING
    assert row["published_at"] is None
    assert row["next_attempt_at"] == PARKED_FOREVER
    assert row["attempt_count"] == 0
    assert row["lease_owner"] is None
    assert [r.succeeded for r in results] == [0, 0, 0]


def test_a_row_its_handler_left_deferred_is_never_marked_published(conn):
    """The same hole on YouTube: over the daily upload budget the adapter
    defers the row to the next UTC midnight and returns normally. The
    deferral releases this pass's lease and stays the only wait; deferred a
    second time, the row must still not read as published."""
    delivery_id = enqueue(conn, "proj", 0, "youtube", "hash-a")
    clock = {"now": NOW}

    def over_budget(row):
        live = _row(conn, row["id"])["status"]
        tomorrow = datetime.fromisoformat(clock["now"]).date() + timedelta(days=1)
        transition(
            conn,
            row["id"],
            live,
            "deferred",
            next_attempt_at=f"{tomorrow.isoformat()}T00:00:00+00:00",
            next_attempt_reason="platform_deferral",
        )

    start = datetime.fromisoformat(NOW)
    results = []
    for at in (start, start + timedelta(hours=1), start + timedelta(days=1)):
        clock["now"] = at.isoformat()
        results.append(
            drain_queue(conn, {"youtube": over_budget}, owner="worker-under-test", now=clock["now"])
        )

    row = _row(conn, delivery_id)
    assert row["status"] == "deferred"
    assert row["published_at"] is None
    assert row["next_attempt_at"] == "2026-08-25T00:00:00+00:00"
    assert row["next_attempt_reason"] == "platform_deferral"
    assert row["attempt_count"] == 0
    assert row["lease_owner"] is None
    assert [r.succeeded for r in results] == [0, 0, 0]
    assert [r.attempted for r in results] == [1, 0, 1]


def test_select_due_honours_a_live_cap_by_scheduled_at_priority(conn):
    ids = [
        enqueue(
            conn, "proj", i, "youtube", f"hash-{i}", scheduled_at=f"2026-09-17T0{i}:00:00+00:00"
        )
        for i in (3, 1, 2)
    ]
    set_platform_daily_cap(conn, "youtube", cap=1, state="SUPPRESSED", mode="live")
    due = _select_due(conn, now="2026-09-17T12:00:00+00:00")
    assert [row["id"] for row in due] == [ids[1]]  # short_index=1 -> earliest scheduled_at


def test_select_due_live_cap_subtracts_deliveries_already_published_today(conn):
    published_id = enqueue(conn, "proj", 90, "youtube", "hash-90")
    assert transition(conn, published_id, "queued", "published", published_at_source=CONFIRMED)
    conn.execute(
        "UPDATE deliveries SET published_at=? WHERE id=?",
        ("2026-09-17T05:00:00+00:00", published_id),
    )
    conn.commit()
    due_id = enqueue(conn, "proj", 91, "youtube", "hash-91")
    set_platform_daily_cap(conn, "youtube", cap=1, state="SUPPRESSED", mode="live")

    due = _select_due(conn, now="2026-09-17T12:00:00+00:00")
    assert due == []
    assert due_id  # sanity: the row exists, just capped out


def test_select_due_shadow_cap_selects_everything(conn):
    ids = [enqueue(conn, "proj", i, "youtube", f"hash-{i}") for i in (1, 2, 3)]
    set_platform_daily_cap(conn, "youtube", cap=1, state="SUPPRESSED", mode="shadow")
    due = _select_due(conn, now="2026-09-17T12:00:00+00:00")
    assert {row["id"] for row in due} == set(ids)


def test_select_due_uncapped_platform_is_unaffected_by_another_platforms_cap(conn):
    ig_id = enqueue(conn, "proj", 1, "instagram", "hash-ig")
    enqueue(conn, "proj", 1, "youtube", "hash-yt")
    set_platform_daily_cap(conn, "youtube", cap=1, state="SUPPRESSED", mode="live")
    due = _select_due(conn, now="2026-09-17T12:00:00+00:00")
    assert ig_id in {row["id"] for row in due}


def test_apply_suppression_response_shadow_mode_never_caps_or_pauses(conn):
    states = {
        "youtube": {f"2026-09-{d:02d}": "SUPPRESSED" for d in range(11, 18)},
    }
    decisions = apply_suppression_response(
        conn,
        brand="gomoveshift",
        states=states,
        today="2026-09-17",
        ladder=(1, 2, 3),
        mode="shadow",
        now="2026-09-17T00:00:00+00:00",
    )
    assert decisions == [
        {
            "platform": "youtube",
            "state": "SUPPRESSED",
            "cap": 1,
            "healthy_days": 0,
            "auto_pause": True,
            "mode": "shadow",
        }
    ]
    assert is_platform_paused(conn, "youtube") is False
    cap_row = get_platform_daily_cap(conn, "youtube")
    assert cap_row["mode"] == "shadow"
    logged_actions = {
        row["action"] for row in conn.execute("SELECT action FROM response_log").fetchall()
    }
    assert logged_actions == {"cap_set", "would_pause"}


def test_apply_suppression_response_live_mode_pauses_on_seven_day_streak(conn):
    states = {
        "youtube": {f"2026-09-{d:02d}": "SUPPRESSED" for d in range(11, 18)},
    }
    decisions = apply_suppression_response(
        conn,
        brand="gomoveshift",
        states=states,
        today="2026-09-17",
        ladder=(1, 2, 3),
        mode="live",
        now="2026-09-17T00:00:00+00:00",
    )
    assert decisions[0]["auto_pause"] is True
    assert is_platform_paused(conn, "youtube") is True
    row = conn.execute("SELECT reason FROM platform_pauses WHERE platform='youtube'").fetchone()
    assert row["reason"] == "auto:suppressed"
    cap_row = get_platform_daily_cap(conn, "youtube")
    assert cap_row["mode"] == "live"
    assert cap_row["cap"] == 1


def test_apply_suppression_response_recovering_steps_the_ladder(conn):
    states = {
        "facebook": {
            **{f"2026-08-{d:02d}": "HEALTHY" for d in range(20, 32)},
            **{f"2026-09-{d:02d}": "HEALTHY" for d in range(1, 17)},
            "2026-09-17": "RECOVERING",
        },
    }
    decisions = apply_suppression_response(
        conn,
        brand="gomoveshift",
        states=states,
        today="2026-09-17",
        ladder=(1, 2, 3),
        mode="live",
        now="2026-09-17T00:00:00+00:00",
    )
    assert decisions[0]["state"] == "RECOVERING"
    assert decisions[0]["cap"] == 3  # well past 14 healthy days -> ladder ceiling
    assert decisions[0]["auto_pause"] is False
    assert is_platform_paused(conn, "facebook") is False


# ─── a failure is recorded from the row's LIVE status ────────────────────
#
# Real handlers move the row themselves before they can fail (Instagram goes
# queued -> submitting -> uploaded; TikTok submitting -> uploaded ->
# awaiting_manual_publish). The compare-and-set in ``transition`` expected the
# status read when the row was SELECTED, so a permanent
# failure of an Instagram delivery wrote nothing (row still 'uploaded', no
# error text) while the summary still counted failed_permanent=1.

OWNER = "drain-under-test"


def _moves_then_raises(conn, exc, to_status="uploaded", **move_fields):
    """A handler that advances its own row, as the adapters do, then fails."""

    def handler(row):
        assert transition(conn, row["id"], row["status"], to_status, **move_fields)
        # The move really happened: a failing handler that never moved its
        # row would not exercise the stale-status path at all.
        assert _status(conn, row["id"])[0] == to_status
        raise exc

    return handler


def test_permanent_failure_after_the_handler_moved_the_row_is_recorded(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    handler = _moves_then_raises(conn, PermanentDeliveryError("container rejected"))

    result = drain_queue(conn, {"instagram": handler}, owner=OWNER, now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == "failed_permanent"
    assert row["error_class"] == "PermanentDeliveryError"
    assert "container rejected" in row["error_detail"]
    assert result.failed_permanent == 1
    assert result.retried == 0
    assert any("permanent" in e and "container rejected" in e for e in result.errors)


def test_generic_failure_after_the_handler_moved_the_row_backs_off_and_releases(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    handler = _moves_then_raises(conn, RuntimeError("upload dropped"))

    result = drain_queue(conn, {"instagram": handler}, owner=OWNER, now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == "uploaded"
    assert row["attempt_count"] == 1
    assert row["next_attempt_at"] > NOW
    assert row["next_attempt_reason"] == "retry_backoff"
    assert row["error_class"] == "RuntimeError"
    assert "upload dropped" in row["error_detail"]
    assert row["lease_owner"] is None and row["lease_expires_at"] is None
    assert result.retried == 1
    assert result.failed_permanent == 0


def test_generic_failure_past_max_attempts_after_a_move_is_failed_permanent(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    conn.execute("UPDATE deliveries SET attempt_count=2 WHERE id=?", (delivery_id,))
    conn.commit()
    handler = _moves_then_raises(conn, RuntimeError("still failing"))

    result = drain_queue(conn, {"instagram": handler}, owner=OWNER, now=NOW, max_attempts=3)

    row = _row(conn, delivery_id)
    assert row["status"] == "failed_permanent"
    assert row["attempt_count"] == 3
    assert "MaxAttempts" in row["error_class"]
    assert "still failing" in row["error_detail"]
    assert result.failed_permanent == 1


def test_transient_media_failure_after_a_move_takes_its_normal_branch(conn):
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    conn.execute("UPDATE deliveries SET attempt_count=99 WHERE id=?", (delivery_id,))
    conn.commit()
    handler = _moves_then_raises(conn, TransientMediaError("drive unmounted"))

    result = drain_queue(conn, {"instagram": handler}, owner=OWNER, now=NOW, max_attempts=8)

    row = _row(conn, delivery_id)
    assert row["status"] == "uploaded"
    assert row["attempt_count"] == 100
    assert row["next_attempt_reason"] == "retry_backoff"
    assert row["error_class"] == "TransientMediaError"
    assert row["lease_owner"] is None
    assert result.retried == 1
    assert result.failed_permanent == 0
    assert any("transient media" in e and "drive unmounted" in e for e in result.errors)


def test_throttled_failure_after_a_move_takes_its_normal_branch(conn):
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    handler = _moves_then_raises(conn, ThrottledDeliveryError("5 drafts pending"))

    result = drain_queue(conn, {"tiktok": handler}, owner=OWNER, now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == "uploaded"
    assert row["attempt_count"] == 1
    assert row["next_attempt_reason"] == "platform_throttle"
    assert row["error_class"] == "ThrottledDeliveryError"
    assert row["lease_owner"] is None
    assert result.retried == 1
    assert result.throttled == 1
    assert any("throttled" in e and "5 drafts pending" in e for e in result.errors)


@pytest.mark.parametrize(
    "exc",
    [
        PermanentDeliveryError("late permanent"),
        RuntimeError("late generic"),
        TransientMediaError("late transient"),
        ThrottledDeliveryError("late throttle"),
    ],
    ids=["permanent", "generic", "transient", "throttled"],
)
def test_a_row_the_handler_made_terminal_is_never_overwritten_or_counted(conn, exc):
    """The handler published the row and then something after it raised. The
    delivery IS published: it is not a failure, its receipt must survive, and
    the summary must not claim a failure or a retry that never happened."""
    delivery_id = enqueue(conn, "proj", 0, "instagram", "hash-a")
    handler = _moves_then_raises(
        conn,
        exc,
        to_status="published",
        published_at_source=CONFIRMED,
        platform_post_id="media-1",
    )

    result = drain_queue(conn, {"instagram": handler}, owner=OWNER, now=NOW)

    row = _row(conn, delivery_id)
    assert row["status"] == "published"
    assert row["platform_post_id"] == "media-1"
    assert row["error_class"] is None and row["error_detail"] is None
    assert row["attempt_count"] == 0
    assert row["next_attempt_at"] is None
    assert row["lease_owner"] is None and row["lease_expires_at"] is None
    assert result.attempted == 1
    assert result.failed_permanent == 0
    assert result.retried == 0
    assert result.throttled == 0
    (line,) = result.errors
    assert f"instagram#{delivery_id}" in line
    assert "published" in line
    assert str(exc) in line
