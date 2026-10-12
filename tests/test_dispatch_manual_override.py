"""A manual console override sticks through the next automatic
``apply_suppression_response`` run.

An operator's ``platform_daily_caps`` row is distinguished from an automatic
suppression decision purely by its ``reason`` prefix (``"manual:..."`` vs.
``"w2:<brand>:<state>"``) -- there is no separate flag column, so the check
lives against that convention. A manual row must not be silently clobbered
the next time the nightly run classifies that platform, or an operator's
"stop touching this platform" is undone by the very system it overrode.
"""

from __future__ import annotations

import pytest

from gomoveshift.publishing.dispatch import apply_suppression_response
from gomoveshift.publishing.queue import (
    clear_platform_daily_cap,
    get_platform_daily_cap,
    open_queue_db,
    set_platform_daily_cap,
)


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def test_manual_override_is_not_overwritten_by_apply_suppression_response(conn):
    set_platform_daily_cap(
        conn, "youtube", cap=5, state="HEALTHY", mode="live", reason="manual:HEALTHY"
    )

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

    cap_row = get_platform_daily_cap(conn, "youtube")
    assert cap_row["cap"] == 5
    assert cap_row["state"] == "HEALTHY"
    assert cap_row["reason"] == "manual:HEALTHY"
    # The decision summary still reports on the platform (so the nightly
    # print line accounts for every classified platform) but never claims
    # to have auto-paused it.
    assert decisions[0]["platform"] == "youtube"
    assert decisions[0]["auto_pause"] is False


def test_manual_override_logs_skipped_manual_not_cap_set(conn):
    set_platform_daily_cap(
        conn, "instagram", cap=2, state="RECOVERING", mode="shadow", reason="manual:RECOVERING"
    )

    apply_suppression_response(
        conn,
        brand="gomoveshift",
        states={"instagram": {"2026-09-17": "HEALTHY"}},
        today="2026-09-17",
        ladder=(1, 2, 3),
        mode="shadow",
        now="2026-09-17T00:00:00+00:00",
    )

    actions = {row["action"] for row in conn.execute("SELECT action FROM response_log").fetchall()}
    assert actions == {"skipped-manual"}


def test_non_manual_cap_is_still_overwritten(conn):
    set_platform_daily_cap(
        conn, "tiktok", cap=1, state="SUPPRESSED", mode="shadow", reason="w2:gomoveshift:suppressed"
    )

    apply_suppression_response(
        conn,
        brand="gomoveshift",
        states={"tiktok": {"2026-09-17": "HEALTHY"}},
        today="2026-09-17",
        ladder=(1, 2, 3),
        mode="shadow",
        now="2026-09-17T00:00:00+00:00",
    )

    cap_row = get_platform_daily_cap(conn, "tiktok")
    assert cap_row["reason"] == "w2:gomoveshift:healthy"


def test_clear_platform_daily_cap_removes_the_row(conn):
    set_platform_daily_cap(
        conn, "facebook", cap=3, state="HEALTHY", mode="live", reason="manual:HEALTHY"
    )

    removed = clear_platform_daily_cap(conn, "facebook")

    assert removed is True
    assert get_platform_daily_cap(conn, "facebook") is None


def test_clear_platform_daily_cap_on_a_missing_row_returns_false(conn):
    assert clear_platform_daily_cap(conn, "youtube") is False
