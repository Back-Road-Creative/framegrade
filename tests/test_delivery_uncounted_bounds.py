"""Every retry that does not count against ``max_attempts`` is bounded in time,
and every start-over gives the fresh upload a fresh attempt budget.

Before: an unknown Facebook outcome re-read its status every hour for good with
no alert; a Facebook/Instagram row that started over kept the attempt count its
uncounted reads had run up, so the fresh upload failed permanently on its first
ordinary error; an Instagram container still processing spent the row's 8
ordinary tries in ~3 h and failed it while the container lived for 24 h.

New symbols are read off the modules inside each test, so on the unfixed code
these fail on their assertions, not at import."""

import ast
import inspect
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest
import requests
from freezegun import freeze_time
from test_publishing_facebook_adapter import CREDS, _resp
from test_publishing_facebook_reconcile import UNSENT_STATUS_BODY, _fresh_upload_post
from test_publishing_instagram_adapter import _Clock, _FakeReels
from test_publishing_models import _record

from gomoveshift.health.store import load_all
from gomoveshift.publishing import dispatch, handlers, queue
from gomoveshift.publishing import facebook_adapter as facebook
from gomoveshift.publishing.dispatch import ThrottledDeliveryError, TransientMediaError, drain_queue
from gomoveshift.publishing.instagram_adapter import InstagramShortsAdapter
from gomoveshift.publishing.queue import acquire_lease, enqueue, open_queue_db, transition

T0 = datetime.fromisoformat("2026-11-10T12:00:00+00:00")
FB_UNKNOWN = "awaiting_facebook_reconcile"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _row(conn, delivery_id):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


def _seed(conn, platform, *, status="queued", upload_id=None, attempts=0):
    record = _record()
    delivery = enqueue(conn, record.project_name, record.short_index, platform, "h1")
    fields = {"attempt_count": attempts}
    if upload_id:
        fields.update(platform_upload_id=upload_id, upload_account="page-1")
    transition(conn, delivery, "queued", status, **fields)
    return delivery


def _raises(make):
    def handler(_row):
        raise make()

    return handler


#: (name, platform, seeded status, upload id, handler or None for a dying worker)
EXEMPT_KINDS = [
    ("transient media", "instagram", "queued", None, _raises(lambda: TransientMediaError("gone"))),
    ("throttled", "tiktok", "queued", None, _raises(lambda: ThrottledDeliveryError("pending"))),
    (
        "reconcile pending",
        "instagram",
        "uploaded",
        "c-1",
        _raises(lambda: dispatch.ReconcilePendingError("still processing")),
    ),
    ("facebook unknown", "facebook", FB_UNKNOWN, "v-1", _raises(lambda: RuntimeError("read"))),
    (
        "facebook unknown, permanent read",
        "facebook",
        FB_UNKNOWN,
        "v-1",
        _raises(lambda: dispatch.PermanentDeliveryError("account changed")),
    ),
    ("facebook unknown, worker dies", "facebook", FB_UNKNOWN, "v-1", None),
]


@pytest.mark.parametrize(
    "platform, status, upload_id, handler",
    [k[1:] for k in EXEMPT_KINDS],
    ids=[k[0] for k in EXEMPT_KINDS],
)
def test_every_uncounted_retry_parks_for_a_person_once_it_outlives_the_bound(
    conn, tmp_path, platform, status, upload_id, handler
):
    incidents = tmp_path / "incidents"
    delivery = _seed(conn, platform, status=status, upload_id=upload_id)
    calls = []

    def counting(row):
        calls.append(row["id"])
        handler(row)

    parked = 0
    for hour in range(30):
        now = T0 + timedelta(hours=hour)
        if handler is None:  # a worker that took the row and died holding it
            acquire_lease(conn, delivery, "dead", now.isoformat(), now=now.isoformat())
        routes = {platform: counting} if handler else {}
        result = drain_queue(
            conn, routes, now=now.isoformat(), max_attempts=3, incidents_dir=incidents
        )
        parked += result.needs_operator
        state = _row(conn, delivery)
        assert state["status"] != "failed_permanent"
        if now - T0 < timedelta(hours=24):
            assert state["status"] == status, f"parked early at hour {hour}"
    calls_when_parked = len(calls)
    drain_queue(
        conn,
        {platform: counting},
        now=(T0 + timedelta(days=3)).isoformat(),
        incidents_dir=incidents,
    )
    state = _row(conn, delivery)
    assert state["status"] == queue.STATUS_NEEDS_OPERATOR
    assert state["next_attempt_reason"] == queue.NEXT_ATTEMPT_NEEDS_OPERATOR
    assert state["platform_upload_id"] == upload_id  # kept: never re-uploaded
    assert state["lease_owner"] is None
    assert len(calls) == calls_when_parked  # never selected again
    assert parked == (0 if handler is None else 1)
    rows, _ = load_all(incidents)
    open_rows = [r for r in rows if not r.get("resolved_at")]
    assert [(r["kind"], r["delivery_id"]) for r in open_rows] == [
        (dispatch.NEEDS_OPERATOR_KIND, delivery)
    ]


def test_an_ordinary_failure_ends_an_uncounted_run(conn, tmp_path):
    delivery = _seed(conn, "instagram")
    transient = _raises(lambda: TransientMediaError("gone"))
    drain_queue(conn, {"instagram": transient}, now=T0.isoformat(), incidents_dir=tmp_path)
    assert _row(conn, delivery)["uncounted_since"] == T0.isoformat()
    later = T0 + timedelta(hours=2)
    flaky = _raises(lambda: ConnectionError("blip"))
    drain_queue(conn, {"instagram": flaky}, now=later.isoformat(), incidents_dir=tmp_path)
    assert _row(conn, delivery)["uncounted_since"] is None


def test_every_branch_that_skips_max_attempts_goes_through_the_bounded_helper():
    """Structural: a new exemption (an exception class documented EXEMPT from
    ``max_attempts``, or the unknown-Facebook test) must be wired through
    ``_uncounted_retry``, never its own ``transition``."""
    source = inspect.getsource(dispatch)
    tree = ast.parse(source)
    exempt = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and "EXEMPT from ``max_attempts``" in " ".join((ast.get_docstring(node) or "").split())
    }
    assert {"TransientMediaError", "ThrottledDeliveryError", "ReconcilePendingError"} <= exempt
    markers = exempt | {"_facebook_unknown"}

    def names(expr, negated=False):
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            return set()
        found = set()
        for child in ast.iter_child_nodes(expr):
            found |= names(child)
        if isinstance(expr, ast.Name) and expr.id in markers:
            found.add(expr.id)
        return found

    def calls_helper(body):
        return any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_uncounted_retry"
            for stmt in body
            for n in ast.walk(stmt)
        )

    wired = set()
    branches = 0
    for func in tree.body:
        if not isinstance(func, ast.FunctionDef) or func.name in ("_facebook_unknown",):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.If) and names(node.test):
                branches += 1
                assert calls_helper(node.body), (
                    f"{func.name} line {node.lineno}: a branch on {sorted(names(node.test))} "
                    "must record its retry through _uncounted_retry"
                )
                wired |= names(node.test)
    assert branches >= 3 and wired == markers


def test_a_restarted_delivery_gets_a_fresh_attempt_budget(conn):
    delivery = _seed(conn, "instagram", status="uploaded", upload_id="c-1", attempts=20)
    assert queue.restart_delivery(conn, delivery, "uploaded")
    flaky = _raises(lambda: ConnectionError("blip"))
    result = drain_queue(conn, {"instagram": flaky}, now=T0.isoformat(), max_attempts=8)
    state = _row(conn, delivery)
    assert result.retried == 1 and result.failed_permanent == 0
    assert (state["status"], state["attempt_count"], state["platform_upload_id"]) == (
        "queued",
        1,
        None,
    )


@freeze_time(T0)
def test_facebook_start_over_does_not_inherit_the_unknown_reads_attempts(conn):
    delivery = _seed(conn, "facebook", status=FB_UNKNOWN, upload_id="v-1", attempts=20)
    seen = []
    broke = requests.exceptions.ConnectionError("write timed out")
    session = Mock(
        post=Mock(side_effect=_fresh_upload_post(seen, bytes_reply=broke)),
        get=Mock(return_value=_resp(200, UNSENT_STATUS_BODY)),
    )
    adapter = facebook.FacebookShortsAdapter(
        facebook.FacebookReelsAdapter(CREDS, session=session), worker_id="w1"
    )
    record = _record()

    def handler(_row):
        adapter.publish(record, conn=conn, delivery_id=delivery, video_bytes=b"x")

    result = drain_queue(
        conn, {"facebook": handler}, owner="w1", now=T0.isoformat(), max_attempts=8
    )
    state = _row(conn, delivery)
    assert seen == ["start", None]  # the fresh upload ran and its bytes broke off
    assert result.retried == 1 and result.failed_permanent == 0
    assert (state["status"], state["attempt_count"]) == ("queued", 1)


class _ExpiredThenOffline(_FakeReels):
    def create_container(self, payload):
        raise ConnectionError("graph unreachable")


def _instagram(conn, reels, *, attempts):
    delivery = _seed(conn, "instagram", status="uploaded", upload_id="c-old", attempts=attempts)
    clock = _Clock()
    adapter = InstagramShortsAdapter(
        reels, worker_id="w1", sleep=clock.sleep, monotonic=clock.monotonic
    )
    record = _record()
    handler = handlers.make_instagram_handler(adapter, conn, lambda *_: record, lambda _r: b"x")
    return delivery, handler


@freeze_time(T0)
def test_instagram_expired_start_over_does_not_inherit_old_attempts(conn, monkeypatch):
    monkeypatch.setattr("gomoveshift.publishing.instagram_adapter.make_in_spec_bytes", lambda b: b)
    delivery, handler = _instagram(conn, _ExpiredThenOffline("EXPIRED"), attempts=20)
    result = drain_queue(conn, {"instagram": handler}, owner="w1", now=T0.isoformat())
    state = _row(conn, delivery)
    assert result.retried == 1 and result.failed_permanent == 0
    assert state["attempt_count"] == 1 and state["platform_upload_id"] is None


def test_a_container_still_processing_is_polled_outside_max_attempts_then_starts_over(
    conn, tmp_path, monkeypatch
):
    """Delivery #391's shape: the container was still processing on every one
    of 8 tries and the row failed for good. Now it is polled again, uncounted,
    never answered with a second container while IN_PROGRESS, and once it
    EXPIRES it starts over on a fresh one with a fresh budget."""
    monkeypatch.setattr("gomoveshift.publishing.instagram_adapter.make_in_spec_bytes", lambda b: b)
    reels = _FakeReels("IN_PROGRESS")
    delivery, handler = _instagram(conn, reels, attempts=0)
    for hour in range(12):
        now = T0 + timedelta(hours=hour)
        with freeze_time(now):
            result = drain_queue(
                conn,
                {"instagram": handler},
                owner="w1",
                now=now.isoformat(),
                max_attempts=8,
                incidents_dir=tmp_path,
            )
        assert result.failed_permanent == 0
    state = _row(conn, delivery)
    assert state["status"] == "uploaded" and state["platform_upload_id"] == "c-old"
    assert state["attempt_count"] == 12 and state["error_class"] == "ReconcilePendingError"
    assert reels.created == [] and reels.published == []

    reels.statuses = ["EXPIRED", "FINISHED"]
    now = T0 + timedelta(hours=24, minutes=1)
    with freeze_time(now):
        result = drain_queue(conn, {"instagram": handler}, owner="w1", now=now.isoformat())
    state = _row(conn, delivery)
    assert result.succeeded == 1 and state["status"] == "published"
    assert state["platform_upload_id"] == "c-new" and reels.published == ["c-new"]
