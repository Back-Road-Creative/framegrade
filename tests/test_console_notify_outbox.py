"""Notification correctness (A1): an alert is recorded only once it was DELIVERED.

Before this, ``check_missed_slots`` and ``check_pending_gates`` recorded "seen" before (or
regardless of) the send, and ``notify()`` answered ``True`` when no destination existed, so an
unconfigured or malformed destination silently consumed every alert. These tests pin the repaired
contract: an undelivered alert stays pending and is retried (bounded, backed off, durable across
a restart), "no usable destination" is its own result, and ``/healthz`` reports the effective
configuration through the same resolver ``notify()`` uses.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import notify
from gomoveshift.console.app import create_app
from gomoveshift.publishing.queue import enqueue, open_queue_db, transition

_TOPIC = "gms-synthetic_topic_0123"
_NTFY_URL = f"https://ntfy.sh/{_TOPIC}"


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch, tmp_path):
    """Retry on the very next pass (the backoff itself is tested below), in a tmp HOME."""
    monkeypatch.setenv("GMS_NOTIFY_BACKOFF_MIN", "0")
    monkeypatch.delenv("GMS_NOTIFY_MAX_ATTEMPTS", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GMS_CONFIG_HOME", str(home / ".config" / "gomoveshift"))
    monkeypatch.delenv("GMS_CONSOLE_NOTIFY_URL", raising=False)


@pytest.fixture
def sends(monkeypatch):
    """Replace ``notify.notify``: ``sends.result`` is what it answers; every call is recorded."""
    box = SimpleNamespace(calls=[], result=False)

    def _fake(event, identity, status):
        box.calls.append((event, identity, status))
        return box.result

    monkeypatch.setattr(notify, "notify", _fake)
    return box


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _late_row(tmp_path):
    conn = open_queue_db(tmp_path)
    # A fixed past instant is late under any real clock, so the test never reads one.
    when = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
    delivery_id = enqueue(conn, "proj", 1, "instagram", "h1", scheduled_at=when)
    conn.close()
    return delivery_id


def _gate(tmp_path, name="gate_a"):
    output_dir = tmp_path / "out"
    (output_dir / "_gates").mkdir(parents=True, exist_ok=True)
    (output_dir / "_gates" / f"{name}.question.json").write_text(
        json.dumps({"question": "Episode breakdown correct?"})
    )
    return str(output_dir)


# --- A-1: a send that did not go out leaves the item unrecorded --------------------------------


def test_missed_slot_is_not_recorded_when_the_send_fails_and_the_next_pass_retries(tmp_path, sends):
    _late_row(tmp_path)
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 1
    seen = tmp_path / "notify_seen_missed_slots.json"
    assert not seen.exists() or json.loads(seen.read_text()) == {}

    sends.result = True
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 2  # exactly one more attempt, then delivered -> never again


def test_gates_are_not_recorded_when_the_send_fails_and_both_events_retry(
    tmp_path, sends, monkeypatch
):
    monkeypatch.setenv("GMS_GATE_STALE_MIN", "0")
    out = _gate(tmp_path)
    notify.check_pending_gates([out], data_dir=tmp_path)
    assert [c[0] for c in sends.calls] == ["gate_pending", "gate_stale"]
    seen = tmp_path / "notify_seen_gates.json"
    assert not seen.exists() or all(not v for v in json.loads(seen.read_text()).values())

    sends.result = True
    sends.calls.clear()
    notify.check_pending_gates([out], data_dir=tmp_path)
    notify.check_pending_gates([out], data_dir=tmp_path)
    assert [c[0] for c in sends.calls] == ["gate_pending", "gate_stale"]


def test_an_unavailable_destination_is_never_recorded_as_delivered(tmp_path, sends):
    sends.result = notify.UNAVAILABLE
    _late_row(tmp_path)
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 2  # still pending, retried every pass
    sends.result = True
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 3


# --- A-2: no usable destination is its own result, not success ---------------------------------


def test_no_destination_is_unavailable_not_true():
    result = notify.notify("job_failed", {"job_id": 1}, "failed")
    assert result is notify.UNAVAILABLE
    assert result is not True


@pytest.mark.parametrize(
    "url", ["https://ntfy.sh/", "https://ntfy.sh", "https://ntfy.sh/bad topic!"]
)
def test_an_ntfy_url_with_an_invalid_topic_is_unavailable_not_true(monkeypatch, url):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", url)

    def _boom(*args, **kwargs):
        raise AssertionError("no request for an unusable destination")

    monkeypatch.setattr(notify.urllib.request, "urlopen", _boom)
    assert notify.notify("job_failed", {"job_id": 1}, "failed") is notify.UNAVAILABLE


def test_a_delivered_push_is_true(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    monkeypatch.setattr(notify.urllib.request, "urlopen", lambda req, timeout=None: None)
    assert notify.notify("job_failed", {"job_id": 1}, "failed") is True


# --- A-3: durable pending state, bounded retry and backoff -------------------------------------


def test_retry_is_bounded(tmp_path, sends, monkeypatch):
    monkeypatch.setenv("GMS_NOTIFY_MAX_ATTEMPTS", "3")
    _late_row(tmp_path)
    for _ in range(6):
        notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 3
    assert notify.outbox_status(tmp_path)["exhausted"] == 1


def test_backoff_holds_a_failed_item_until_it_is_due(tmp_path, sends, monkeypatch):
    monkeypatch.setenv("GMS_NOTIFY_BACKOFF_MIN", "30")
    _late_row(tmp_path)
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 1  # second pass is inside the backoff window


def test_pending_state_survives_a_restart_and_delivered_is_never_resent(tmp_path, sends):
    _late_row(tmp_path)
    notify.check_missed_slots(tmp_path, _config(tmp_path))  # fails, goes pending
    assert notify.outbox_status(tmp_path)["pending"] == 1
    # "restart": nothing but the data dir carries over
    outbox = json.loads((tmp_path / "notify_outbox.json").read_text())
    assert outbox["pending"]

    sends.result = True
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert notify.outbox_status(tmp_path)["pending"] == 0
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 2


# --- A-4: dedup, and a recovered item is not alerted and its pending entry is cleared ----------


def test_a_recovered_item_clears_its_pending_entry_without_alerting(tmp_path, sends):
    delivery_id = _late_row(tmp_path)
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert notify.outbox_status(tmp_path)["pending"] == 1

    conn = open_queue_db(tmp_path)
    transition(conn, delivery_id, "queued", "failed_permanent")
    conn.close()
    sends.calls.clear()
    notify.check_missed_slots(tmp_path, _config(tmp_path))
    assert sends.calls == []
    assert notify.outbox_status(tmp_path)["pending"] == 0


# --- A-5: first scan reconciles currently actionable items once --------------------------------


def _failed_row(tmp_path, index, status="failed_permanent"):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", index, "tiktok", f"h{index}")
    if status != "queued":
        transition(conn, delivery_id, "queued", status)
    conn.close()
    return delivery_id


def test_first_scan_notifies_actionable_items_once_and_skips_recovered(tmp_path, sends):
    sends.result = True
    actionable = _failed_row(tmp_path, 1)
    _failed_row(tmp_path, 2, status="queued")  # recovered / healthy
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    assert [c[1]["delivery_id"] for c in sends.calls] == [actionable]


def test_first_scan_with_no_destination_does_not_baseline_the_backlog_away(tmp_path, sends):
    sends.result = notify.UNAVAILABLE
    _failed_row(tmp_path, 1)
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    sends.result = True
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    assert len(sends.calls) == 2  # delivered once a destination exists


def test_the_backlog_is_sent_in_bounded_batches(tmp_path, sends):
    sends.result = True
    for index in range(notify.RECONCILE_BATCH + 2):
        _failed_row(tmp_path, index)
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    assert len(sends.calls) == notify.RECONCILE_BATCH
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    assert len(sends.calls) == notify.RECONCILE_BATCH + 2


def test_exhausted_head_rows_do_not_starve_the_tail(tmp_path, sends, monkeypatch):
    """Rows out of attempts consume no batch slot: rows past the cap are still attempted."""
    monkeypatch.setenv("GMS_NOTIFY_MAX_ATTEMPTS", "1")
    sends.result = False  # every push fails, so each attempted row is exhausted after one try
    total = notify.RECONCILE_BATCH + 2
    for index in range(total):
        _failed_row(tmp_path, index)
    for _ in range(4):
        notify.check_queue_regressions(tmp_path, _config(tmp_path))
    attempted = {call[1]["delivery_id"] for call in sends.calls}
    assert len(attempted) == total


# --- A-6: /healthz reports the effective configuration -----------------------------------------


def _write_url_file(text):
    from gomoveshift.paths import gms_config_home

    path = gms_config_home() / notify.NOTIFY_URL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _health():
    resp = TestClient(create_app()).get("/healthz")
    return resp, resp.json()


def test_healthz_sees_the_notify_url_file_not_only_the_env():
    _write_url_file(_NTFY_URL + "\n")
    resp, body = _health()
    assert body["notify_configured"] is True
    assert body["notify_source"] == "file"
    assert _TOPIC not in resp.text


def test_healthz_reports_env_source_without_the_topic(monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    resp, body = _health()
    assert body["notify_source"] == "env"
    assert _TOPIC not in resp.text


def test_healthz_unconfigured_for_no_destination_and_for_a_bad_topic(monkeypatch):
    _, body = _health()
    assert body["notify_configured"] is False
    assert body["notify_source"] is None
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", "https://ntfy.sh/bad topic!")
    _, body = _health()
    assert body["notify_configured"] is False


def test_healthz_reports_delivery_and_observer_failure_counts(tmp_path, monkeypatch, sends):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    _late_row(tmp_path)
    notify.check_missed_slots(tmp_path, _config(tmp_path))  # one failed delivery, pending
    _, body = _health()
    assert body["notify_pending"] == 1
    assert body["notify_delivery_failures"] == 1
    assert body["observer_failures"] is None  # ticker disabled
