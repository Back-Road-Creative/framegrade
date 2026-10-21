"""Console failure webhooks (console/notify.py) -- failures were discovered, never
pushed. ``GMS_CONSOLE_NOTIFY_URL`` unset is a silent no-op; a failed job and a queue regression
(``failed_permanent``/``awaiting_manual_publish``) each notify exactly once."""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.console import jobs, notify
from gomoveshift.console.jobs import JobStore
from gomoveshift.publishing.queue import enqueue, open_queue_db, transition


class _CapturingHandler(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _CapturingHandler.received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # silence test-run noise
        pass


@pytest.fixture
def notify_server():
    _CapturingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, _CapturingHandler.received
    server.shutdown()
    thread.join(timeout=2.0)


def test_notify_is_a_silent_noop_without_the_env_var(monkeypatch):
    monkeypatch.delenv("GMS_CONSOLE_NOTIFY_URL", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("must not make a network call when unset")

    monkeypatch.setattr(notify.urllib.request, "urlopen", _boom)
    notify.notify("job_failed", {"job_id": 1}, "failed")  # must not raise


def test_notify_posts_the_expected_payload(notify_server, monkeypatch):
    server, received = notify_server
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", f"http://127.0.0.1:{server.server_port}/hook")
    monkeypatch.setenv("GMS_CONSOLE_URL", "http://console.example/")

    notify.notify("job_failed", {"job_id": 7, "job_name": "status"}, "failed")

    assert len(received) == 1
    payload = received[0]
    assert payload["event"] == "job_failed"
    assert payload["job_id"] == 7
    assert payload["job_name"] == "status"
    assert payload["status"] == "failed"
    assert payload["console_url"] == "http://console.example/"
    assert "timestamp" in payload


def test_notify_failure_is_logged_never_raised(monkeypatch, caplog):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", "http://127.0.0.1:1/unreachable")
    with caplog.at_level("WARNING"):
        notify.notify("job_failed", {"job_id": 1}, "failed")  # must not raise
    assert "console notify failed" in caplog.text


def test_reap_notifies_exactly_once_on_job_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "notify_job_failed", lambda job: calls.append(job))
    monkeypatch.setattr(jobs, "_validate_argv", lambda argv: None)
    store = JobStore(tmp_path / "jobs.sqlite3")
    argv = [sys.executable, "-c", "import sys; sys.exit(1)"]
    job_id = jobs.launch(store, "status", argv, tmp_path / "logs")

    deadline = time.time() + 5.0
    row = jobs.get_job(store, job_id)
    while row["status"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        row = jobs.get_job(store, job_id)

    assert row["status"] == "failed"
    assert len(calls) == 1
    assert calls[0]["id"] == job_id

    jobs.get_job(store, job_id)  # a second reap must not renotify
    assert len(calls) == 1


def test_check_pending_gates_notifies_pending_once_and_stale_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        notify, "notify", lambda event, identity, status: calls.append((event, identity, status))
    )
    monkeypatch.setenv("GMS_GATE_STALE_MIN", "0")  # every question is immediately "stale"
    output_dir = tmp_path / "out"
    (output_dir / "_gates").mkdir(parents=True)
    (output_dir / "_gates" / "gate_a.question.json").write_text(
        json.dumps({"question": "Episode breakdown correct?"})
    )

    notify.check_pending_gates([str(output_dir)], data_dir=tmp_path)
    notify.check_pending_gates([str(output_dir)], data_dir=tmp_path)  # must not re-notify

    events = [c[0] for c in calls]
    assert events.count("gate_pending") == 1
    assert events.count("gate_stale") == 1
    assert calls[0][1] == {
        "gate_id": "gate_a",
        "output_dir": str(output_dir),
        "question": "Episode breakdown correct?",
    }


def test_check_queue_regressions_notifies_once_per_status(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        notify, "notify", lambda event, identity, status: calls.append((event, identity, status))
    )
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    notify.check_queue_regressions(tmp_path, config)  # first use only seeds

    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(conn, "proj", 0, "tiktok", "hash-a")
    transition(conn, delivery_id, "queued", "failed_permanent")
    conn.close()

    notify.check_queue_regressions(tmp_path, config)
    notify.check_queue_regressions(tmp_path, config)

    assert len(calls) == 1
    assert calls[0][0] == "queue_regression"
    assert calls[0][1]["delivery_id"] == delivery_id
    assert calls[0][2] == "failed_permanent"


# --- phone pushes through ntfy.sh --------------------------------------------------------------

_TOPIC = "gms-testtopic_0123456789"
_NTFY_URL = f"https://ntfy.sh/{_TOPIC}"


@pytest.fixture
def phone(tmp_path, monkeypatch):
    """Tmp HOME, no env URL, and urlopen replaced by a recorder -- no network."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("GMS_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("GMS_CONSOLE_NOTIFY_URL", raising=False)
    monkeypatch.delenv("GMS_CONSOLE_URL", raising=False)
    sent: list = []

    def _fake(req, timeout=None):
        sent.append(req)

    monkeypatch.setattr(notify.urllib.request, "urlopen", _fake)
    return SimpleNamespace(home=home, sent=sent)


def _write_url_file(home, text):
    path = home / ".config" / "gomoveshift" / "notify_url"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _body(req):
    return json.loads(req.data.decode("utf-8"))


def test_env_url_beats_the_file(phone, monkeypatch):
    _write_url_file(phone.home, "https://ntfy.sh/from-the-file\n")
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", "http://hook.example.test/x")

    notify.notify("job_failed", {"job_id": 1, "job_name": "n", "rc": 2}, "failed")

    assert [r.full_url for r in phone.sent] == ["http://hook.example.test/x"]


def test_file_is_the_fallback_and_only_its_first_line_counts(phone):
    _write_url_file(phone.home, f"{_NTFY_URL}\nhttps://ignored.example.test/\n")

    notify.notify("job_failed", {"job_id": 1, "job_name": "n", "rc": 2}, "failed")

    assert [r.full_url for r in phone.sent] == ["https://ntfy.sh/"]
    assert _body(phone.sent[0])["topic"] == _TOPIC


def test_config_home_override_moves_the_file(phone, monkeypatch, tmp_path):
    """The default file lives under ``~/.config/gomoveshift``; ``GMS_CONFIG_HOME``
    moves it, and the default location is then ignored."""
    _write_url_file(phone.home, "https://ntfy.sh/from-the-default\n")
    override = tmp_path / "elsewhere"
    override.mkdir()
    (override / "notify_url").write_text(f"{_NTFY_URL}\n")
    monkeypatch.setenv("GMS_CONFIG_HOME", str(override))

    notify.notify("job_failed", {"job_id": 1, "job_name": "n", "rc": 2}, "failed")

    assert [r.full_url for r in phone.sent] == ["https://ntfy.sh/"]
    assert _body(phone.sent[0])["topic"] == _TOPIC


@pytest.mark.parametrize("content", [None, "", "\n", "   \n"])
def test_missing_or_empty_file_is_a_noop(phone, content):
    if content is not None:
        _write_url_file(phone.home, content)

    notify.notify("job_failed", {"job_id": 1}, "failed")

    assert phone.sent == []


def test_ntfy_mode_posts_json_to_the_root_with_a_readable_message(phone, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)

    notify.notify("job_failed", {"job_id": 7, "job_name": "dispatch-queue", "rc": 3}, "failed")

    (req,) = phone.sent
    assert req.full_url == "https://ntfy.sh/"
    assert req.get_method() == "POST"
    assert req.get_header("Content-type") == "application/json"
    body = _body(req)
    assert body["topic"] == _TOPIC
    assert body["title"] and body["message"]
    assert "dispatch-queue" in body["message"]
    assert "event" not in body  # a phone push, not the webhook payload


def test_non_ntfy_url_keeps_the_old_webhook_payload(phone, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", "http://hook.example.test/x")
    monkeypatch.setenv("GMS_CONSOLE_URL", "http://console.example/")

    notify.notify(
        "queue_regression",
        {"delivery_id": 4, "platform": "tiktok", "_row": {"a": 1}},
        "failed_permanent",
    )

    (req,) = phone.sent
    payload = _body(req)
    assert set(payload) == {
        "delivery_id",
        "platform",
        "event",
        "status",
        "timestamp",
        "console_url",
    }
    assert payload["event"] == "queue_regression"
    assert payload["console_url"] == "http://console.example/"


def test_ntfy_failure_is_never_raised_and_never_logs_the_topic(phone, monkeypatch, caplog):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)

    def _boom(req, timeout=None):
        raise OSError(f"cannot reach {req.full_url} {_TOPIC}")

    monkeypatch.setattr(notify.urllib.request, "urlopen", _boom)
    with caplog.at_level("DEBUG"):
        notify.notify("job_failed", {"job_id": 1}, "failed")  # must not raise

    assert "console notify failed" in caplog.text
    assert _TOPIC not in caplog.text
    assert "ntfy.sh" not in caplog.text


@pytest.mark.parametrize(
    "url", ["https://ntfy.sh/", "https://ntfy.sh", "https://ntfy.sh/bad topic!"]
)
def test_ntfy_url_without_a_valid_topic_is_a_noop(phone, monkeypatch, url):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", url)

    notify.notify("job_failed", {"job_id": 1}, "failed")

    assert phone.sent == []


def _queue_row(tmp_path, platform, *, sidecar=None, scheduled_at=None, status="queued", **fields):
    conn = open_queue_db(tmp_path)
    delivery_id = enqueue(
        conn, "proj", 3, platform, "h1", sidecar_path=sidecar, scheduled_at=scheduled_at
    )
    if status != "queued":
        transition(conn, delivery_id, "queued", status, **fields)
    conn.close()
    return delivery_id


def _config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))


def _sidecar(tmp_path, *, caption="Sunrise over the Burren #ireland", title="Burren"):
    path = tmp_path / "proj_publish_plan.json"
    tiktok = {"title": title, "caption": caption}
    other = {"short_index": 9, "payloads": {"tiktok": {"caption": "WRONG", "title": "WRONG"}}}
    path.write_text(
        json.dumps({"shorts": [other, {"short_index": 3, "payloads": {"tiktok": tiktok}}]})
    )
    return str(path)


def _seeded(tmp_path):
    notify.check_queue_regressions(tmp_path, _config(tmp_path))


def test_tiktok_inbox_push_carries_the_caption_and_the_eastern_slot(phone, tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    _seeded(tmp_path)
    _queue_row(
        tmp_path,
        "tiktok",
        sidecar=_sidecar(tmp_path),
        scheduled_at="2026-10-03T18:00:00+00:00",
        status="awaiting_manual_publish",
    )

    notify.check_queue_regressions(tmp_path, _config(tmp_path))

    (req,) = phone.sent
    body = _body(req)
    assert body["title"].startswith("TikTok")
    assert "Sunrise over the Burren #ireland" in body["message"]
    assert "WRONG" not in body["message"]
    assert "2:00 PM" in body["message"] and "EDT" in body["message"]
    assert "Inbox" in body["message"]
    assert body["click"] == "https://www.tiktok.com/"
    assert body["priority"] == 4


def test_tiktok_pack_push_points_at_the_pack_folder_not_the_inbox(phone, tmp_path, monkeypatch):
    """Pack mode leaves a file in the pack folder for TikTok Studio; nothing is
    in the app inbox, and the slot may be days away, so the push must not say
    "today" or "Inbox"."""
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    monkeypatch.setenv("GMS_TIKTOK_MODE", "pack")
    _seeded(tmp_path)
    _queue_row(
        tmp_path,
        "tiktok",
        sidecar=_sidecar(tmp_path),
        scheduled_at="2026-10-03T18:00:00+00:00",
        status="awaiting_manual_publish",
    )

    notify.check_queue_regressions(tmp_path, _config(tmp_path))

    (req,) = phone.sent
    body = _body(req)
    assert "today" not in body["title"].lower()
    assert "Inbox" not in body["message"]
    assert "TikTok Studio" in body["message"] and "captions.txt" in body["message"]
    assert "2:00 PM" in body["message"] and "EDT" in body["message"]
    assert "Sunrise over the Burren #ireland" in body["message"]


def test_tiktok_caption_falls_back_to_the_title(phone, tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    _seeded(tmp_path)
    _queue_row(
        tmp_path,
        "tiktok",
        sidecar=_sidecar(tmp_path, caption="", title="Just the title"),
        status="awaiting_manual_publish",
    )

    notify.check_queue_regressions(tmp_path, _config(tmp_path))

    assert "Just the title" in _body(phone.sent[0])["message"]


@pytest.mark.parametrize("sidecar", [None, "/nonexistent/plan.json", "NOT-JSON"])
def test_tiktok_push_still_goes_out_when_the_sidecar_cannot_be_read(
    phone, tmp_path, monkeypatch, sidecar
):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    if sidecar == "NOT-JSON":
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        sidecar = str(bad)
    _seeded(tmp_path)
    _queue_row(tmp_path, "tiktok", sidecar=sidecar, status="awaiting_manual_publish")

    notify.check_queue_regressions(tmp_path, _config(tmp_path))

    (req,) = phone.sent
    assert _body(req)["title"].startswith("TikTok")


def test_failed_permanent_push_redacts_token_looking_text_and_truncates(
    phone, tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    _seeded(tmp_path)
    detail = (
        "HTTP 401 access_token=ya29.SECRETVALUE Authorization: Bearer abc.DEF-123 "
        "OAuth tokenXYZ987 " + "x" * 600
    )
    _queue_row(tmp_path, "instagram", status="failed_permanent", error_detail=detail)

    notify.check_queue_regressions(tmp_path, _config(tmp_path))

    message = _body(phone.sent[0])["message"]
    for secret in ("ya29.SECRETVALUE", "abc.DEF-123", "tokenXYZ987"):
        assert secret not in message
    assert "[redacted]" in message
    assert "instagram" in message.lower() and "proj" in message
    assert len(message) < 600


def test_missed_slot_push_names_platform_project_short_and_eastern_time(
    phone, tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    _queue_row(tmp_path, "facebook", scheduled_at="2026-01-15T17:00:00+00:00")

    notify.check_missed_slots(data_dir=tmp_path, config=_config(tmp_path))

    message = _body(phone.sent[0])["message"]
    assert "facebook" in message.lower() and "proj" in message and "3" in message
    assert "12:00 PM" in message and "EST" in message


def test_first_run_notifies_the_actionable_backlog_once_then_new_rows_once(
    phone, tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    _queue_row(tmp_path, "tiktok", status="failed_permanent")  # pre-existing, still actionable

    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    assert len(phone.sent) == 1  # reconciled once, idempotent on the rerun
    assert (tmp_path / "notify_seen_queue.json").exists()

    conn = open_queue_db(tmp_path)
    new_id = enqueue(conn, "proj", 4, "tiktok", "h2")
    transition(conn, new_id, "queued", "awaiting_manual_publish")
    conn.close()
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    notify.check_queue_regressions(tmp_path, _config(tmp_path))

    assert len(phone.sent) == 2


def test_an_undelivered_push_is_not_recorded_so_the_next_pass_retries_it(
    phone, tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    monkeypatch.setenv("GMS_NOTIFY_BACKOFF_MIN", "0")  # retry on the very next pass
    _seeded(tmp_path)
    _queue_row(tmp_path, "tiktok", status="awaiting_manual_publish")

    def _down(req, timeout=None):
        raise notify.urllib.error.URLError("unreachable")

    monkeypatch.setattr(notify.urllib.request, "urlopen", _down)
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    seen = tmp_path / "notify_seen_queue.json"
    assert not seen.exists() or json.loads(seen.read_text()) == {}

    monkeypatch.setattr(
        notify.urllib.request, "urlopen", lambda req, timeout=None: phone.sent.append(req)
    )
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    assert len(phone.sent) == 1


def test_a_row_that_leaves_a_watched_status_alerts_again_on_its_next_failure(
    phone, tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)
    _seeded(tmp_path)
    delivery_id = _queue_row(tmp_path, "tiktok", status="failed_permanent")
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    assert len(phone.sent) == 1

    conn = open_queue_db(tmp_path)  # the operator requeues it, and it fails again
    conn.execute("UPDATE deliveries SET status='queued' WHERE id=?", (delivery_id,))
    conn.commit()
    notify.check_queue_regressions(tmp_path, _config(tmp_path))
    conn.execute("UPDATE deliveries SET status='failed_permanent' WHERE id=?", (delivery_id,))
    conn.commit()
    conn.close()
    notify.check_queue_regressions(tmp_path, _config(tmp_path))

    assert len(phone.sent) == 2


def test_every_other_event_gets_a_readable_title_and_body(phone, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_NOTIFY_URL", _NTFY_URL)

    notify.notify("job_interrupted", {"job_id": 1, "job_name": "pipeline"}, "interrupted")
    notify.notify("gate_pending", {"gate_id": "g", "question": "Breakdown correct?"}, "pending")
    notify.notify("gate_stale", {"gate_id": "g", "question": "Breakdown correct?"}, "stale")
    notify.notify("console_error", {"path": "/x", "exc_type": "KeyError"}, "boom")
    notify.notify("something_new", {}, "odd")

    assert len(phone.sent) == 5
    for req in phone.sent:
        body = _body(req)
        assert body["title"].strip() and body["message"].strip()
        assert "_" not in body["title"]


# --- dispatch-queue runs the regression observer -------------------------------------------------


def _drain_run(tmp_path, capsys, *, raises=None):
    from unittest import mock

    from gomoveshift.cli.handlers.dispatch_queue import cmd_dispatch_queue
    from gomoveshift.publishing.dispatch import DrainResult
    from test_cli_dispatch_queue import _args
    from test_cli_dispatch_queue import _config as _dq_config
    from test_cli_dispatch_queue import _record
    from test_cli_dispatch_queue import _sidecar as _dq_sidecar

    conn = open_queue_db(tmp_path)
    enqueue(conn, "proj", 3, "tiktok", "HASH1")
    conn.close()
    _dq_sidecar(tmp_path, _record())
    order: list[str] = []

    def _drain(*a, **k):
        order.append("drain")
        return DrainResult(attempted=1, succeeded=1)

    with (
        mock.patch("gomoveshift.cli.handlers.dispatch_queue._build_routed_handlers"),
        mock.patch("gomoveshift.cli.handlers.dispatch_queue.drain_queue", side_effect=_drain),
        mock.patch.object(notify, "check_queue_regressions") as reg,
        mock.patch.object(notify, "check_missed_slots"),
    ):
        reg.side_effect = raises or (lambda **k: order.append("regressions"))
        cfg = _dq_config(tmp_path, tiktok="first_party")
        rc = cmd_dispatch_queue(_args(apply=True), cfg)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    return rc, lines, order, reg, cfg


def test_dispatch_queue_runs_the_regression_observer_after_the_drain(tmp_path, capsys):
    rc, lines, order, reg, cfg = _drain_run(tmp_path, capsys)

    assert rc == 0
    assert order == ["drain", "regressions"]
    assert reg.call_args.kwargs["config"] is cfg
    assert lines[-1].startswith("attempted=1 succeeded=1")


def test_a_raising_regression_observer_never_changes_rc_or_the_last_line(tmp_path, capsys):
    rc, lines, _order, reg, _cfg = _drain_run(
        tmp_path, capsys, raises=RuntimeError("ntfy exploded")
    )

    assert rc == 0
    reg.assert_called_once()
    assert lines[-1].startswith("attempted=1 succeeded=1")
