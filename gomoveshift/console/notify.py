"""Console failure webhooks: failures were discovered, never pushed.

``notify()`` POSTs a small JSON payload to ``GMS_CONSOLE_NOTIFY_URL`` -- else the first line of
``notify_url`` under the config home (``gomoveshift.paths.gms_config_home``) -- and is a silent no-op when neither names a URL. A notify
failure is logged, never raised: a broken webhook must never fail a job or a queue scan.

``notify()`` answers ``True`` (delivered), ``False`` (attempted, did not go out) or ``UNAVAILABLE``
(no usable destination: nothing was attempted). The observers record an alert as seen only on
delivery; everything else stays pending in ``notify_outbox.json`` and is retried with a bounded,
backed-off schedule that survives a restart (``_deliver``).

A URL whose host is ``NTFY_HOST`` gets plain-words phone pushes through ntfy.sh (times in
Eastern) instead; any other URL gets the JSON webhook unchanged. The URL and its topic are a
password -- never logged or printed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from ..paths import gms_config_home
from .pubqueue import TERMINAL_STATUSES, paused_platforms, queue_rows
from .settings import console_data_dir

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 5.0
_SEEN_STATE_FILENAME = "notify_seen_queue.json"
_GATE_SEEN_STATE_FILENAME = "notify_seen_gates.json"
_MISSED_SEEN_STATE_FILENAME = "notify_seen_missed_slots.json"

#: Phone-push host. The allowlist scanner reads this constant.
NTFY_HOST = "ntfy.sh"
_NTFY_ENDPOINT = "https://ntfy.sh/"
#: File name, under the config home, where ``tools/phone_alerts_setup.py`` writes the topic URL;
#: env wins over it.
NOTIFY_URL_FILE = "notify_url"
_NTFY_TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_EASTERN = ZoneInfo("America/New_York")
_TIKTOK_URL = "https://www.tiktok.com/"
_DETAIL_MAX = 300
_CAPTION_MAX = 3000

#: Key an observer tucks the delivery row under so the phone message can name the short and
#: read its caption. ``notify()`` removes it, so the generic webhook payload never carries it.
_ROW_KEY = "_row"

#: Minutes a pending review-gate question may wait before ``gate_stale``
#: fires (once). ``GMS_GATE_STALE_MIN`` overrides; see
#: ``console/registry/gms_settings.yaml``.
DEFAULT_GATE_STALE_MIN = 30

#: Delivery statuses worth pushing a notification for. `awaiting_manual_publish`
#: is not itself a failure, but it needs a human before it can move on -- same
#: as a failed job, it goes unnoticed unless someone happens to open the tab.
_WATCHED_STATUSES = ("failed_permanent", "awaiting_manual_publish")

_SECRET_KEY_RE = re.compile(
    r"""(?i)((?:access_token|refresh_token|id_token|client_secret|api_key|token)["']?\s*[=:]\s*["']?)"""
    r"""[^\s"'&,}]+"""
)
_SCHEME_TOKEN_RE = re.compile(r"(?i)\b(Bearer|OAuth)\s+\S+")


def _redact(text: str) -> str:
    """Token-looking text replaced before anything is truncated or sent."""
    text = _SECRET_KEY_RE.sub(r"\1[redacted]", text)
    return _SCHEME_TOKEN_RE.sub(r"\1 [redacted]", text)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


class _Unavailable:
    """``notify()``'s answer when no usable destination exists: falsy, and not ``False``."""

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "UNAVAILABLE"


UNAVAILABLE = _Unavailable()

#: Max individually-notified backlog items per observer pass: a first scan over a long backlog
#: drains over the next ticks instead of flooding the phone in one burst.
RECONCILE_BATCH = 10
_OUTBOX_FILENAME = "notify_outbox.json"
_DEFAULT_MAX_ATTEMPTS = 6
_DEFAULT_BACKOFF_MIN = 2.0
_BACKOFF_CAP_MIN = 240.0


def _resolve_destination() -> Tuple[str, Optional[str]]:
    """``(url, source)``: ``GMS_CONSOLE_NOTIFY_URL`` (``"env"``), else the first line of the
    notify-url file (``"file"``); ``("", None)`` if neither. The one resolver ``notify()`` and
    ``/healthz`` share."""
    env = os.environ.get("GMS_CONSOLE_NOTIFY_URL", "").strip()
    if env:
        return env, "env"
    try:
        with open(gms_config_home() / NOTIFY_URL_FILE, encoding="utf-8") as handle:
            url = handle.readline().strip()
    except OSError:
        return "", None
    return (url, "file") if url else ("", None)


def _notify_url() -> str:
    return _resolve_destination()[0]


def _ntfy_topic(url: str) -> Optional[str]:
    """The topic of an ntfy URL, or ``None`` when ``url`` is not ntfy or its topic is invalid."""
    parts = urlsplit(url)
    if (parts.hostname or "").lower() != NTFY_HOST:
        return None
    topic = parts.path.strip("/").split("/")[0]
    return topic if _NTFY_TOPIC_RE.match(topic) else None


def _destination_usable(url: str) -> bool:
    if not url:
        return False
    if (urlsplit(url).hostname or "").lower() == NTFY_HOST:
        return _ntfy_topic(url) is not None
    return True


def destination_status() -> Dict[str, Any]:
    """Effective configuration for ``/healthz``: never the URL or topic."""
    url, source = _resolve_destination()
    usable = _destination_usable(url)
    return {"configured": usable, "source": source if usable else None}


def _outbox_path(data_dir: Optional[Path]) -> Path:
    base = data_dir if data_dir is not None else console_data_dir()
    return Path(base) / _OUTBOX_FILENAME


def _load_outbox(path: Path) -> Dict[str, Any]:
    box: Dict[str, Any] = {}
    try:
        loaded = json.loads(path.read_text())
        box = loaded if isinstance(loaded, dict) else {}
    except (json.JSONDecodeError, OSError):
        box = {}
    box.setdefault("pending", {})
    box.setdefault("stats", {})
    return box


def _save_outbox(path: Path, box: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(box))
    tmp.replace(path)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _max_attempts() -> int:
    return int(_env_float("GMS_NOTIFY_MAX_ATTEMPTS", _DEFAULT_MAX_ATTEMPTS))


def outbox_status(data_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Counts for ``/healthz``: pending alerts, exhausted ones, delivery failures so far."""
    box = _load_outbox(_outbox_path(data_dir))
    limit = _max_attempts()
    pending = box["pending"]
    stats = box["stats"]
    return {
        "pending": len(pending),
        "exhausted": sum(1 for e in pending.values() if e.get("attempts", 0) >= limit),
        "delivery_failures": int(stats.get("delivery_failures", 0)),
        "last_failure_at": stats.get("last_failure_at"),
        "last_success_at": stats.get("last_success_at"),
    }


def _deliver(data_dir: Optional[Path], bucket: str, key: str, send) -> bool:
    """Run ``send()`` for one alert unless it is backed off or out of attempts; ``True`` only when
    it was delivered. An undelivered alert stays pending in the outbox (durable), a failed attempt
    backs the next one off exponentially, and attempts are capped, so a dead destination is
    neither hammered nor forgotten silently (``/healthz`` counts it)."""
    path = _outbox_path(data_dir)
    box = _load_outbox(path)
    ident = f"{bucket}:{key}"
    entry = box["pending"].get(ident)
    now = datetime.now(timezone.utc)
    if entry:
        if entry.get("attempts", 0) >= _max_attempts():
            return False
        due = _scheduled_moment(entry.get("next_at"))
        if due is not None and due > now:
            return False
    result = send()
    stats = box["stats"]
    if result is UNAVAILABLE:
        box["pending"].setdefault(ident, {"attempts": 0, "next_at": now.isoformat()})
        _save_outbox(path, box)
        return False
    if result is False:
        attempts = (entry or {}).get("attempts", 0) + 1
        delay = min(
            _env_float("GMS_NOTIFY_BACKOFF_MIN", _DEFAULT_BACKOFF_MIN) * 2 ** (attempts - 1),
            _BACKOFF_CAP_MIN,
        )
        box["pending"][ident] = {
            "attempts": attempts,
            "next_at": (now + timedelta(minutes=delay)).isoformat(),
            "last_failure_at": now.isoformat(),
        }
        stats["delivery_failures"] = int(stats.get("delivery_failures", 0)) + 1
        stats["last_failure_at"] = now.isoformat()
        _save_outbox(path, box)
        return False
    box["pending"].pop(ident, None)
    stats["last_success_at"] = now.isoformat()
    _save_outbox(path, box)
    return True


def _clear_pending(data_dir: Optional[Path], bucket: str, live_keys) -> None:
    """Drop pending entries of ``bucket`` whose item is no longer actionable (recovered)."""
    path = _outbox_path(data_dir)
    box = _load_outbox(path)
    prefix = f"{bucket}:"
    live = {f"{prefix}{key}" for key in live_keys}
    stale = [i for i in box["pending"] if i.startswith(prefix) and i not in live]
    if not stale:
        return
    for ident in stale:
        del box["pending"][ident]
    _save_outbox(path, box)


def _eastern(value: Any) -> str:
    """A stored UTC timestamp as ``Sat Oct 3, 2:00 PM EDT``; ``""`` when absent or unreadable."""
    moment = _scheduled_moment(value)
    if moment is None:
        return ""
    local = moment.astimezone(_EASTERN)
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{local.strftime('%a %b')} {local.day}, {clock} {local.strftime('%Z')}"


def _short_label(row: Dict[str, Any]) -> str:
    project, index = row.get("project_name"), row.get("short_index")
    if project is None and index is None:
        return ""
    return f"{project or 'project'} short #{index}" if index is not None else f"{project}"


def _tiktok_caption(row: Dict[str, Any]) -> str:
    """The short's ``payloads.tiktok`` caption (else its title), read straight from the row's
    plan sidecar -- never through ``publishing/**``. ``""`` when it cannot be read."""
    path = row.get("sidecar_path")
    if not path:
        return ""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for short in data.get("shorts", []):
            if short.get("short_index") != row.get("short_index"):
                continue
            if short.get("project_name") not in (None, row.get("project_name")):
                continue
            tiktok = (short.get("payloads") or {}).get("tiktok") or {}
            return str(tiktok.get("caption") or tiktok.get("title") or "")
    except Exception:  # noqa: BLE001 -- a push without the caption beats no push
        return ""
    return ""


def _tiktok_pack_mode() -> bool:
    """True when TikTok shorts go to the Studio pack (``$GMS_TIKTOK_MODE``);
    an unreadable mode is not pack, so the push keeps the inbox wording."""
    from ..publishing.handlers import resolve_tiktok_mode

    try:
        return resolve_tiktok_mode() == "pack"
    except ValueError:
        return False


def _queue_message(
    event: str, identity: Dict[str, Any], status: str, row: Dict[str, Any]
) -> Tuple[str, str, Dict[str, Any]]:
    platform = str(identity.get("platform") or row.get("platform") or "a platform")
    nice = {"tiktok": "TikTok"}.get(platform.lower(), platform.capitalize())
    label = _short_label(row)
    slot = _eastern(row.get("scheduled_at") or identity.get("scheduled_at"))
    if event == "slot_missed":
        what = f"{nice} {label}".strip()
        when = f"was scheduled for {slot} and has not gone out" if slot else "has not gone out"
        return f"{nice}: missed a posting slot", f"{what} {when}.", {"priority": 3}
    if status == "awaiting_manual_publish":
        head = f"Slot: {slot}\n" if slot else ""
        caption = _clip(_tiktok_caption(row), _CAPTION_MAX)
        if _tiktok_pack_mode():
            from ..publishing.tiktok_pack import resolve_pack_dir
            from .pubqueue import _windows_path

            title = "TikTok: short ready to schedule"
            steps = (
                f"Schedule {_windows_path(resolve_pack_dir())} in TikTok Studio "
                "(caption in captions.txt)"
            )
        else:
            title = "TikTok: post today's short"
            steps = (
                "Open TikTok -> Inbox -> tap the upload notification -> paste this caption -> Post"
            )
        body = f"{head}{steps}" + (f"\n\n{caption}" if caption else "")
        return (title, body, {"click": _TIKTOK_URL, "priority": 4, "tags": ["iphone"]})
    if status == "failed_permanent":
        detail = _clip(_redact(str(row.get("error_detail") or "no error detail")), _DETAIL_MAX)
        body = f"{nice} {label}".strip() + f"\n{detail}"
        return f"{nice}: post failed", body, {"priority": 4, "tags": ["warning"]}
    return f"{nice}: needs a look", f"{label} is now {status}".strip(), {}


def _readable(event: str, identity: Dict[str, Any], status: str) -> Tuple[str, str, Dict[str, Any]]:
    """``(title, body, extras)`` for a phone push: plain words, never the raw event name."""
    row = identity.get(_ROW_KEY) or {}
    if event in ("queue_regression", "slot_missed"):
        return _queue_message(event, identity, status, row)
    if event == "job_failed":
        body = f"{identity.get('job_name') or 'A job'} exited with code {identity.get('rc')}."
        return "GoMoveShift job failed", body, {"priority": 4}
    if event == "job_interrupted":
        body = f"{identity.get('job_name') or 'A job'} was cut off by a console restart."
        return "GoMoveShift job interrupted", body, {"priority": 3}
    if event in ("gate_pending", "gate_stale"):
        title = "Review question " + ("waiting" if event == "gate_pending" else "still waiting")
        return (
            title,
            _clip(str(identity.get("question") or "A review gate needs an answer."), 500),
            {},
        )
    if event == "console_error":
        detail = _clip(_redact(str(status)), _DETAIL_MAX)
        return "GoMoveShift console error", f"{identity.get('exc_type') or 'Error'}: {detail}", {}
    return "GoMoveShift: " + event.replace("_", " "), _clip(_redact(str(status)), _DETAIL_MAX), {}


def _post(url: str, payload: Dict[str, Any], event: str) -> bool:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # The exception text can quote the URL (the topic is a password): log only its type.
        logger.warning("console notify failed for event %r: %s", event, type(exc).__name__)
        return False
    return True


def notify(event: str, identity: Dict[str, Any], status: str) -> Any:
    """POST ``{event, **identity, status, timestamp, console_url}`` to the notify URL, or a
    plain-words push to ntfy.sh when the URL's host is ``NTFY_HOST``. ``UNAVAILABLE`` when no URL
    is set or an ntfy topic is invalid (nothing attempted); ``False`` when a push was attempted
    and did not go out; ``True`` when it was accepted. Never raises. An observer records a row
    only on delivery, so anything else is retried on its next pass."""
    try:
        url = _notify_url()
        if not url:
            return UNAVAILABLE
        identity = dict(identity)
        message_identity = dict(identity)
        identity.pop(_ROW_KEY, None)
        if (urlsplit(url).hostname or "").lower() == NTFY_HOST:
            segments = urlsplit(url).path.strip("/").split("/")
            topic = segments[0]
            if not _NTFY_TOPIC_RE.match(topic):
                logger.warning("console notify skipped for event %r: no valid ntfy topic", event)
                return UNAVAILABLE
            title, message, extras = _readable(event, message_identity, status)
            console = os.environ.get("GMS_CONSOLE_URL", "")
            if console and "click" not in extras:
                extras = {**extras, "click": console}
            return _post(
                _NTFY_ENDPOINT,
                {"topic": topic, "title": title, "message": message, **extras},
                event,
            )
        payload = {
            **identity,
            "event": event,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "console_url": os.environ.get("GMS_CONSOLE_URL", ""),
        }
        return _post(url, payload, event)
    except Exception as exc:  # noqa: BLE001 -- a notifier must never fail what it observes
        logger.warning("console notify failed for event %r: %s", event, type(exc).__name__)
        return False


def notify_job_failed(job: Dict[str, Any]) -> None:
    """Job identity for a nonzero-rc job, reaped by ``console/jobs.py``."""
    notify(
        "job_failed",
        {"job_id": job.get("id"), "job_name": job.get("name"), "rc": job.get("rc")},
        "failed",
    )


def notify_job_interrupted(job: Dict[str, Any]) -> None:
    """Job identity for a row ``sweep_orphaned_jobs`` found stuck ``running``
    with no live in-process handle -- a console restart mid-job, otherwise
    discovered only by someone opening the tab."""
    notify(
        "job_interrupted",
        {"job_id": job.get("id"), "job_name": job.get("name"), "started_at": job.get("created_at")},
        "interrupted",
    )


def _seen_state_path(data_dir: Optional[Path]) -> Path:
    base = data_dir if data_dir is not None else console_data_dir()
    return Path(base) / _SEEN_STATE_FILENAME


def _load_seen(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


#: Minutes a delivery may sit past its ``scheduled_at`` before the console
#: notifies ``slot_missed`` (once). ``GMS_SLOT_MISSED_MIN`` overrides; see
#: ``console/registry/gms_settings.yaml``. A drain runs on an interval, so a
#: row a minute late is waiting its turn, not missed -- same reasoning, and the
#: same default, as ``DEFAULT_GATE_STALE_MIN`` above.
DEFAULT_SLOT_MISSED_MIN = 30


def _missed_seen_state_path(data_dir: Optional[Path]) -> Path:
    base = data_dir if data_dir is not None else console_data_dir()
    return Path(base) / _MISSED_SEEN_STATE_FILENAME


def _scheduled_moment(value: Any) -> Optional[datetime]:
    """``scheduled_at`` as an aware UTC datetime, or ``None`` if it cannot be
    read. A naive stored value is treated as UTC, which is what the column
    documents. An unparseable one is skipped rather than guessed at: a
    malformed timestamp is not evidence that a slot was missed."""
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def check_missed_slots(data_dir: Optional[Path] = None, config: Optional[Any] = None) -> None:
    """Read-only scan (via ``pubqueue.queue_rows``) for a delivery whose slot
    went by while it was still waiting -- a drain that stopped, a ticker never
    started, a platform paused and forgotten. Every other observer here fires
    on something that HAPPENED; this one fires on the absence of it, which is
    why nothing caught it before.

    A row is missed when its ``scheduled_at`` is more than
    ``GMS_SLOT_MISSED_MIN`` minutes past, its status is not terminal, and its
    platform is not paused. ``slot_missed`` is notified once per delivery, with
    seen state in ``<data_dir>/notify_seen_missed_slots.json`` -- its own file,
    not ``notify_seen_queue.json``: "this row regressed" and "this row is late"
    are independent facts about the same delivery, and sharing the file would
    let either one clear the other's memory.

    A paused platform's rows are skipped WITHOUT being recorded. The pause
    suppresses the notification; it does not consume it. So a resume surfaces
    whatever is genuinely late now, which is exactly what the operator who
    resumed needs to be told -- and matches the queue layer's own refusal to
    backfill or throttle a resumed platform."""
    path = _missed_seen_state_path(data_dir)
    seen = _load_seen(path)
    changed = False
    live: set = set()
    grace = float(os.environ.get("GMS_SLOT_MISSED_MIN", DEFAULT_SLOT_MISSED_MIN))
    deadline = datetime.now(timezone.utc) - timedelta(minutes=grace)
    paused = set(paused_platforms(config))
    for row in queue_rows(config):
        if row.get("status") in TERMINAL_STATUSES:
            continue
        moment = _scheduled_moment(row.get("scheduled_at"))
        if moment is None or moment > deadline:
            continue
        if row.get("platform") in paused:
            continue
        key = str(row.get("id"))
        live.add(key)
        if seen.get(key):
            continue
        identity = {
            "delivery_id": row.get("id"),
            "platform": row.get("platform"),
            "scheduled_at": row.get("scheduled_at"),
            _ROW_KEY: _row_facts(row),
        }
        # Recorded only once delivered: an undelivered slot stays pending and is retried.
        if not _deliver(data_dir, "missed", key, lambda: notify("slot_missed", identity, "missed")):
            continue
        seen[key] = str(row.get("scheduled_at"))
        changed = True
    _clear_pending(data_dir, "missed", live)
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(seen))


def _row_facts(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "platform",
        "project_name",
        "short_index",
        "scheduled_at",
        "error_detail",
        "sidecar_path",
    )
    return {key: row.get(key) for key in keys}


def check_queue_regressions(data_dir: Optional[Path] = None, config: Optional[Any] = None) -> None:
    """Read-only scan of the delivery queue (via ``pubqueue.queue_rows``, never
    ``publishing/**`` directly) for rows newly in a watched status. Tracks
    last-seen (id -> status) in ``<data_dir>/notify_seen_queue.json`` so a
    repeat sighting of the same status never re-notifies.

    There is no silent baseline: rows already in a watched status when the scan first runs are
    actionable, so they are notified once -- at most ``RECONCILE_BATCH`` per pass, the rest on
    the following passes -- and never again; recovered rows were never watched, so they stay
    quiet. A row is recorded only once its push was delivered (no destination, a failed push or
    a backed-off retry leave it pending), and forgotten when it leaves a watched status."""
    path = _seen_state_path(data_dir)
    seen = _load_seen(path)
    changed = False
    live: set = set()
    sent = 0
    for row in queue_rows(config):
        status = row.get("status")
        key = str(row.get("id"))
        if status not in _WATCHED_STATUSES:
            # Left the watched status (a requeue): forget it, so a second failure alerts again.
            if seen.pop(key, None) is not None:
                changed = True
            continue
        live.add(f"{key}#{status}")
        if seen.get(key) == status:
            continue
        if sent >= RECONCILE_BATCH:
            continue  # the rest of a backlog drains on the next pass
        identity = {"delivery_id": row.get("id"), "platform": row.get("platform")}
        message = {**identity, _ROW_KEY: _row_facts(row)}
        attempted = []

        def _send(message=message, status=status, attempted=attempted):
            attempted.append(True)
            return notify("queue_regression", message, status)

        delivered = _deliver(data_dir, "queue", f"{key}#{status}", _send)
        if attempted:
            sent += 1  # only a real push spends the batch; exhausted/backed-off rows cost nothing
        if not delivered:
            continue  # not delivered: stay unrecorded, the next pass retries
        seen[key] = status
        changed = True
    _clear_pending(data_dir, "queue", live)
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(seen))


def check_pending_gates(output_dirs, data_dir: Optional[Path] = None) -> None:
    """Read-only scan via ``drives.list_pending_gates`` (never re-implements
    gate discovery). Notifies ``gate_pending`` once per question file and
    ``gate_stale`` once a question has waited longer than
    ``GMS_GATE_STALE_MIN`` minutes. Per-gate ``pending``/``stale`` seen state
    lives in ``<data_dir>/notify_seen_gates.json``, keyed by
    ``output_dir``/``gate_id`` -- same pattern as
    :func:`check_queue_regressions`.
    """
    from .drives import list_pending_gates  # noqa: PLC0415 -- avoids an import cycle

    path = _gate_seen_state_path(data_dir)
    seen = _load_seen(path)
    changed = False
    live: set = set()
    stale_min = float(os.environ.get("GMS_GATE_STALE_MIN", DEFAULT_GATE_STALE_MIN))
    now = datetime.now(timezone.utc).timestamp()
    for row in list_pending_gates(output_dirs):
        key = f"{row['output_dir']}/{row['gate_id']}"
        live.update({f"{key}#pending", f"{key}#stale"})
        state = seen.get(key) or {}
        identity = {
            "gate_id": row["gate_id"],
            "output_dir": row["output_dir"],
            "question": row["question"],
        }
        if not state.get("pending") and _deliver(
            data_dir, "gate", f"{key}#pending", lambda: notify("gate_pending", identity, "pending")
        ):
            state["pending"] = True
            changed = True
        age_min = (now - row["mtime"]) / 60.0
        if (
            age_min >= stale_min
            and not state.get("stale")
            and _deliver(
                data_dir, "gate", f"{key}#stale", lambda: notify("gate_stale", identity, "stale")
            )
        ):
            state["stale"] = True
            changed = True
        if state:
            seen[key] = state
    _clear_pending(data_dir, "gate", live)
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(seen))


def _gate_seen_state_path(data_dir: Optional[Path]) -> Path:
    base = data_dir if data_dir is not None else console_data_dir()
    return Path(base) / _GATE_SEEN_STATE_FILENAME
