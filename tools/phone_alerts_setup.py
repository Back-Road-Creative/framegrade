#!/usr/bin/env python3
"""Set up phone alerts: one private ntfy.sh topic the pipeline pushes to.

Stdlib only (system ``python3``, no venv, no TTY, no prompts). No args makes the topic, or says it
exists, and prints the phone steps; ``--test`` sends one test push; ``--rotate`` replaces the topic.

The topic is the only secret: it lives in ``notify_url`` under the config home (mode 0600) and is
printed only to the terminal of whoever runs this tool -- never by ``--test``, never into a log.
The config home is ``$GMS_CONFIG_HOME``, else ``$XDG_CONFIG_HOME/gomoveshift``, else
``~/.config/gomoveshift`` (the same rule as ``gomoveshift.paths.gms_config_home``, repeated here
because this tool runs on the system ``python3`` with the package not installed).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import urllib.error
import urllib.request

NTFY_ENDPOINT = "https://ntfy.sh/"
URL_PREFIX = "https://ntfy.sh/"
_TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # ntfy topic charset, as notify._NTFY_TOPIC_RE
URL_FILE = "notify_url"
TIMEOUT_SECONDS = 5.0


def _config_home() -> str:
    override = os.environ.get("GMS_CONFIG_HOME", "").strip()
    if override:
        return os.path.expanduser(override)
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        return os.path.join(os.path.expanduser(xdg), "gomoveshift")
    return os.path.join(os.path.expanduser("~"), ".config", "gomoveshift")


def _path() -> str:
    return os.path.join(_config_home(), URL_FILE)


def _read_topic() -> str | None:
    try:
        with open(_path(), encoding="utf-8") as handle:
            line = handle.readline().strip()
    except OSError:
        return None
    topic = line[len(URL_PREFIX) :].strip("/") if line.startswith(URL_PREFIX) else ""
    return topic if _TOPIC_RE.match(topic) else None


def _write_topic(topic: str, *, replace: bool) -> None:
    """Write the URL file 0600 (parent 0700). ``O_EXCL`` so a first write can never clobber."""
    path = _path()
    parent = os.path.dirname(path)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    if os.stat(parent).st_uid == os.getuid():
        os.chmod(parent, 0o700)
    target = path + ".new" if replace else path
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if replace and os.path.exists(target):
        os.unlink(target)
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{URL_PREFIX}{topic}\n")
    if replace:
        os.replace(target, path)


def _new_topic() -> str:
    return "gms-" + secrets.token_urlsafe(18)


def _steps(topic: str) -> str:
    return "\n".join(
        [
            "On your phone:",
            "  1. Install the free ntfy app (App Store / Play Store).",
            "  2. Tap + and subscribe to this topic (treat it like a password):",
            f"       {topic}",
            "  3. Run this tool again with --test and check that the push arrives:",
            "       python3 tools/phone_alerts_setup.py --test",
        ]
    )


def _send_test(topic: str) -> bool:
    body = {
        "topic": topic,
        "title": "GoMoveShift phone alerts",
        "message": "Test push. If you can read this, alerts reach your phone.",
        "tags": ["white_check_mark"],
    }
    req = urllib.request.Request(  # noqa: S310 -- fixed https endpoint
        NTFY_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS)  # noqa: S310 -- fixed https endpoint
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test", action="store_true", help="send one test push")
    group.add_argument("--rotate", action="store_true", help="replace the topic")
    args = parser.parse_args(argv)

    topic = _read_topic()
    if topic is None and os.path.exists(_path()) and not args.rotate:
        print(
            f"{_path()} exists but is not an ntfy.sh URL; left unchanged. Use --rotate to replace it."
        )
        return 1
    if args.test:
        if topic is None:
            print(
                "No phone-alert topic yet. Run this tool with no arguments first.", file=sys.stderr
            )
            return 1
        if _send_test(topic):
            print("Test push sent: OK. Check your phone.")
            return 0
        print("Test push failed (could not reach the push service).", file=sys.stderr)
        return 1
    if args.rotate:
        topic = _new_topic()
        _write_topic(topic, replace=True)
        print("Topic replaced. The old one no longer receives anything.\n")
        print(_steps(topic))
        return 0
    if topic is not None:
        print(f"A phone-alert topic already exists in {_path()}; left unchanged.")
        print("Use --rotate to replace it.\n")
        print(_steps(topic))
        return 0
    topic = _new_topic()
    _write_topic(topic, replace=False)
    print(f"Created {_path()} (mode 0600).\n")
    print(_steps(topic))
    return 0


if __name__ == "__main__":
    sys.exit(main())
