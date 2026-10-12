"""The console compose services as ``docker compose config`` renders them.

``gomoveshift-console`` inherits ``console-base`` with a same-file compose
``extends``, so its plain YAML holds only what it adds. ``resolved_service``
applies the ``extends`` merge the way compose does: volumes by container target,
environment by key, every other key overridden by the child.
``test_console_compose.test_service_helper_matches_the_rendered_config`` pins
this against the real render wherever docker runs.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

COMPOSE_PATH = Path(__file__).resolve().parent.parent / "docker" / "console-compose.yaml"


def volume_target(entry: str) -> str:
    """Container side of a ``host:target[:mode]`` volume string, defaults expanded."""
    fields = re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", entry).split(":")
    return (fields[:-1] if fields[-1] in ("ro", "rw") else fields)[-1]


def env_key(entry: str) -> str:
    return entry.split("=", 1)[0]


def _merge_by(base: list[str], own: list[str], key) -> list[str]:
    merged = {key(e): e for e in base}
    merged.update({key(e): e for e in own})
    return list(merged.values())


def resolved_service(name: str = "gomoveshift-console", compose: dict | None = None) -> dict:
    services = (compose or yaml.safe_load(COMPOSE_PATH.read_text()))["services"]
    own = services[name]
    if "extends" not in own:
        return own
    base = services[own["extends"]["service"]]
    merged = {**base, **{k: v for k, v in own.items() if k != "extends"}}
    merged["volumes"] = _merge_by(base["volumes"], own.get("volumes", []), volume_target)
    merged["environment"] = _merge_by(base["environment"], own.get("environment", []), env_key)
    return merged
