"""Metricool is retired -- every publishing route is ``first_party`` or
``off``. These tests pin the ``off`` semantics at each of the three
consumers, plus a structural grep that keeps ``"metricool"`` out of the
publishing/config/paths surface except one documented legacy-key comment.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.cli.handlers.dispatch_queue import _routed
from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline._phase_d_outputs import _has_first_party_route

_SERVICE_ROOT = Path(__file__).parent.parent


def _drive_config(**routes):
    all_off = {
        "publishing_route_youtube": "off",
        "publishing_route_instagram": "off",
        "publishing_route_facebook": "off",
        "publishing_route_tiktok": "off",
    }
    all_off.update(routes)
    return DriveConfig(**all_off)


class TestRoutedSkipsOff:
    def test_routed_true_for_first_party(self):
        config = SimpleNamespace(drive=_drive_config(publishing_route_youtube="first_party"))
        assert _routed(config, "youtube") is True

    def test_routed_false_for_off(self):
        config = SimpleNamespace(drive=_drive_config())
        assert _routed(config, "youtube") is False


class TestHasFirstPartyRoute:
    def test_false_when_all_four_routes_are_off(self):
        config = SimpleNamespace(drive=_drive_config())
        assert _has_first_party_route(config) is False

    def test_true_when_any_route_is_first_party(self):
        config = SimpleNamespace(drive=_drive_config(publishing_route_tiktok="first_party"))
        assert _has_first_party_route(config) is True


class TestConfigDefaults:
    def test_default_config_routes_every_platform_off(self):
        # A config that names no route publishes nowhere: Phase D's
        # publish-plan sidecar is entered only by an explicit first_party.
        cfg = Config().drive
        assert cfg.publishing_route_youtube == "off"
        assert cfg.publishing_route_instagram == "off"
        assert cfg.publishing_route_facebook == "off"
        assert cfg.publishing_route_tiktok == "off"


class TestNoMetricoolRouteValueLeftInCode:
    """Structural pin: ``"metricool"``/``'metricool'`` may no longer appear
    as a quoted string LITERAL (a value a route could hold) anywhere in the
    publishing/config/paths surface -- only prose mentions of the retired
    Metricool CSV product survive, and those are not checked here.
    A quoted literal reappearing here is a regression back toward the
    retired route vocabulary.
    """

    _LITERAL = re.compile(r"""["']metricool["']""", re.IGNORECASE)
    # The ONE place the retired value may be spelled: the constant the route
    # validator uses to recognise an un-migrated config.yaml and coerce it to
    # ``off``. Anything else naming the value is a regression.
    _ALLOWLIST = {"gomoveshift/config.py": "RETIRED_PUBLISHING_ROUTES"}

    def test_retired_route_constant_is_the_only_allowlisted_line(self):
        from gomoveshift.config import RETIRED_PUBLISHING_ROUTES

        assert RETIRED_PUBLISHING_ROUTES == frozenset({"metricool"})

    def test_no_quoted_metricool_literal_remains(self):
        targets = [
            _SERVICE_ROOT / "gomoveshift" / "publishing",
            _SERVICE_ROOT / "gomoveshift" / "config.py",
            _SERVICE_ROOT / "gomoveshift" / "paths.py",
        ]
        offenders = []
        for target in targets:
            files = target.rglob("*.py") if target.is_dir() else [target]
            for f in files:
                rel = str(f.relative_to(_SERVICE_ROOT))
                text = f.read_text(encoding="utf-8")
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if self._LITERAL.search(line) and self._ALLOWLIST.get(rel, "\0") not in line:
                        offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert offenders == [], "\n".join(offenders)
