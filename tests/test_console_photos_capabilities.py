"""The Photos screen names the capabilities this build actually has, and
leads with the creator journey.

The page used to assert ``Selector, reel and astro are not built yet`` while
``pipeline/photos/`` shipped ``selector.py``, ``reel.py`` and twenty-odd astro
modules, every one of them reachable from ``gomoveshift photos ...``. The
status is now DERIVED from the installed modules rather than written as prose,
so the page cannot go stale against the build again.
"""

from fastapi.testclient import TestClient

from gomoveshift.console import photos
from gomoveshift.console.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _row(name: str) -> dict:
    return next(r for r in photos.capability_status() if r["name"] == name)


def test_capability_status_reports_the_engines_that_exist():
    # selector.py / reel.py / astro_orchestrator.py all import in this build --
    # nothing may report them as missing.
    for name in ("Selector", "Reel", "Astro"):
        assert _row(name)["installed"] is True, name
    for name in ("Import photos", "Develop"):
        assert _row(name)["installed"] is True, name


def test_capability_status_separates_console_driven_from_command_line_only():
    assert _row("Import photos")["console"] is True
    assert _row("Develop")["console"] is True
    # The console does not drive these yet -- that is a console gap, not a
    # missing engine, and each row carries the command that does run it.
    for name in ("Selector", "Reel", "Astro"):
        row = _row(name)
        assert row["console"] is False, name
        assert row["command"].startswith("gomoveshift photos "), name


def test_photos_tab_never_claims_a_shipped_engine_is_unbuilt():
    body = _client().get("/photos").text
    assert "are not built yet" not in body
    for command in ("gomoveshift photos reel", "gomoveshift photos astro"):
        assert command in body


def test_photos_tab_leads_with_the_creator_journey_stages():
    body = _client().get("/photos").text
    for stage in ("Import", "Suggested treatment", "Review", "Schedule", "Results"):
        assert stage in body


def test_photos_tab_keeps_tuning_available_behind_an_advanced_view():
    body = _client().get("/photos").text
    assert "<details" in body and "Advanced" in body
    # Available, not removed: every dial the form used to offer is still there.
    for dial in ("vignette", "clarity", "grain", "halation", "white_balance_g"):
        assert f'name="{dial}"' in body


def test_photos_tab_carries_the_selected_journey_across_navigation():
    body = _client().get("/photos?journey=example-2026").text
    # Prefilled into the form the operator will submit ...
    assert 'value="example-2026"' in body
    # ... and threaded onto the journey-stage links so the selection survives
    # a hop to the review/schedule screens and back.
    assert "journey=example-2026" in body
