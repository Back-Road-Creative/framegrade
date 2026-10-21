"""The journey an operator picks on one console screen follows them to
/library, /schedule and /seo -- the ``?journey=`` mechanism already used by
/photos (``photos.journey_stages``) and /library (``library.build_context``).

Ground truth, verified against the code before writing these tests:
* ``/library`` ALREADY reads and filters on ``journey`` (``app.py::library_tab``
  -> ``library.build_context``). What it lacked was a *visible* indicator and a
  clear link.
* ``/schedule`` and ``/seo`` genuinely ignored ``?journey=`` before this change.
* The content ledger (``pipeline/schedule_report.py``) has no per-entry journey
  field, only ``region``. The only existing region<->journey mapping is
  ``config.drive.active_journey_by_region`` (one CURRENT active journey per
  region), so that is what ``/schedule`` filters through -- not a new mapping.
* SEO performance rows (``insights.performance_view``) are keyed by
  ``content_id``, which the sync writes as a SLUGIFIED VIDEO TITLE
  (``longform_metadata.py::_fetch_channel_videos``), not a video id or a
  journey. There is no honest way to filter those rows by journey without
  inventing an unverified heuristic, and the corrections/denylist ledger is a
  global place-name list with no journey concept at all. So /seo shows the
  journey selection (visible + carried in nav) but does NOT pretend to filter
  data that carries no journey identity -- it says so on the page instead.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from gomoveshift.config import Config
from gomoveshift.console import schedule as console_schedule
from gomoveshift.console.app import create_app

JOURNEY_A = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [photo-001.jpg]\n"
    "    photo_hashes: {photo-001.jpg: abc123}\n---\nBody text\n"
)
JOURNEY_B = (
    "---\ndays:\n- date: '2025-02-02'\n  photos:\n  - photos: [photo-002.jpg]\n"
    "    photo_hashes: {photo-002.jpg: def456}\n---\nBody text\n"
)


def _client() -> TestClient:
    return TestClient(create_app())


def _site(tmp_path):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(JOURNEY_A)
    (jdir / "journey-b.md").write_text(JOURNEY_B)
    return site


# --- /library: already filtered; this pins the visible indicator + clear
# link + the "no matches" wording that distinguishes a filtered-to-zero
# journey from the generic empty state. ---


def test_library_shows_visible_journey_indicator_and_clear_link(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library", params={"journey": "journey-a"})
    assert response.status_code == 200
    assert "journey-a" in response.text
    assert 'href="/library"' in response.text  # the clear link


def test_library_unknown_journey_is_not_the_unfiltered_list(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    unfiltered = _client().get("/library")
    assert "photo:journey-a" in unfiltered.text

    bogus = _client().get("/library", params={"journey": "no-such-journey"})
    assert bogus.status_code == 200
    assert "photo:journey-a" not in bogus.text
    assert "photo:journey-b" not in bogus.text
    assert "no-such-journey" in bogus.text


def test_library_no_journey_param_behaviour_unchanged(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library")
    assert response.status_code == 200
    assert "photo:journey-a" in response.text
    assert "photo:journey-b" in response.text


# --- /schedule: filters booked long-forms / availability by the region
# mapped to the journey via config.drive.active_journey_by_region. ---

_FAKE_REPORT = {
    "as_of": "2026-08-10T09:00:00-04:00",
    "weekly_longform_capacity": 3,
    "designed_days": ["tue", "fri", "sun"],
    "scheduled_longforms": [
        {
            "date": "2026-08-11",
            "time": "16:00:00",
            "region": "usa",
            "title": "Booked USA Drive",
            "kind": "longform",
        },
        {
            "date": "2026-08-13",
            "time": "14:00:00",
            "region": "ireland",
            "title": "Booked Ireland Drive",
            "kind": "longform",
        },
    ],
    "scheduled_shorts_count": 4,
    "queue_ends": "2026-08-13",
    "next_available": [
        {
            "region": "usa",
            "day": "tue",
            "slot": "2026-08-18T16:00:00-04:00",
            "shorts_windows": ["14:00", "18:00"],
            "shorts_required": 2,
            "shorts_slots": 14,
            "earliest_for_region": True,
        },
        {
            "region": "ireland",
            "day": "sun",
            "slot": "2026-08-17T14:00:00-04:00",
            "shorts_windows": ["08:00", "12:00"],
            "shorts_required": 2,
            "shorts_slots": 14,
            "earliest_for_region": True,
        },
    ],
    "next_available_superlong": [
        {"region": "usa", "day": "sat", "slot": "2026-08-15T15:00:00-04:00"},
    ],
}


def _config_with_journey_map(mapping):
    config = Config.load()
    config.drive.active_journey_by_region = dict(mapping)
    return config


def test_schedule_view_filters_by_journey_via_active_journey_region_map(monkeypatch):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config, **_kw: dict(_FAKE_REPORT)
    )
    config = _config_with_journey_map({"usa": "journey-a", "ireland": "journey-b"})
    result = console_schedule.schedule_view(config, journey="journey-a")
    assert [e["region"] for e in result["scheduled_longforms"]] == ["usa"]
    assert [e["region"] for e in result["next_available"]] == ["usa"]
    assert result["journey"] == "journey-a"
    assert result["journey_region"] == "usa"


def test_schedule_view_unknown_journey_filters_to_nothing_not_everything(monkeypatch):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config, **_kw: dict(_FAKE_REPORT)
    )
    config = _config_with_journey_map({"usa": "journey-a", "ireland": "journey-b"})
    result = console_schedule.schedule_view(config, journey="no-such-journey")
    assert result["scheduled_longforms"] == []
    assert result["next_available"] == []
    assert result["next_available_superlong"] == []
    assert result["journey"] == "no-such-journey"
    assert result["journey_region"] is None


def test_schedule_view_no_journey_param_behaviour_unchanged(monkeypatch):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config, **_kw: dict(_FAKE_REPORT)
    )
    config = _config_with_journey_map({"usa": "journey-a"})
    result = console_schedule.schedule_view(config)
    assert len(result["scheduled_longforms"]) == 2
    assert result["journey"] == ""


def test_schedule_route_renders_journey_indicator_and_filters(monkeypatch):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config, **_kw: dict(_FAKE_REPORT)
    )
    fake_config = _config_with_journey_map({"usa": "journey-a", "ireland": "journey-b"})
    monkeypatch.setattr(console_schedule.Config, "load", staticmethod(lambda: fake_config))
    response = _client().get("/schedule", params={"journey": "journey-a"})
    assert response.status_code == 200
    assert "Booked USA Drive" in response.text
    assert "Booked Ireland Drive" not in response.text
    assert "journey-a" in response.text
    assert 'href="/schedule"' in response.text  # the clear link


def test_schedule_route_unknown_journey_renders_visibly_not_full_list(monkeypatch):
    monkeypatch.setattr(
        console_schedule, "build_schedule_report", lambda ledger, config, **_kw: dict(_FAKE_REPORT)
    )
    fake_config = _config_with_journey_map({"usa": "journey-a"})
    monkeypatch.setattr(console_schedule.Config, "load", staticmethod(lambda: fake_config))
    response = _client().get("/schedule", params={"journey": "no-such-journey"})
    assert response.status_code == 200
    assert "Booked USA Drive" not in response.text
    assert "Booked Ireland Drive" not in response.text
    assert "no-such-journey" in response.text


# --- /seo: shows the journey selection and carries it, but documents (rather
# than fakes) that its data has no journey identity to filter on. ---


def test_seo_route_shows_journey_indicator_and_clear_link(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "does-not-exist.yaml"))
    response = _client().get("/seo", params={"journey": "journey-a"})
    assert response.status_code == 200
    assert "journey-a" in response.text
    assert 'href="/seo"' in response.text  # the clear link


def test_seo_route_no_journey_param_behaviour_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CORRECTIONS_PATH", str(tmp_path / "does-not-exist.yaml"))
    baseline = _client().get("/seo")
    assert baseline.status_code == 200
    assert "journey-a" not in baseline.text


# --- Navigation carries the selection between screens. ---


def test_nav_links_carry_the_current_journey_selection(monkeypatch, tmp_path):
    site = _site(tmp_path)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    response = _client().get("/library", params={"journey": "journey-a"})
    assert response.status_code == 200
    assert 'href="/schedule?journey=journey-a"' in response.text
    assert 'href="/seo?journey=journey-a"' in response.text


def test_nav_links_carry_nothing_when_no_journey_selected():
    response = _client().get("/")
    assert response.status_code == 200
    assert 'href="/schedule"' in response.text
    assert 'href="/seo"' in response.text
