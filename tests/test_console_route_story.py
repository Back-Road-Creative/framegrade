"""Private console controls for durable Route Story drafts."""

from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app


def test_console_creates_and_lists_private_route_story_drafts(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    track = sources / "trip.gpx"
    track.write_text("track", encoding="utf-8")
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    monkeypatch.setenv("GMS_ROUTE_STORY_SOURCE_ROOTS", str(sources))

    client = TestClient(create_app())
    created = client.post(
        "/route-stories",
        json={"sources": [str(track)], "recipe": {"fps": 30}},
    )

    assert created.status_code == 201
    project = created.json()["id"]
    listed = client.get("/route-stories")
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()] == [project]
    assert listed.json()[0]["privacy_status"] == "review_required"


def test_console_reads_a_draft_telemetry_summary_from_its_gpx(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    track = sources / "trip.gpx"
    track.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    monkeypatch.setenv("GMS_ROUTE_STORY_SOURCE_ROOTS", str(sources))
    client = TestClient(create_app())
    project = client.post(
        "/route-stories", json={"sources": [str(track)], "recipe": {"fps": 30}}
    ).json()["id"]

    response = client.get(f"/route-stories/{project}/telemetry")

    assert response.status_code == 200
    assert response.json()["current_elevation_m"] == 110.0
    assert response.json()["distance_m"] > 1_000
    assert response.json()["current_grade_percent"] is not None


def test_console_describes_a_private_satellite_opening_from_saved_sources(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    track = sources / "trip.gpx"
    track.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><time>2026-09-16T12:00:00Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    satellite = sources / "usa.png"
    from PIL import Image

    Image.new("RGB", (360, 180), "#4a7f45").save(satellite)
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    monkeypatch.setenv("GMS_ROUTE_STORY_SOURCE_ROOTS", str(sources))
    client = TestClient(create_app())
    project = client.post(
        "/route-stories",
        json={
            "sources": [str(track), str(satellite)],
            "recipe": {
                "fps": 30,
                "opening": {
                    "frames": 300,
                    "final_zoom": 10,
                    "satellite": {
                        "west": -180,
                        "south": -90,
                        "east": 180,
                        "north": 90,
                        "credit": "Test",
                    },
                },
            },
        },
    ).json()["id"]

    response = client.get(f"/route-stories/{project}/opening")

    assert response.status_code == 200
    assert response.json()["camera"]["end"]["longitude"] == -80.0
    assert response.json()["camera"]["end"]["latitude"] == 35.0
    assert response.json()["satellite"]["credit"] == "Test"

    Image.new("RGB", (360, 180), "#123456").save(satellite)
    changed = client.get(f"/route-stories/{project}/opening")
    assert changed.status_code == 422
    assert "changed" in changed.text
