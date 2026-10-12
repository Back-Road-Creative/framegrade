"""Durable, brand-scoped Route Story project records."""

from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story.project import RouteStoryStore


def _source(root: Path, name: str, body: str = "source") -> Path:
    path = root / name
    path.write_text(body, encoding="utf-8")
    return path


def test_project_is_idempotent_and_persists_a_reviewable_recipe(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    track = _source(root, "trip.gpx")
    photo = _source(root, "photo.jpg")
    store = RouteStoryStore(tmp_path / "projects")
    recipe = {"fps": 30, "scenes": [{"kind": "route_follow", "start_frame": 0, "end_frame": 30}]}

    first = store.create("gomoveshift", [track, photo], [root], recipe)
    second = store.create("gomoveshift", [photo, track], [root], recipe)
    row = store.get("gomoveshift", first)

    assert first == second
    assert row["status"] == "draft"
    assert row["privacy_status"] == "review_required"
    assert row["recipe"] == recipe
    assert {Path(item["path"]).name for item in row["sources"]} == {"trip.gpx", "photo.jpg"}
    assert all(item["sha256"] for item in row["sources"])


def test_project_rejects_sources_outside_configured_roots(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    outside = _source(tmp_path, "outside.gpx")

    with pytest.raises(ValueError, match="configured roots"):
        RouteStoryStore(tmp_path / "projects").create("gomoveshift", [outside], [root], {"fps": 30})


def test_project_brand_boundary_prevents_cross_brand_reads(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    project = RouteStoryStore(tmp_path / "projects").create(
        "gomoveshift", [_source(root, "trip.gpx")], [root], {"fps": 30}
    )

    with pytest.raises(KeyError):
        RouteStoryStore(tmp_path / "projects").get("examplebrand", project)


def test_update_persists_media_placement_decisions_for_replay(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    store = RouteStoryStore(tmp_path / "projects")
    project = store.create("gomoveshift", [_source(root, "trip.gpx")], [root], {"fps": 30})

    updated = store.update(
        "gomoveshift",
        project,
        media_placements={"DSC_0001.NEF": {"position_source": "nearest_observation"}},
    )

    assert updated["media_placements"]["DSC_0001.NEF"]["position_source"] == "nearest_observation"
    row = store.get("gomoveshift", project)
    assert row["media_placements"] == updated["media_placements"]
    assert row["updated_at"] >= row["created_at"]
    assert row["status"] == "draft"


def test_update_rejects_an_unknown_project(tmp_path):
    store = RouteStoryStore(tmp_path / "projects")

    with pytest.raises(KeyError):
        store.update("gomoveshift", "missing", media_placements={})


def test_update_requires_at_least_one_field(tmp_path):
    root = tmp_path / "sources"
    root.mkdir()
    store = RouteStoryStore(tmp_path / "projects")
    project = store.create("gomoveshift", [_source(root, "trip.gpx")], [root], {"fps": 30})

    with pytest.raises(ValueError, match="field"):
        store.update("gomoveshift", project)
