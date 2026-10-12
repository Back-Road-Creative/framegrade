"""Private night results are discoverable without becoming gallery assets."""

import json
import sqlite3

from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app


def seed(root, brand="gomoveshift", project="a" * 64):
    root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "projects.sqlite3") as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS night_projects (id TEXT PRIMARY KEY, brand TEXT, data TEXT)"
        )
        db.execute(
            "INSERT INTO night_projects VALUES (?, ?, ?)",
            (
                project,
                brand,
                json.dumps(
                    {
                        "id": project,
                        "brand": brand,
                        "recipe": "natural",
                        "status": "complete",
                        "items": [{"status": "complete", "source": "/private/card/stack.linear"}],
                    }
                ),
            ),
        )


def test_library_exposes_private_projects_without_site_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("GMS_BRAND", "gomoveshift")
    monkeypatch.delenv("GMS_SITE_PATH", raising=False)
    seed(tmp_path / "night-projects")
    seed(tmp_path / "night-projects", "another-brand", "b" * 64)
    response = TestClient(create_app()).get("/library")
    assert response.status_code == 200
    assert "Private night projects" in response.text
    assert "/photos/night/" + "a" * 64 in response.text
    assert "b" * 64 not in response.text
    assert "/private/card" not in response.text
    assert "Not approved for publishing" in response.text


def test_missing_private_store_is_not_created(tmp_path):
    from gomoveshift.catalog.night import read_night_projects

    root = tmp_path / "absent"
    assert read_night_projects(root, "gomoveshift") == {"projects": [], "unavailable": False}
    assert not root.exists()


def test_private_catalog_rejects_symlink_and_corruption(tmp_path):
    from gomoveshift.catalog.night import read_night_projects

    target = tmp_path / "target"
    seed(target)
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    assert read_night_projects(link, "gomoveshift")["unavailable"]
    (target / "projects.sqlite3").write_bytes(b"broken")
    assert read_night_projects(target, "gomoveshift")["unavailable"]


def test_private_catalog_skips_forged_metadata_and_bounds_listing(tmp_path):
    from gomoveshift.catalog.night import read_night_projects

    seed(tmp_path)
    with sqlite3.connect(tmp_path / "projects.sqlite3") as db:
        row = json.loads(db.execute("SELECT data FROM night_projects").fetchone()[0])
        row["brand"] = "foreign"
        db.execute("UPDATE night_projects SET data=?", (json.dumps(row),))
    assert read_night_projects(tmp_path, "gomoveshift") == {"projects": [], "unavailable": True}
    for index in range(55):
        seed(tmp_path, project=f"{index:064x}")
    result = read_night_projects(tmp_path, "gomoveshift")
    assert len(result["projects"]) == 50
    assert result["projects"][0]["id"] == f"{54:064x}"


def test_disabled_photos_hides_private_library_links(tmp_path, monkeypatch):
    from gomoveshift.console import app as console_app

    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(console_app, "brand_disabled_tabs", lambda _: {"photos"})
    seed(tmp_path / "night-projects")
    response = TestClient(create_app()).get("/library")
    assert response.status_code == 200
    assert "Private night projects" not in response.text


def test_deeply_nested_record_does_not_hide_valid_neighbors(tmp_path):
    from gomoveshift.catalog.night import read_night_projects

    seed(tmp_path)
    with sqlite3.connect(tmp_path / "projects.sqlite3") as db:
        db.execute(
            "INSERT INTO night_projects VALUES (?, ?, ?)",
            ("b" * 64, "gomoveshift", "[" * 20000 + "]" * 20000),
        )
    result = read_night_projects(tmp_path, "gomoveshift")
    assert result["unavailable"]
    assert [row["id"] for row in result["projects"]] == ["a" * 64]


def test_typed_product_project_is_visible_with_product_progress(tmp_path):
    from gomoveshift.catalog.night import read_night_projects

    identifier = "c" * 64
    seed(tmp_path, project=identifier)
    with sqlite3.connect(tmp_path / "projects.sqlite3") as db:
        row = json.loads(db.execute("SELECT data FROM night_projects").fetchone()[0])
        row.update(
            schema=2,
            kind="night_products",
            status="partial",
            requested_products=["best_stills", "timelapse"],
            tasks=[
                {"product": "best_stills", "status": "complete"},
                {"product": "timelapse", "status": "partial"},
            ],
            private_exports=[{"export_id": "e" * 64}],
        )
        row.pop("items")
        db.execute("UPDATE night_projects SET data=?", (json.dumps(row),))

    result = read_night_projects(tmp_path, "gomoveshift")

    assert result["unavailable"] is False
    assert result["projects"] == [
        {
            "id": identifier,
            "recipe": "natural",
            "status": "partial",
            "completed": 1,
            "total": 2,
            "kind": "products",
            "selected": 1,
        }
    ]
