"""A real picker over the selectable photo set for the selected journey,
instead of requiring the operator to type an absolute path per photo.

``journey_photo_options`` reads the same directory the selector treats as the
live gallery (``selector.IMAGES_BASE`` under ``selector.default_site_root()``)
so the console never re-derives a second notion of "what's selectable".  An
empty/unresolvable set names which one of four distinct facts it is --
collapsing them to one blank list is the exact silence this test guards against.

The typed-path textarea keeps working; picked photos are additive and both
route through the same ``validate_develop_paths`` gate before a job launches.
"""

from gomoveshift.console import jobs, photos


class _FakePopen:
    def poll(self):
        return 0


class _FakePath:
    """Minimal stand-in supporting ``/`` and a raising ``iterdir`` -- used to
    exercise the "directory exists but can't be read" branch without relying
    on real filesystem permission enforcement (unreliable as root/dev)."""

    def __init__(self, raise_on_iterdir=False):
        self._raise = raise_on_iterdir

    def __truediv__(self, other):
        return self

    def exists(self):
        return True

    def iterdir(self):
        if self._raise:
            raise OSError("permission denied")
        return iter(())


def test_journey_photo_options_no_journey_selected():
    result = photos.journey_photo_options("")
    assert result["status"] == "no_journey"
    assert result["photos"] == []


def test_journey_photo_options_no_images_directory(tmp_path):
    result = photos.journey_photo_options("example-2026", site_root=tmp_path)
    assert result["status"] == "no_dir"
    assert result["photos"] == []
    assert "example-2026" in result["message"]


def test_journey_photo_options_empty_directory(tmp_path):
    (tmp_path / "static" / "images" / "journeys" / "example-2026").mkdir(parents=True)
    result = photos.journey_photo_options("example-2026", site_root=tmp_path)
    assert result["status"] == "empty"
    assert result["photos"] == []


def test_journey_photo_options_lists_real_photos(tmp_path):
    images_dir = tmp_path / "static" / "images" / "journeys" / "example-2026"
    images_dir.mkdir(parents=True)
    (images_dir / "b.jpg").write_bytes(b"x")
    (images_dir / "a.jpg").write_bytes(b"x")
    result = photos.journey_photo_options("example-2026", site_root=tmp_path)
    assert result["status"] == "ok"
    assert [p["filename"] for p in result["photos"]] == ["a.jpg", "b.jpg"]
    assert result["photos"][0]["path"] == str(images_dir / "a.jpg")


def test_journey_photo_options_unreadable_directory_names_that_fact():
    result = photos.journey_photo_options(
        "example-2026", site_root=_FakePath(raise_on_iterdir=True)
    )
    assert result["status"] == "unreadable"
    assert result["photos"] == []


def test_photos_tab_renders_picker_photos_for_a_selected_journey(tmp_path, monkeypatch):
    from gomoveshift import config as config_module

    images_dir = tmp_path / "static" / "images" / "journeys" / "example-2026"
    images_dir.mkdir(parents=True)
    photo = images_dir / "a.jpg"
    photo.write_bytes(b"x")
    cfg = config_module.get_config()
    cfg.photos.site_root = str(tmp_path)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: cfg)

    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    resp = TestClient(create_app()).get("/photos", params={"journey": "example-2026"})
    assert resp.status_code == 200
    assert str(photo) in resp.text
    assert 'name="picked_paths"' in resp.text


def test_photos_tab_names_missing_images_directory_for_a_journey(tmp_path, monkeypatch):
    from gomoveshift import config as config_module

    cfg = config_module.get_config()
    cfg.photos.site_root = str(tmp_path)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: cfg)

    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    resp = TestClient(create_app()).get("/photos", params={"journey": "no-such-journey"})
    assert resp.status_code == 200
    assert "no-such-journey" in resp.text
    assert "no images directory" in resp.text.lower()


def test_picked_paths_merge_with_typed_paths_and_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    typed = tmp_path / "a.jpg"
    typed.write_bytes(b"x")
    picked = tmp_path / "b.jpg"
    picked.write_bytes(b"x")

    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    resp = TestClient(create_app()).post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(typed),
            "picked_paths": [str(picked)],
            "out": str(tmp_path / "develop-out"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert str(typed) in row["argv"]
    assert str(picked) in row["argv"]


def test_picked_paths_alone_still_validate_before_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    missing = tmp_path / "missing.jpg"
    resp = TestClient(create_app()).post(
        "/photos/run",
        data={
            "action": "develop",
            "picked_paths": [str(missing)],
            "out": str(tmp_path / "develop-out"),
        },
    )
    assert resp.status_code == 400
    assert "no such file" in resp.text


def test_develop_with_neither_typed_nor_picked_paths_is_400(tmp_path, monkeypatch):
    # The textarea lost its HTML `required` now that the picker is a second,
    # additive way to supply photos -- "gave neither" must still be an
    # upstream gate, not a bare argparse failure inside the job log.
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    resp = TestClient(create_app()).post(
        "/photos/run",
        data={"action": "develop", "out": str(tmp_path / "develop-out")},
    )
    assert resp.status_code == 400
    assert "required" in resp.text.lower()
