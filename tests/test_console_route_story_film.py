"""Console door: launch a route-story-film job from a tracked profile."""

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import jobs
from gomoveshift.console.app import create_app
from gomoveshift.console.route_story import list_profiles, profile_dirs, resolve_profile


def _write_profile(directory, name="trip.yaml", body="output: film.mp4\n"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_list_profiles_reads_the_console_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    console_dir = profile_dirs()[-1]
    _write_profile(console_dir, "b.yaml")

    names = list_profiles()

    assert "b.yaml" in names


def test_resolve_profile_refuses_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))

    for bad in ("../x.yaml", "/abs/path.yaml", "no-such.yaml", "trip.yml"):
        try:
            resolve_profile(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected resolve_profile({bad!r}) to refuse")


def test_resolve_profile_finds_a_known_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    console_dir = profile_dirs()[-1]
    written = _write_profile(console_dir, "known.yaml")

    assert resolve_profile("known.yaml") == written


def test_get_profiles_endpoint_lists_known_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    console_dir = profile_dirs()[-1]
    _write_profile(console_dir, "known.yaml")
    client = TestClient(create_app())

    resp = client.get("/route-stories/profiles")

    assert resp.status_code == 200
    assert "known.yaml" in resp.json()


@pytest.mark.parametrize(
    "bad_profile",
    ["../x.yaml", "/etc/passwd.yaml", "nope.yaml"],
    ids=["traversal", "abs", "unknown"],
)
def test_post_film_refuses_a_bad_profile_name(tmp_path, monkeypatch, bad_profile):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = TestClient(create_app())

    resp = client.post("/route-stories/film", data={"profile": bad_profile})

    assert resp.status_code == 400


def test_post_film_launches_a_known_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    console_dir = profile_dirs()[-1]
    written = _write_profile(console_dir, "known.yaml")

    calls = []

    def _fake_launch(store, name, argv, log_dir, env=None, brand=None, rerun_of=None):
        calls.append((name, argv))
        return 42

    monkeypatch.setattr(jobs, "launch", _fake_launch)
    client = TestClient(create_app())

    resp = client.post(
        "/route-stories/film", data={"profile": "known.yaml"}, follow_redirects=False
    )

    assert resp.status_code == 303
    assert resp.headers["location"] == "/jobs/42"
    assert len(calls) == 1
    name, argv = calls[0]
    assert name == "route-story-film"
    assert argv[-3:] == ["route-story-film", "--profile", str(written)]


def test_profiles_are_read_from_the_inputs_dir(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    _write_profile(inputs / "route_story_profiles", "trip.yaml")
    monkeypatch.setenv("GMS_INPUTS_DIR", str(inputs))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))

    assert profile_dirs()[0] == inputs / "route_story_profiles"
    assert list_profiles() == ["trip.yaml"]
    assert resolve_profile("trip.yaml") == inputs / "route_story_profiles" / "trip.yaml"


def test_a_missing_inputs_dir_is_a_clear_422_on_the_profiles_endpoint(tmp_path, monkeypatch):
    missing = tmp_path / "gone"
    monkeypatch.setenv("GMS_INPUTS_DIR", str(missing))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = TestClient(create_app())

    resp = client.get("/route-stories/profiles")

    assert resp.status_code == 422
    assert "GMS_INPUTS_DIR" in resp.text
    assert str(missing) in resp.text


def test_a_missing_inputs_dir_refuses_a_film_launch_with_400(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INPUTS_DIR", str(tmp_path / "gone"))
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path / "console"))
    client = TestClient(create_app())

    resp = client.post("/route-stories/film", data={"profile": "trip.yaml"})

    assert resp.status_code == 400
    assert "GMS_INPUTS_DIR" in resp.text
