"""The develop form's ``paths``/``out`` fields stay visible
free text (no ``required`` control is hidden inside a closed ``<details>`` --
verified against the template directly), but nothing checked them against the
filesystem before a job was launched. A missing input photo or an output path
that is actually a file only surfaced deep inside the job log, long after the
operator's form was gone. This is the upstream gate: refuse before ``jobs.launch``,
with a message that names the specific problem.
"""

from gomoveshift.console import jobs, photos


class _FakePopen:
    def poll(self):
        return 0


def test_no_required_control_is_hidden_inside_a_closed_details_element():
    # The comment in photos.html deliberately keeps every `required`
    # control outside `<details>`. Confirm that invariant holds structurally
    # rather than trusting the comment.
    from pathlib import Path

    text = (
        Path(__file__).parent.parent / "gomoveshift" / "console" / "templates" / "photos.html"
    ).read_text()
    details_start = text.index("<details")
    details_end = text.index("</details>") + len("</details>")
    assert "required" not in text[details_start:details_end]


def test_validate_develop_paths_rejects_a_missing_photo(tmp_path):
    missing = tmp_path / "does-not-exist.jpg"
    errors = photos.validate_develop_paths([str(missing)], str(tmp_path / "out"))
    assert errors == [f"{missing}: no such file"]


def test_validate_develop_paths_rejects_a_directory_given_as_a_photo(tmp_path):
    a_dir = tmp_path / "a-directory"
    a_dir.mkdir()
    errors = photos.validate_develop_paths([str(a_dir)], str(tmp_path / "out"))
    assert errors == [f"{a_dir}: is a directory, not a photo file"]


def test_validate_develop_paths_rejects_out_that_is_a_file(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    out_file = tmp_path / "out.txt"
    out_file.write_text("not a directory")
    errors = photos.validate_develop_paths([str(photo)], str(out_file))
    assert errors == [f"{out_file}: is a file, not a directory"]


def test_validate_develop_paths_accepts_a_real_photo_and_a_not_yet_created_out(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    errors = photos.validate_develop_paths([str(photo)], str(tmp_path / "brand-new-out-dir"))
    assert errors == []


def test_photos_run_develop_missing_photo_is_400_with_a_specific_message(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    client = TestClient(create_app())
    missing = tmp_path / "missing.jpg"
    resp = client.post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(missing),
            "out": str(tmp_path / "develop-out"),
        },
    )
    assert resp.status_code == 400
    assert "no such file" in resp.text


def test_photos_run_develop_with_a_real_photo_still_launches(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    from fastapi.testclient import TestClient

    from gomoveshift.console.app import create_app

    client = TestClient(create_app())
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"x")
    resp = client.post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(photo),
            "out": str(tmp_path / "develop-out"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
