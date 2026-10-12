"""Console Photos tab -- ingest + develop launch only.

Selector/reel/astro are not covered here. ``build_photos_argv`` mirrors the real
``gomoveshift photos <action> ...`` CLI (``cli/parser.py::_add_photos_parser``).
"""

import sys

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import jobs, photos
from gomoveshift.console.app import create_app


class _FakePopen:
    def poll(self):
        return 0


def _client() -> TestClient:
    return TestClient(create_app())


def test_build_photos_argv_ingest_matches_a_manual_invocation():
    argv = photos.build_photos_argv(
        "ingest", card_dir="/mnt/card", journey="example-2026", day="2026-08-25"
    )
    assert argv == [
        sys.executable,
        "-m",
        "gomoveshift",
        "photos",
        "ingest",
        "/mnt/card",
        "--journey",
        "example-2026",
        "--day",
        "2026-08-25",
        "--dry-run",
    ]
    assert "photos" in jobs.ALLOWED_SUBCOMMANDS


def test_build_photos_argv_ingest_with_location_and_no_dry_run():
    argv = photos.build_photos_argv(
        "ingest",
        card_dir="/mnt/card",
        journey="example-2026",
        day="2026-08-25",
        location="Kerry, Ireland",
        dry_run=False,
    )
    assert argv[-2:] == ["--location", "Kerry, Ireland"]
    assert "--dry-run" not in argv


def test_build_photos_argv_develop_matches_a_manual_invocation():
    argv = photos.build_photos_argv(
        "develop", paths=["/mnt/card/a.jpg", "/mnt/card/b.jpg"], out="/tmp/develop-out"
    )
    assert argv == [
        sys.executable,
        "-m",
        "gomoveshift",
        "photos",
        "develop",
        "/mnt/card/a.jpg",
        "/mnt/card/b.jpg",
        "--out",
        "/tmp/develop-out",
    ]
    assert "--look" not in argv
    styled = photos.build_photos_argv(
        "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", styles="natural,vivid"
    )
    assert styled[-2:] == ["--styles", "natural,vivid"]


def test_build_photos_argv_develop_with_look_appends_the_flag():
    argv = photos.build_photos_argv(
        "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", look="cyberpunk"
    )
    assert argv[-2:] == ["--look", "cyberpunk"]


def test_build_photos_argv_develop_unknown_look_is_refused():
    with pytest.raises(ValueError):
        photos.build_photos_argv(
            "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", look="not-a-look"
        )


def test_build_photos_argv_develop_accepts_a_configured_look(monkeypatch):
    from gomoveshift import config as config_module

    cfg = config_module.get_config()
    cfg.photos.looks["my_custom_look"] = {"styles": "moody"}
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: cfg)
    argv = photos.build_photos_argv(
        "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", look="my_custom_look"
    )
    assert argv[-2:] == ["--look", "my_custom_look"]


def test_build_photos_argv_unknown_action_raises():
    with pytest.raises(ValueError):
        photos.build_photos_argv("selector", card_dir="/mnt/card")


def test_photos_tab_renders_both_forms():
    resp = _client().get("/photos")
    assert resp.status_code == 200
    assert "/photos/run" in resp.text
    assert "ingest" in resp.text.lower()
    assert "develop" in resp.text.lower()


def test_photos_run_ingest_launches_through_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    resp = _client().post(
        "/photos/run",
        data={
            "action": "ingest",
            "card_dir": str(tmp_path / "card"),
            "journey": "example-2026",
            "day": "2026-08-25",
            "dry_run": "on",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert row["name"] == "photos"
    assert str(tmp_path / "card") in row["argv"]
    assert "--dry-run" in row["argv"]


def test_photos_run_develop_launches_through_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    (tmp_path / "a.jpg").write_bytes(b"x")
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert row["name"] == "photos"
    assert str(tmp_path / "a.jpg") in row["argv"]
    assert str(tmp_path / "develop-out") in row["argv"]


def test_photos_run_unknown_action_is_400(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post("/photos/run", data={"action": "selector"})
    assert resp.status_code == 400


def test_photos_run_develop_with_look_reaches_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    (tmp_path / "a.jpg").write_bytes(b"x")
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "look": "cyberpunk",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert "--look" in row["argv"]
    assert "cyberpunk" in row["argv"]


def test_photos_run_develop_unknown_look_is_400(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "look": "not-a-look",
        },
    )
    assert resp.status_code == 400


def test_photos_tab_lists_a_known_builtin_look():
    resp = _client().get("/photos")
    assert resp.status_code == 200
    assert "cyberpunk" in resp.text


@pytest.mark.parametrize(
    ("kwarg", "flag", "value"),
    [
        ("vignette", "--vignette", 0.5),
        ("clarity", "--clarity", -0.5),
        ("grain", "--grain", 0.5),
        ("grain_size", "--grain-size", 2.0),
        ("halation", "--halation", 0.5),
    ],
)
def test_build_photos_argv_develop_dial_appears_when_supplied(kwarg, flag, value):
    argv = photos.build_photos_argv(
        "develop",
        paths=["/mnt/card/a.jpg"],
        out="/tmp/develop-out",
        **{kwarg: value},
    )
    assert argv[-2:] == [flag, str(value)]


def test_build_photos_argv_develop_omitted_dials_contribute_nothing():
    argv = photos.build_photos_argv("develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out")
    for flag in ("--vignette", "--clarity", "--grain", "--grain-size", "--halation"):
        assert flag not in argv


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("vignette", 1.5),
        ("vignette", -0.1),
        ("clarity", 1.1),
        ("clarity", -1.1),
        ("grain", 1.5),
        ("grain", -0.1),
        ("grain_size", 0.1),
        ("grain_size", 4.1),
        ("halation", 1.5),
        ("halation", -0.1),
    ],
)
def test_build_photos_argv_develop_out_of_range_dial_raises(kwarg, value):
    with pytest.raises(ValueError):
        photos.build_photos_argv(
            "develop",
            paths=["/mnt/card/a.jpg"],
            out="/tmp/develop-out",
            **{kwarg: value},
        )


def test_build_photos_argv_develop_several_dials_at_once():
    argv = photos.build_photos_argv(
        "develop",
        paths=["/mnt/card/a.jpg"],
        out="/tmp/develop-out",
        vignette=0.3,
        grain=0.4,
        grain_size=1.5,
    )
    assert argv[-6:] == [
        "--vignette",
        "0.3",
        "--grain",
        "0.4",
        "--grain-size",
        "1.5",
    ]


def test_photos_run_develop_dials_reach_the_launched_job_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    (tmp_path / "a.jpg").write_bytes(b"x")
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "vignette": "0.5",
            "clarity": "-0.2",
            "grain": "0.3",
            "grain_size": "1.5",
            "halation": "0.1",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    for flag, value in (
        ("--vignette", "0.5"),
        ("--clarity", "-0.2"),
        ("--grain", "0.3"),
        ("--grain-size", "1.5"),
        ("--halation", "0.1"),
    ):
        assert flag in row["argv"]
        assert value in row["argv"]


def test_photos_run_develop_out_of_range_dial_is_400(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "vignette": "5.0",
        },
    )
    assert resp.status_code == 400


def test_photos_tab_renders_the_five_dial_inputs():
    resp = _client().get("/photos")
    assert resp.status_code == 200
    for name in ("vignette", "clarity", "grain", "grain_size", "halation"):
        assert f'name="{name}"' in resp.text


# The white-balance group: signed shifts on the two temperature
# axes (r/b) and their highlight-only halves, plus green-magenta and its
# highlight half. Ranges mirror ``cli/parser.py``'s develop subparser
# (``config.WB_OPERATOR_SPAN`` for r/b/r-high/b-high, 0.5 for g/g-high).
@pytest.mark.parametrize(
    ("kwarg", "flag", "value"),
    [
        ("white_balance_g", "--white-balance-g", 0.3),
        ("white_balance_r", "--white-balance-r", 0.05),
        ("white_balance_b", "--white-balance-b", -0.05),
        ("white_balance_r_high", "--white-balance-r-high", 0.1),
        ("white_balance_b_high", "--white-balance-b-high", -0.1),
        ("white_balance_g_high", "--white-balance-g-high", -0.5),
    ],
)
def test_build_photos_argv_develop_white_balance_dial_appears_when_supplied(kwarg, flag, value):
    argv = photos.build_photos_argv(
        "develop",
        paths=["/mnt/card/a.jpg"],
        out="/tmp/develop-out",
        **{kwarg: value},
    )
    assert argv[-2:] == [flag, str(value)]


def test_build_photos_argv_develop_white_balance_omitted_contributes_nothing():
    argv = photos.build_photos_argv("develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out")
    for flag in (
        "--white-balance-g",
        "--white-balance-r",
        "--white-balance-b",
        "--white-balance-r-high",
        "--white-balance-b-high",
        "--white-balance-g-high",
    ):
        assert flag not in argv


@pytest.mark.parametrize(
    ("kwarg", "value"),
    [
        ("white_balance_g", 0.6),
        ("white_balance_g", -0.6),
        ("white_balance_r", 0.11),
        ("white_balance_r", -0.11),
        ("white_balance_b", 0.11),
        ("white_balance_r_high", 0.2),
        ("white_balance_b_high", -0.2),
        ("white_balance_g_high", 0.51),
    ],
)
def test_build_photos_argv_develop_white_balance_out_of_range_raises(kwarg, value):
    with pytest.raises(ValueError):
        photos.build_photos_argv(
            "develop",
            paths=["/mnt/card/a.jpg"],
            out="/tmp/develop-out",
            **{kwarg: value},
        )


def test_photos_run_develop_white_balance_reaches_the_launched_job_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    (tmp_path / "a.jpg").write_bytes(b"x")
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "white_balance_g": "0.2",
            "white_balance_r": "0.05",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    for flag, value in (("--white-balance-g", "0.2"), ("--white-balance-r", "0.05")):
        assert flag in row["argv"]
        assert value in row["argv"]


def test_photos_run_develop_white_balance_out_of_range_is_400(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "white_balance_r": "5.0",
        },
    )
    assert resp.status_code == 400


# Panorama stitch. Not yet wired into the ``/photos/run`` form (that
# needs its own job-launch branch); these cover the console-side building
# blocks -- path validation and the direct in-process stitch call -- on
# their own.
def test_validate_panorama_paths_requires_at_least_two():
    errors = photos.validate_panorama_paths(["/tmp/a.jpg"])
    assert any("at least 2" in e for e in errors)


def test_validate_panorama_paths_reports_a_missing_file(tmp_path):
    a = tmp_path / "a.jpg"
    a.write_bytes(b"x")
    errors = photos.validate_panorama_paths([str(a), str(tmp_path / "missing.jpg")])
    assert any("no such file" in e for e in errors)


def test_validate_panorama_paths_accepts_two_real_files(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    assert photos.validate_panorama_paths([str(a), str(b)]) == []


def test_run_panorama_rejects_invalid_paths(tmp_path):
    with pytest.raises(ValueError):
        photos.run_panorama([str(tmp_path / "missing.jpg")], str(tmp_path / "out.png"))


def test_run_panorama_calls_the_pipeline_stitch(tmp_path, monkeypatch):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    out = tmp_path / "out.png"

    class _FakeResult:
        out_path = out
        width = 10
        height = 20
        valid_area_fraction = 0.9

    calls = []

    def _fake_stitch(paths, out_path):
        calls.append((list(paths), out_path))
        return _FakeResult()

    monkeypatch.setattr("gomoveshift.pipeline.photos.panorama.stitch_panorama", _fake_stitch)
    result = photos.run_panorama([str(a), str(b)], str(out))
    assert result == {
        "out_path": str(out),
        "width": 10,
        "height": 20,
        "valid_area_fraction": 0.9,
    }
    assert calls == [([str(a), str(b)], str(out))]


def test_photos_tab_renders_the_white_balance_inputs():
    resp = _client().get("/photos")
    assert resp.status_code == 200
    for name in (
        "white_balance_g",
        "white_balance_r",
        "white_balance_b",
        "white_balance_r_high",
        "white_balance_b_high",
        "white_balance_g_high",
    ):
        assert f'name="{name}"' in resp.text


# The develop form offers a 16-bit TIFF master as an opt-in; JPEG stays
# the default and contributes nothing to the argv.
_BASE_DEVELOP = [
    sys.executable,
    "-m",
    "gomoveshift",
    "photos",
    "develop",
    "/mnt/card/a.jpg",
    "--out",
    "/tmp/develop-out",
]


@pytest.mark.parametrize("value", [None, "", "jpeg"])
def test_develop_argv_default_export_format_is_unchanged(value):
    argv = photos.build_photos_argv(
        "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", export_format=value
    )
    assert argv == _BASE_DEVELOP
    assert "--format" not in argv


def test_develop_argv_tiff16_appends_the_format_flag():
    argv = photos.build_photos_argv(
        "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", export_format="tiff16"
    )
    assert argv == [*_BASE_DEVELOP, "--format", "tiff16"]


def test_develop_argv_bad_export_format_raises_naming_allowed_values():
    with pytest.raises(ValueError, match="jpeg, tiff16"):
        photos.build_photos_argv(
            "develop", paths=["/mnt/card/a.jpg"], out="/tmp/develop-out", export_format="png"
        )


def test_photos_run_develop_export_format_reaches_the_launched_job_argv(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    (tmp_path / "a.jpg").write_bytes(b"x")
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "export_format": "tiff16",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    argv = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)["argv"]
    i = argv.index("--format")
    assert argv[i : i + 2] == ["--format", "tiff16"]


def test_photos_run_develop_bad_export_format_is_400(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())
    (tmp_path / "a.jpg").write_bytes(b"x")
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "export_format": "png",
        },
    )
    assert resp.status_code == 400


def test_photos_tab_renders_the_master_format_select_defaulting_to_jpeg():
    resp = _client().get("/photos")
    assert resp.status_code == 200
    assert 'name="export_format"' in resp.text
    assert "JPEG (default)" in resp.text
    assert "16-bit TIFF master: larger files, slower" in resp.text
    assert 'value="tiff16" selected' not in resp.text


def test_photos_form_keeps_the_chosen_master_format_on_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/photos/run",
        data={
            "action": "develop",
            "paths": str(tmp_path / "a.jpg"),
            "out": str(tmp_path / "develop-out"),
            "export_format": "tiff16",
            "vignette": "5.0",
        },
    )
    assert resp.status_code == 400
    assert 'value="tiff16" selected' in resp.text
