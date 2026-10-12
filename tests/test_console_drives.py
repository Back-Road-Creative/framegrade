"""Console Drives tab -- dry-run launch + gate answer form."""

import json
import sys

from fastapi.testclient import TestClient

from gomoveshift.console import drives, jobs
from gomoveshift.console.app import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def _write_tiny_video(path) -> None:
    """A real, readable single-frame .mp4 -- same fixture pattern as
    ``tests/test_web_frame_executor_offload.py``'s ``video_root``."""
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    assert writer.isOpened(), "could not open VideoWriter for fixture"
    for _ in range(5):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    assert path.exists() and path.stat().st_size > 0


def test_build_drive_argv_matches_a_manual_invocation():
    argv = drives.build_drive_argv("/data/day1", output_dir="/data/out", region="usa")
    assert argv == [
        sys.executable,
        "-m",
        "gomoveshift",
        "drive",
        "/data/day1",
        "--region",
        "usa",
        "--output-dir",
        "/data/out",
        "--no-review",
        "--dry-run",
        "--no-tmux",
    ]
    no_dry = drives.build_drive_argv("/data/day1", output_dir="/data/out", dry_run=False)
    assert no_dry[-1] == "--no-tmux" and "--dry-run" not in no_dry
    no_region = drives.build_drive_argv("/data/day1", output_dir="/data/out")
    assert "--region" not in no_region
    assert "drive" in jobs.ALLOWED_SUBCOMMANDS


def test_build_drive_argv_always_asserts_no_tmux():
    """The console is its own supervisor (job store + container
    restart policy) -- every console-launched drive must skip the tmux gate."""
    argv = drives.build_drive_argv("/data/day1", output_dir="/data/out")
    assert "--no-tmux" in argv


def test_build_drive_argv_emits_paired_publish_and_lf_day():
    argv = drives.build_drive_argv(
        "/data/day1", output_dir="/data/out", publish="tue", lf_day="tue"
    )
    assert "--publish" in argv and argv[argv.index("--publish") + 1] == "tue"
    assert "--lf-day" in argv and argv[argv.index("--lf-day") + 1] == "tue"


def test_build_drive_argv_omits_publish_and_lf_day_when_both_absent():
    argv = drives.build_drive_argv("/data/day1", output_dir="/data/out")
    assert "--publish" not in argv
    assert "--lf-day" not in argv


def test_build_drive_argv_raises_when_only_one_of_publish_lf_day_given():
    import pytest

    with pytest.raises(ValueError):
        drives.build_drive_argv("/data/day1", output_dir="/data/out", publish="tue")
    with pytest.raises(ValueError):
        drives.build_drive_argv("/data/day1", output_dir="/data/out", lf_day="tue")


def test_drives_run_launches_through_the_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/drives/run",
        data={
            "brand": "gomoveshift",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "region": "usa",
            "dry_run": "on",
            "publish": "tue",
            "lf_day": "tue",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert row["name"] == "drive"
    assert str(tmp_path / "day1") in row["argv"]
    # The review checkbox defaults OFF -- review stays ON unless the
    # operator explicitly opts out, so --no-review is absent here.
    assert "--dry-run" in row["argv"] and "--no-review" not in row["argv"]


def test_drives_run_accepts_paired_publish_and_lf_day(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/drives/run",
        data={
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "region": "usa",
            "publish": "tue",
            "lf_day": "tue",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert "--publish" in row["argv"] and "--lf-day" in row["argv"]


def test_build_drive_argv_accepts_the_curated_pipeline_flags():
    argv = drives.build_drive_argv(
        "/data/day1",
        output_dir="/data/out",
        lf_mode="full_drive",
        lf_target_duration=15.0,
        superlong=True,
        shorts_target=6,
        shorts_gap=45.0,
    )
    assert argv[argv.index("--lf-mode") + 1] == "full_drive"
    assert argv[argv.index("--lf-target-duration") + 1] == "15.0"
    assert "--superlong" in argv
    assert argv[argv.index("--shorts-target") + 1] == "6"
    assert argv[argv.index("--shorts-gap") + 1] == "45.0"


def test_build_drive_argv_omits_curated_flags_when_absent():
    argv = drives.build_drive_argv("/data/day1", output_dir="/data/out")
    for flag in (
        "--lf-mode",
        "--lf-target-duration",
        "--superlong",
        "--shorts-target",
        "--shorts-gap",
    ):
        assert flag not in argv


def test_build_drive_argv_rejects_an_invalid_lf_mode():
    import pytest

    with pytest.raises(ValueError):
        drives.build_drive_argv("/data/day1", output_dir="/data/out", lf_mode="bogus")


def test_drives_run_accepts_the_curated_pipeline_flags_and_brand(
    shipped_brands, tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/drives/run",
        data={
            "brand": "examplebrand",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "publish": "fri",
            "lf_day": "fri",
            "lf_mode": "full_drive",
            "lf_target_duration": "15",
            "superlong": "on",
            "shorts_target": "6",
            "shorts_gap": "45",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert "--lf-mode" in row["argv"] and "full_drive" in row["argv"]
    assert "--superlong" in row["argv"]
    assert "--shorts-target" in row["argv"]


def test_drives_run_rejects_a_non_numeric_lf_target_duration(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/drives/run",
        data={
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "lf_target_duration": "not-a-number",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_drives_run_rejects_an_unknown_brand(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/drives/run",
        data={
            "brand": "not-a-real-brand",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_drives_tab_renders_the_pipeline_flags_fieldset(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().get("/drives")
    assert resp.status_code == 200
    assert "Pipeline flags" in resp.text
    # The CLI flag stays visible for diagnostics, but is no longer the label
    # text itself -- see the plain-language assertions below.
    assert "--lf-mode" in resp.text
    assert "--shorts-target" in resp.text


def test_drives_tab_uses_plain_language_for_the_long_form_labels(tmp_path, monkeypatch):
    """'LF' is jargon for the long-form video assembled from a day's
    footage (every LF is assembled from one calendar day's
    footage). The primary label must say "long-form" in plain words, and
    the underlying CLI flag must still be visible somewhere for a person
    debugging -- retained, not deleted."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().get("/drives")
    assert resp.status_code == 200
    assert "Long-form day" in resp.text
    assert "Long-form selection mode" in resp.text
    assert "Long-form target duration" in resp.text
    # The primary labels no longer lead with the bare jargon token.
    assert "<label>LF day" not in resp.text
    assert "<label>LF selection mode" not in resp.text
    assert "<label>LF target duration" not in resp.text
    # The CLI flags are retained as diagnostic detail, not deleted.
    assert "--lf-mode" in resp.text
    assert "--lf-target-duration" in resp.text


def test_drives_run_rejects_unpaired_publish_or_lf_day(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().post(
        "/drives/run",
        data={
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "publish": "tue",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400


def test_drives_run_refuses_when_neither_publish_nor_lf_day_given(tmp_path, monkeypatch):
    """The `drive` handler now hard-refuses an unscheduled launch (SystemExit
    before any Phase A work) -- the console form must catch this at submit
    time, not let the operator's job crash after launch."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/drives/run",
        data={
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert "--publish" in resp.text and "--lf-day" in resp.text
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    assert jobs.list_jobs(store) == []


def test_drives_run_no_review_checkbox_opts_out_of_review(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))

    class _FakePopen:
        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: _FakePopen())

    resp = _client().post(
        "/drives/run",
        data={
            "brand": "gomoveshift",
            "directory": str(tmp_path / "day1"),
            "output_dir": str(tmp_path / "out"),
            "region": "usa",
            "dry_run": "on",
            "no_review": "on",
            "publish": "tue",
            "lf_day": "tue",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    job_id = int(resp.headers["location"].rsplit("/", 1)[-1])
    row = jobs.get_job(jobs.JobStore(tmp_path / "jobs.sqlite3"), job_id)
    assert "--no-review" in row["argv"]


def test_drives_gates_renders_pending_question(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    (output_dir / "_gates").mkdir(parents=True)
    (output_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps(
            {"question": "Played each LF for 2026-08-01?", "choices": ["approve", "reject"]}
        ),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get(
        "/drives/gates", params={"output_dir": str(output_dir), "date_key": "2026-08-01"}
    )
    assert resp.status_code == 200
    assert "Played each LF for 2026-08-01?" in resp.text
    resp = _client().get(
        "/drives/gates",
        params={"output_dir": str(tmp_path / "does-not-exist"), "date_key": "2026-08-01"},
    )
    assert resp.status_code == 404


def test_drives_gates_page_uses_plain_language_for_the_gate_title(tmp_path, monkeypatch):
    """'Gate C/D' names the pipeline phases it runs after, not what the
    operator is being asked. review_gate.py's gate_cd() is ONE confirmation
    ("Played each LF and reviewed all metadata...") gating every upload for
    the day -- not two separate gates -- so one plain title is honest here.
    The internal id is retained for diagnostics, not deleted."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    (output_dir / "_gates").mkdir(parents=True)
    (output_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps(
            {"question": "Played each LF for 2026-08-01?", "choices": ["approve", "reject"]}
        ),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get(
        "/drives/gates", params={"output_dir": str(output_dir), "date_key": "2026-08-01"}
    )
    assert resp.status_code == 200
    assert "Final content review" in resp.text
    assert "<h1>Gate C/D" not in resp.text
    # The internal gate id is retained for diagnostics.
    assert "gate_cd" in resp.text


def test_drives_gates_rejects_an_output_dir_this_console_never_launched(tmp_path, monkeypatch):
    """Only output_dirs this console's own job argvs used are answerable
    -- a foreign path (even a real, existing gate dir) is refused as 404, and
    the refusal body must not echo the submitted path back (it would turn the
    error page into a filesystem oracle)."""
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    foreign_dir = tmp_path / "foreign-tenant-out"
    (foreign_dir / "_gates").mkdir(parents=True)
    (foreign_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps({"question": "Secret question", "choices": ["approve", "reject"]}),
        encoding="utf-8",
    )
    # No job row references foreign_dir -- this console never launched a run there.
    resp = _client().get(
        "/drives/gates", params={"output_dir": str(foreign_dir), "date_key": "2026-08-01"}
    )
    assert resp.status_code == 404
    assert str(foreign_dir) not in resp.text
    assert "Secret question" not in resp.text


def test_drives_gates_accepts_a_dot_dot_path_that_resolves_to_an_allowed_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    (output_dir / "_gates").mkdir(parents=True)
    (output_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps({"question": "Q?", "choices": ["approve", "reject"]}), encoding="utf-8"
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    sneaky = str(output_dir / ".." / "out")
    resp = _client().get("/drives/gates", params={"output_dir": sneaky, "date_key": "2026-08-01"})
    assert resp.status_code == 200
    assert "Q?" in resp.text


def test_drives_gates_post_rejects_a_foreign_output_dir_without_writing_or_deleting(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    foreign_dir = tmp_path / "foreign-tenant-out"
    gates_dir = foreign_dir / "_gates"
    gates_dir.mkdir(parents=True)
    question_path = gates_dir / "gate_cd_2026-08-01.question.json"
    question_path.write_text(
        json.dumps({"question": "Q?", "choices": ["approve", "reject"]}), encoding="utf-8"
    )
    # No job row references foreign_dir.
    resp = _client().post(
        "/drives/gates",
        data={
            "output_dir": str(foreign_dir),
            "date_key": "2026-08-01",
            "decision": "approve",
            "note": "hostile",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 404
    assert str(foreign_dir) not in resp.text
    assert not (gates_dir / "gate_cd_2026-08-01.answer").exists()
    assert question_path.exists()


def test_drives_gates_get_rejects_a_traversing_date_key_even_for_an_allowed_output_dir(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get(
        "/drives/gates",
        params={"output_dir": str(output_dir), "date_key": "../../etc/passwd"},
    )
    assert resp.status_code == 404


def test_drives_gates_post_rejects_a_traversing_date_key_and_writes_nothing_outside_gates(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().post(
        "/drives/gates",
        data={
            "output_dir": str(output_dir),
            "date_key": "../escape",
            "decision": "approve",
            "note": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 404
    escaped = tmp_path / "escape.answer"
    assert not escaped.exists()
    assert not any(tmp_path.rglob("*escape*.answer"))


def test_collect_metadata_previews_reads_youtube_json_and_runs_gates(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "drive_lf01_youtube.json").write_text(
        json.dumps(
            {
                "title": "Sullivan County, TN Scenic Drive",
                "description": "A clean description with no repeats.",
                "location": "Sullivan County, Tennessee",
            }
        ),
        encoding="utf-8",
    )
    previews = drives.collect_metadata_previews(output_dir)
    assert len(previews) == 1
    preview = previews[0]
    assert preview["file"] == "drive_lf01_youtube.json"
    assert preview["title"] == "Sullivan County, TN Scenic Drive"
    assert preview["description"] == "A clean description with no repeats."
    gate_names = {g["gate"] for g in preview["gates"]}
    assert {"metadata_quality", "label_quality", "description_redundancy"} <= gate_names
    assert all(g["passed"] for g in preview["gates"])


def test_collect_metadata_previews_surfaces_a_failing_gate_message(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "drive_lf01_youtube.json").write_text(
        json.dumps(
            {
                "title": "drive through To Clonakilty",
                "description": "Some description",
                "location": "",
            }
        ),
        encoding="utf-8",
    )
    previews = drives.collect_metadata_previews(output_dir)
    failing = [g for g in previews[0]["gates"] if not g["passed"]]
    assert failing
    assert any("through To" in g["message"] or "title:" in g["message"] for g in failing)


def test_drives_preview_route_renders_title_and_gate_verdicts(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "drive_lf01_youtube.json").write_text(
        json.dumps(
            {
                "title": "Sullivan County, TN Scenic Drive",
                "description": "A clean description with no repeats.",
                "location": "Sullivan County, Tennessee",
            }
        ),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get("/drives/preview", params={"output_dir": str(output_dir)})
    assert resp.status_code == 200
    assert "Sullivan County, TN Scenic Drive" in resp.text
    assert "A clean description with no repeats." in resp.text
    assert "metadata_quality" in resp.text


def test_drives_preview_route_404s_on_missing_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    resp = _client().get("/drives/preview", params={"output_dir": str(tmp_path / "does-not-exist")})
    assert resp.status_code == 404


def test_drives_preview_route_404s_when_no_generated_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get("/drives/preview", params={"output_dir": str(output_dir)})
    assert resp.status_code == 404


def test_drives_preview_route_404s_and_does_not_echo_a_foreign_output_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    foreign_dir = tmp_path / "foreign-tenant-out"
    foreign_dir.mkdir()
    (foreign_dir / "drive_lf01_youtube.json").write_text(
        json.dumps({"title": "Secret Title", "description": "d", "location": "l"}),
        encoding="utf-8",
    )
    # No job row references foreign_dir.
    resp = _client().get("/drives/preview", params={"output_dir": str(foreign_dir)})
    assert resp.status_code == 404
    assert str(foreign_dir) not in resp.text
    assert "Secret Title" not in resp.text


def test_output_dirs_from_job_argvs_dedupes_and_orders():
    argvs = [
        ["gms", "drive", "d1", "--output-dir", "/out/a", "--no-review"],
        ["gms", "drive", "d2", "--output-dir", "/out/b"],
        ["gms", "drive", "d3", "--output-dir", "/out/a"],
        ["gms", "status"],
    ]
    assert drives.output_dirs_from_job_argvs(argvs) == ["/out/a", "/out/b"]


def test_output_dirs_from_job_argvs_collects_the_parent_of_a_film_output_file():
    argvs = [
        [
            "gms",
            "route-story-film",
            "--profile",
            "trip.yaml",
        ],
        ["gms", "-m", "gomoveshift", "route-story-film", "--output", "/out/film/trip.mp4"],
        ["gms", "drive", "d1", "--output-dir", "/out/a"],
    ]
    assert drives.output_dirs_from_job_argvs(argvs) == ["/out/film", "/out/a"]


def test_list_pending_gates_globs_question_files_across_output_dirs(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    (out_a / "_gates").mkdir(parents=True)
    (out_b / "_gates").mkdir(parents=True)
    (out_a / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps({"question": "Played each LF for 2026-08-01?"})
    )
    (out_b / "_gates" / "gate_cd_2026-08-02.question.json").write_text(
        json.dumps({"question": "Played each LF for 2026-08-02?"})
    )

    pending = drives.list_pending_gates([str(out_a), str(out_b), str(tmp_path / "missing")])
    assert {(p["output_dir"], p["date_key"]) for p in pending} == {
        (str(out_a), "2026-08-01"),
        (str(out_b), "2026-08-02"),
    }


def test_list_pending_gates_finds_project_subdir_and_non_gate_cd_ids(tmp_path):
    # _consume_or_ask writes one level below --output-dir (the project dir);
    # non-gate_cd_ ids (gate_a, gate_preview, ...) must be discoverable too.
    output_dir = tmp_path / "out"
    project_dir = output_dir / "day1"
    (project_dir / "_gates").mkdir(parents=True)
    (project_dir / "_gates" / "gate_a.question.json").write_text(
        json.dumps({"question": "Episode breakdown correct and OST text exact?"})
    )

    pending = drives.list_pending_gates([str(output_dir)])

    assert len(pending) == 1
    row = pending[0]
    assert row["output_dir"] == str(project_dir)
    assert row["gate_id"] == "gate_a"
    assert row["date_key"] is None
    assert row["question"] == "Episode breakdown correct and OST text exact?"
    assert "mtime" in row


def test_drives_gates_pending_route_lists_and_links(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    output_dir = tmp_path / "out"
    (output_dir / "_gates").mkdir(parents=True)
    (output_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps({"question": "Played each LF for 2026-08-01?"})
    )
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get("/drives/gates/pending")
    assert resp.status_code == 200
    assert "Played each LF for 2026-08-01?" in resp.text
    assert f"output_dir={output_dir}&date_key=2026-08-01" in resp.text


def test_drives_gates_post_writes_the_answer_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().post(
        "/drives/gates",
        data={
            "output_dir": str(output_dir),
            "date_key": "2026-08-01",
            "decision": "approve",
            "note": "looks good",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    payload = json.loads((output_dir / "_gates" / "gate_cd_2026-08-01.answer").read_text())
    assert payload["decision"] == "approve"
    assert payload["note"] == "looks good"
    assert payload["answered_at"]


def test_write_gate_answer_retires_the_question_file(tmp_path):
    """An answered gate must drop off the pending list -- the pipeline's
    poll only ever looks at the ``.answer`` path, so retiring the question
    file is safe and makes ``list_pending_gates`` exclude it naturally."""
    output_dir = tmp_path / "out"
    gates_dir = output_dir / "_gates"
    gates_dir.mkdir(parents=True)
    question_path = gates_dir / "gate_cd_2026-08-01.question.json"
    question_path.write_text(
        json.dumps({"question": "Played each LF?", "choices": ["approve", "reject"]})
    )

    drives.write_gate_answer(output_dir, "2026-08-01", "approve", "looks good")

    assert not question_path.exists()
    assert (gates_dir / "gate_cd_2026-08-01.question.answered").exists()
    assert drives.list_pending_gates([str(output_dir)]) == []


def test_write_gate_answer_rejects_a_decision_outside_the_question_choices(tmp_path):
    output_dir = tmp_path / "out"
    gates_dir = output_dir / "_gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps({"question": "Played each LF?", "choices": ["approve", "reject"]})
    )

    import pytest

    with pytest.raises(ValueError):
        drives.write_gate_answer(output_dir, "2026-08-01", "maybe", "")

    assert not (gates_dir / "gate_cd_2026-08-01.answer").exists()


def test_drives_gates_post_with_invalid_decision_returns_400(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    gates_dir = output_dir / "_gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps({"question": "Played each LF?", "choices": ["approve", "reject"]})
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().post(
        "/drives/gates",
        data={
            "output_dir": str(output_dir),
            "date_key": "2026-08-01",
            "decision": "maybe",
            "note": "",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 400
    assert not (gates_dir / "gate_cd_2026-08-01.answer").exists()


# ---------------------------------------------------------------------------
# The gate page shows the footage it is asking the operator to approve.
# ---------------------------------------------------------------------------


def _setup_gate_with_video(tmp_path, monkeypatch, *, artifacts=None, title="Sample Drive"):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    video = output_dir / "lf_01.mp4"
    _write_tiny_video(video)
    metadata = output_dir / "day_lf01_youtube.json"
    metadata.write_text(json.dumps({"title": title}), encoding="utf-8")
    (output_dir / "_gates").mkdir(parents=True)
    if artifacts is None:
        artifacts = [str(video), str(metadata)]
    (output_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps(
            {
                "question": "Played each LF for 2026-08-01?",
                "choices": ["approve", "reject"],
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")
    return output_dir, video, metadata


def test_gate_video_paths_pairs_only_supported_video_suffixes_inside_the_output_dir(tmp_path):
    output_dir = tmp_path / "out"
    video = output_dir / "lf_01.mp4"
    _write_tiny_video(video)
    other = output_dir / "notes.txt"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("not a video")
    outside = tmp_path / "outside" / "secret.mp4"
    _write_tiny_video(outside)
    question = {"artifacts": [str(video), str(other), str(outside)]}

    paths = drives.gate_video_paths(output_dir, question)

    assert paths == [video.resolve()]


def test_drives_gates_renders_a_frame_reference_and_title_for_each_covered_video(
    tmp_path, monkeypatch
):
    output_dir, _video, _metadata = _setup_gate_with_video(tmp_path, monkeypatch)

    resp = _client().get(
        "/drives/gates", params={"output_dir": str(output_dir), "date_key": "2026-08-01"}
    )
    assert resp.status_code == 200
    assert f"/drives/gates/frame?output_dir={output_dir}&date_key=2026-08-01&index=0" in resp.text
    assert "Sample Drive" in resp.text


def test_drives_gates_frame_serves_real_jpeg_bytes_for_a_covered_video(tmp_path, monkeypatch):
    output_dir, _video, _metadata = _setup_gate_with_video(tmp_path, monkeypatch)

    resp = _client().get(
        "/drives/gates/frame",
        params={"output_dir": str(output_dir), "date_key": "2026-08-01", "index": 0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content[:2] == b"\xff\xd8"


def test_drives_gates_frame_rejects_an_output_dir_this_console_never_launched(
    tmp_path, monkeypatch
):
    output_dir, _video, _metadata = _setup_gate_with_video(tmp_path, monkeypatch)
    foreign_dir = tmp_path / "foreign"
    foreign_video = foreign_dir / "lf_01.mp4"
    _write_tiny_video(foreign_video)
    (foreign_dir / "_gates").mkdir(parents=True)
    (foreign_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps(
            {
                "question": "Q?",
                "choices": ["approve", "reject"],
                "artifacts": [str(foreign_video)],
            }
        ),
        encoding="utf-8",
    )

    resp = _client().get(
        "/drives/gates/frame",
        params={"output_dir": str(foreign_dir), "date_key": "2026-08-01", "index": 0},
    )
    assert resp.status_code == 404


def test_drives_gates_frame_refuses_a_traversal_artifact(tmp_path, monkeypatch):
    """The artifact list itself may not be trusted verbatim: an entry whose
    resolved path escapes the gate's own output_dir (a literal ``..``
    segment here) must never be reachable through the frame route."""
    outside = tmp_path / "outside" / "secret.mp4"
    _write_tiny_video(outside)
    traversing = str(tmp_path / "out" / ".." / "outside" / "secret.mp4")
    output_dir, _video, _metadata = _setup_gate_with_video(
        tmp_path, monkeypatch, artifacts=[traversing]
    )

    resp = _client().get(
        "/drives/gates/frame",
        params={"output_dir": str(output_dir), "date_key": "2026-08-01", "index": 0},
    )
    assert resp.status_code == 404


def test_drives_gates_shows_an_honest_could_not_read_state_for_a_missing_video(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    missing_video = output_dir / "lf_01.mp4"
    (output_dir / "_gates").mkdir(parents=True)
    (output_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps(
            {
                "question": "Played each LF for 2026-08-01?",
                "choices": ["approve", "reject"],
                "artifacts": [str(missing_video)],
            }
        ),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get(
        "/drives/gates", params={"output_dir": str(output_dir), "date_key": "2026-08-01"}
    )
    assert resp.status_code == 200
    assert "could not be read" in resp.text
    assert "lf_01.mp4" in resp.text
    assert "/drives/gates/frame" not in resp.text
    assert "<img" not in resp.text


def test_drives_gates_frame_refuses_a_non_video_suffix(tmp_path, monkeypatch):
    text_file = None

    def _artifacts(output_dir):
        nonlocal text_file
        text_file = output_dir / "notes.txt"
        text_file.write_text("not a video")
        return [str(text_file)]

    monkeypatch.setenv("GMS_CONSOLE_DATA_DIR", str(tmp_path))
    output_dir = tmp_path / "out"
    (output_dir / "_gates").mkdir(parents=True)
    artifacts = _artifacts(output_dir)
    (output_dir / "_gates" / "gate_cd_2026-08-01.question.json").write_text(
        json.dumps({"question": "Q?", "choices": ["approve", "reject"], "artifacts": artifacts}),
        encoding="utf-8",
    )
    store = jobs.JobStore(tmp_path / "jobs.sqlite3")
    store.create("drive", ["gms", "drive", "d", "--output-dir", str(output_dir)], log_path="")

    resp = _client().get(
        "/drives/gates/frame",
        params={"output_dir": str(output_dir), "date_key": "2026-08-01", "index": 0},
    )
    assert resp.status_code == 404
