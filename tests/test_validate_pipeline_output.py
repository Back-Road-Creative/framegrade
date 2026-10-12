"""Tests for ``tools/validate_pipeline_output.py``.

Fixtures build a synthetic pipeline output tree under ``tmp_path`` so the
validator has real files to probe.  Duration bounds are shrunk via CLI flags
to make the 30 s synthetic LFs pass the duration checks.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from _state_helpers import make_state

# Importing the validator as a module: add the repo root so ``tools`` is found.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import validate_pipeline_output as v  # noqa: E402

FFMPEG = "/usr/local/bin/ffmpeg"

# Shared CLI arguments so every test probes real synthetic 30 s files.
DEFAULT_FLAGS = [
    "--min-lf-duration",
    "25",
    "--max-lf-duration",
    "35",
]

PROJECT = "tp"


# ---------------------------------------------------------------------------
# Cheap synthetic media helpers
# ---------------------------------------------------------------------------


def _make_lf(path: Path, duration: int = 30) -> None:
    """Encode a ~30 s hevc+aac mp4 at 64x36. Takes <1 s."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x36:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=8000:cl=mono:d={duration}",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "log-level=error",
            "-crf",
            "32",
            "-c:a",
            "aac",
            "-b:a",
            "16k",
            "-t",
            str(duration),
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _make_drift_lf(path: Path, v_dur: int, a_dur: int) -> None:
    """Mux a video-only track and audio-only track with mismatched lengths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    video = path.with_suffix(".vo.mp4")
    audio = path.with_suffix(".ao.m4a")
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x36:d={v_dur}",
            "-c:v",
            "libx265",
            "-preset",
            "ultrafast",
            "-x265-params",
            "log-level=error",
            "-crf",
            "32",
            "-t",
            str(v_dur),
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=8000:cl=mono:d={a_dur}",
            "-c:a",
            "aac",
            "-b:a",
            "16k",
            "-t",
            str(a_dur),
            str(audio),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-c", "copy", str(path)],
        check=True,
        capture_output=True,
    )
    video.unlink(missing_ok=True)
    audio.unlink(missing_ok=True)


def _make_video_only_sl(path: Path, duration: int = 20) -> None:  # no audio stream
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s=64x36:d={duration}",
        "-c:v", "libx265", "-preset", "ultrafast", "-t", str(duration), str(path),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Scaffolding: build a valid output tree and snapshot
# ---------------------------------------------------------------------------


def _fill_size_marker(path: Path) -> None:
    """Grow a file so it clears the 100 MB minimum size check.

    Sparse: the validator reads ``st_size``, so a truncate gives the same
    number without writing 100 MB per test into pytest's basetemp.
    """
    min_size = 100 * 1024 * 1024 + 1024
    if path.stat().st_size >= min_size:
        return
    os.truncate(path, min_size)


def _copy_sparse(src: Path, dst: Path) -> None:
    """Copy ``src`` keeping its holes; ``shutil.copyfile`` writes them as real zeros."""
    with src.open("rb") as s, dst.open("wb") as d:
        while chunk := s.read(1 << 20):
            if chunk.count(0) == len(chunk):
                d.seek(len(chunk), os.SEEK_CUR)
            else:
                d.write(chunk)
        d.truncate(s.tell())


def _build_valid_output(tmp_path: Path, lf_count: int = 2) -> dict:
    """Create a passing output tree. Returns a dict of paths."""
    output_base = tmp_path / "out"
    project_dir = output_base / PROJECT
    long_forms = project_dir / "long_forms"
    thumbnails = project_dir / "thumbnails"
    shorts = project_dir / "shorts"
    state_dir = tmp_path / "state"
    snapshot_dir = tmp_path / "snap"

    for d in (long_forms, thumbnails, shorts, state_dir, snapshot_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_start = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
    run_start_ts = run_start.timestamp()

    # Snapshot dir mtime is the run-start anchor.
    import os

    os.utime(snapshot_dir, (run_start_ts, run_start_ts))

    lfs: list[Path] = []
    for i in range(1, lf_count + 1):
        lf = long_forms / f"lf_{i:02d}.mp4"
        _make_lf(lf)
        _fill_size_marker(lf)
        lfs.append(lf)

        # Recording-date content-fidelity check (#15) reads youtube.json.
        _DESC = (
            "Drive log " + str(i) + ". Scenic route through the test region with "
            "a couple of stops. End-card and CTAs follow.\n\n"
            "📹 Recorded: 2026-04-01\n\n"
            "#test #drive"
        )
        (project_dir / f"{PROJECT}_lf{i:02d}_youtube.json").write_text(
            json.dumps({"title": f"lf{i:02d}", "description": _DESC})
        )
        (thumbnails / f"lf_{i:02d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        shorts_dir = shorts / f"lf_{i:02d}"
        shorts_dir.mkdir(parents=True, exist_ok=True)
        for s in range(2):
            (shorts_dir / f"short_{s:02d}.mp4").write_bytes(b"dummy")
        (shorts_dir / "shorts_rendered.json").write_text(json.dumps({"num_rendered": 2}))

    # Schedule: always at least two slots spanning 7 days, so the schedule
    # check is orthogonal to the LF count in every other test case.
    base_date = datetime(2026, 4, 1, tzinfo=timezone.utc)
    slot_count = max(lf_count, 2)
    lf_slots = [
        {"lf_index": i, "date": (base_date + timedelta(days=(i - 1) * 7)).isoformat()}
        for i in range(1, slot_count + 1)
    ]
    (state_dir / "pipeline_state.json").write_text(
        json.dumps({"drive_schedule": {"lf_slots": lf_slots}})
    )

    # youtube_uploads.json: live and snapshot identical.
    payload = json.dumps({"uploads": []})
    (project_dir / "youtube_uploads.json").write_text(payload)
    (snapshot_dir / "youtube_uploads.json").write_text(payload)

    # Bump every file's mtime safely into the "after run_start" window.
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    for lf in lfs:
        os.utime(lf, (now_ts, now_ts))
    for f in project_dir.glob(f"{PROJECT}_lf*_youtube.json"):
        os.utime(f, (now_ts, now_ts))

    return {
        "output_base": output_base,
        "project_dir": project_dir,
        "long_forms": long_forms,
        "thumbnails": thumbnails,
        "shorts": shorts,
        "state_dir": state_dir,
        "snapshot_dir": snapshot_dir,
        "run_start": run_start,
        "lfs": lfs,
    }


def _cli_args(paths: dict, extra: list[str] | None = None) -> list[str]:
    args = [
        "--project",
        PROJECT,
        "--output-base",
        str(paths["output_base"]),
        "--state-dir",
        str(paths["state_dir"]),
        "--snapshot-dir",
        str(paths["snapshot_dir"]),
        "--run-start",
        paths["run_start"].isoformat(),
        *DEFAULT_FLAGS,
    ]
    if extra:
        args.extend(extra)
    return args


def _parse_and_run(paths: dict, extra: list[str] | None = None):
    parser = v.build_parser()
    namespace = parser.parse_args(_cli_args(paths, extra))
    return v.run_checks(namespace)


def _result_by_number(results, number: int):
    for r in results:
        if r.number == number:
            return r
    raise AssertionError(f"no check numbered {number}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_pass(tmp_path: Path):
    paths = _build_valid_output(tmp_path, lf_count=2)
    results = _parse_and_run(paths)
    failures = [r for r in results if not r.passed]
    assert failures == [], [f"{r.number} {r.name}: {r.detail}" for r in failures]


def test_fail_lf_count(tmp_path: Path):
    paths = _build_valid_output(tmp_path, lf_count=2)
    # Nuke the LFs so the count drops to 0.
    for lf in paths["lfs"]:
        lf.unlink()
    results = _parse_and_run(paths, extra=["--expected-lf-min", "1"])
    assert _result_by_number(results, 1).passed is False


def test_fail_av_drift(tmp_path: Path):
    paths = _build_valid_output(tmp_path, lf_count=2)
    # Overwrite lf_01 with a drift-injected mux (30 s video, 25 s audio ~= 16.7%).
    bad = paths["long_forms"] / "lf_01.mp4"
    bad.unlink()
    _make_drift_lf(bad, v_dur=30, a_dur=25)
    _fill_size_marker(bad)
    results = _parse_and_run(paths)
    drift = _result_by_number(results, 4)
    assert drift.passed is False
    assert "lf_01" in drift.detail


def test_fail_stale_youtube_json(tmp_path: Path):
    """Stale youtube.json sidecar → check #8 FAIL."""
    import os

    paths = _build_valid_output(tmp_path, lf_count=1)
    stale_ts = (paths["run_start"] - timedelta(hours=1)).timestamp()
    yt = paths["project_dir"] / f"{PROJECT}_lf01_youtube.json"
    os.utime(yt, (stale_ts, stale_ts))
    results = _parse_and_run(paths)
    youtube_fresh = _result_by_number(results, 8)
    assert youtube_fresh.passed is False


def test_fail_ghost_metadata(tmp_path: Path):
    """Ghost youtube.json (out-of-range LF index) → the ghost-metadata check FAILs.

    The ghost-metadata regex only fires on ghost ``*_lf99_youtube.json``
    sidecars.
    """
    paths = _build_valid_output(tmp_path, lf_count=2)
    ghost = paths["project_dir"] / f"{PROJECT}_lf99_youtube.json"
    ghost.write_text(json.dumps({"title": "ghost", "description": ""}))
    results = _parse_and_run(paths)
    ghost_check = _result_by_number(results, 10)
    assert ghost_check.passed is False
    assert "lf99" in ghost_check.detail


def test_fail_shorts_count_mismatch(tmp_path: Path):
    paths = _build_valid_output(tmp_path, lf_count=1)
    lf01_shorts = paths["shorts"] / "lf_01"
    # Add 3 more mp4s → total 5 on disk, but JSON still says num_rendered=2.
    for s in range(2, 5):
        (lf01_shorts / f"short_{s:02d}.mp4").write_bytes(b"dummy")
    results = _parse_and_run(paths)
    shorts = _result_by_number(results, 11)
    assert shorts.passed is False
    assert "num_rendered=2" in shorts.detail


def test_fail_youtube_changed(tmp_path: Path):
    paths = _build_valid_output(tmp_path, lf_count=1)
    live = paths["project_dir"] / "youtube_uploads.json"
    live.write_text(json.dumps({"uploads": [{"lf": 1, "video_id": "abc"}]}))
    results = _parse_and_run(paths)
    y = _result_by_number(results, 13)
    assert y.passed is False
    assert "sha256 mismatch" in y.detail


def test_cli_exits_0_on_all_pass(tmp_path: Path):
    paths = _build_valid_output(tmp_path, lf_count=1)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "validate_pipeline_output.py"),
            *_cli_args(paths),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Summary:" in proc.stdout


def test_cli_exits_1_on_any_fail(tmp_path: Path):
    paths = _build_valid_output(tmp_path, lf_count=1)
    # Break the youtube_uploads snapshot match.
    live = paths["project_dir"] / "youtube_uploads.json"
    live.write_text(json.dumps({"mutated": True}))
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "validate_pipeline_output.py"),
            *_cli_args(paths),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL" in proc.stdout


def test_canned_probe_via_monkeypatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """ffprobe monkeypatch path: confirms the pure-check logic without ffmpeg."""
    paths = _build_valid_output(tmp_path, lf_count=1)
    canned = {
        "format": {"duration": "30.0"},
        "streams": [
            {"codec_type": "video", "codec_name": "hevc", "duration": "30.0"},
            {"codec_type": "audio", "codec_name": "aac", "duration": "30.0"},
        ],
    }
    monkeypatch.setattr(v, "ffprobe", lambda _path: canned)
    results = _parse_and_run(paths)
    failures = [r for r in results if not r.passed]
    assert failures == [], [f"{r.number} {r.name}: {r.detail}" for r in failures]


def test_stale_metadata_fails_off_mount_and_missing_file_fails_under_mount(tmp_path: Path):
    """A 30 s-stale youtube.json fails off a drive mount; a missing one fails under it."""
    import os

    paths = _build_valid_output(tmp_path, lf_count=1)
    # Set mtime to 30 s before run_start; off a mount path there is no tolerance, so this must fail
    stale_ts = (paths["run_start"] - timedelta(seconds=30)).timestamp()
    yt = paths["project_dir"] / f"{PROJECT}_lf01_youtube.json"
    os.utime(yt, (stale_ts, stale_ts))

    # Without /mnt/ prefix, 30 s stale should fail
    results = _parse_and_run(paths)
    youtube_fresh = _result_by_number(results, 8)
    assert youtube_fresh.passed is False

    # Under a /mnt/ path a plainly missing youtube.json still fails (exercises
    # the mount-path branch without a real drvfs filesystem).
    run_start = paths["run_start"]
    result = v.check_youtube_json_fresh(
        Path("/mnt/d/test_does_not_exist"), [(1, Path("/tmp/unused.mp4"))], PROJECT, run_start
    )
    assert result.passed is False
    assert "missing" in result.detail


# ---------------------------------------------------------------------------
# Mode-aware validation (skip-upload audit vs post-upload)
#
# Under ``--skip-upload``, Phase D intentionally does not copy LFs into the
# delivery-path layout (``<output>/<proj>/long_forms/*``). The
# delivery-path-specific checks in ``run_validation`` therefore fail every
# time under skip-upload, drowning real failures in noise. These tests pin
# the two-mode split: ``skip_upload_audit`` vs ``post_upload``.
# ---------------------------------------------------------------------------


def _build_skip_upload_tree(tmp_path: Path, lf_count: int = 2) -> dict:
    """Build a cache-side layout that matches what ``--skip-upload`` leaves on
    disk: ``<stages>/day_<date>/regular_lfs/lf_NN.mp4`` + per-LF metadata
    JSON/CSV under ``project_dir``.
    """
    import os

    output_base = tmp_path / "out"
    project_dir = output_base / PROJECT
    stages_dir = tmp_path / "cache" / PROJECT / "_stages"
    snapshot_dir = tmp_path / "snap"

    for d in (project_dir, stages_dir, snapshot_dir):
        d.mkdir(parents=True, exist_ok=True)

    run_start = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
    run_start_ts = run_start.timestamp()
    os.utime(snapshot_dir, (run_start_ts, run_start_ts))

    # Cache-side LF layout: one LF per day for simplicity.
    base_date = datetime(2026, 4, 1)
    day_dirs = []
    lfs = []
    for i in range(1, lf_count + 1):
        day = (base_date + timedelta(days=i - 1)).strftime("%Y-%m-%d")
        day_dir = stages_dir / f"day_{day}"
        lf_subdir = day_dir / "regular_lfs"
        lf_subdir.mkdir(parents=True, exist_ok=True)
        lf_path = lf_subdir / f"lf_{i:02d}.mp4"
        _make_lf(lf_path)
        _fill_size_marker(lf_path)
        day_dirs.append(day_dir)
        lfs.append((i, lf_path))

        # Metadata sidecars still live under project_dir (Phase D writes
        # these BEFORE copy-out; they exist regardless of skip-upload).
        # Recording-date content-fidelity check (#15) reads youtube.json.
        _DESC = (
            "Drive log " + str(i) + ". Scenic route through the test region with "
            "a couple of stops. End-card and CTAs follow.\n\n"
            "📹 Recorded: 2026-04-01\n\n"
            "#test #drive"
        )
        (project_dir / f"{PROJECT}_lf{i:02d}_youtube.json").write_text(
            json.dumps({"title": f"lf{i:02d}", "description": _DESC})
        )

    # Schedule lives on pipeline_state.json at stages_dir root.
    slot_count = max(lf_count, 2)
    lf_slots = [
        {
            "lf_index": i,
            "date": (
                datetime(2026, 4, 1, tzinfo=timezone.utc) + timedelta(days=(i - 1) * 7)
            ).isoformat(),
        }
        for i in range(1, slot_count + 1)
    ]
    (stages_dir / "pipeline_state.json").write_text(
        json.dumps({"drive_schedule": {"lf_slots": lf_slots}, "skip_upload": True})
    )

    # Bump metadata mtime past run_start.
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    for f in project_dir.glob(f"{PROJECT}_lf*_youtube.json"):
        os.utime(f, (now_ts, now_ts))

    return {
        "output_base": output_base,
        "project_dir": project_dir,
        "stages_dir": stages_dir,
        "snapshot_dir": snapshot_dir,
        "run_start": run_start,
        "day_dirs": day_dirs,
        "lfs": lfs,
    }


def test_mode_detected_from_state_skip_upload_true(tmp_path: Path):
    """state.skip_upload=True → derive_mode_from_state → skip_upload_audit."""
    state = make_state(skip_upload=True, youtube_upload=False)
    assert v.derive_mode_from_state(state) == v.Mode.SKIP_UPLOAD_AUDIT


def test_mode_detected_from_state_youtube_upload_true_skip_upload_false(tmp_path: Path):
    """state.youtube_upload=True + skip_upload=False → post_upload."""
    state = make_state(skip_upload=False, youtube_upload=True)
    assert v.derive_mode_from_state(state) == v.Mode.POST_UPLOAD


def test_mode_detected_from_state_both_missing_defaults_post_upload():
    """Neither flag set → default to post_upload (existing behaviour)."""
    state: dict = {}
    assert v.derive_mode_from_state(state) == v.Mode.POST_UPLOAD


def test_mode_detected_from_state_none_defaults_post_upload():
    """state=None → default to post_upload (no state available, preserve
    existing caller contract)."""
    assert v.derive_mode_from_state(None) == v.Mode.POST_UPLOAD


def test_skip_upload_mode_checks_cache_side_lfs(tmp_path: Path):
    """skip_upload_audit mode discovers LFs under stages_dir/day_*/regular_lfs
    and DOES NOT require them under project_dir/long_forms."""
    paths = _build_skip_upload_tree(tmp_path, lf_count=2)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["stages_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
    )
    # All skip-upload-mode checks must pass against this layout.
    failures = [r for r in report.checks if not r.passed]
    assert failures == [], [f"{r.number} {r.name}: {r.detail}" for r in failures]


def test_skip_upload_mode_passes_when_outputs_exist(tmp_path: Path):
    """End-to-end skip-upload audit: run_validation returns 'audit passed'
    label on the report + zero failures."""
    paths = _build_skip_upload_tree(tmp_path, lf_count=2)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["stages_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
    )
    assert report.failed_count == 0
    table = report.format_table()
    assert "audit passed (upload not attempted)" in table.lower()


def test_skip_upload_mode_fails_when_cache_lfs_missing(tmp_path: Path):
    """If the cache-side LFs are absent, the audit must still fail — that's
    the whole point of the skip-upload mode (catch bad encode runs before
    they hit upload)."""
    paths = _build_skip_upload_tree(tmp_path, lf_count=1)
    # Delete the one cache LF.
    for _, lf_path in paths["lfs"]:
        lf_path.unlink()
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["stages_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
    )
    assert report.failed_count > 0


def test_skip_upload_mode_skips_delivery_path_checks(tmp_path: Path):
    """skip_upload_audit mode drops the delivery-path-only checks. #09
    (thumbnails) and #13 (youtube_uploads_unchanged vs snapshot) must not
    appear in the report — they cannot be evaluated pre-upload; #01
    (lf_count) still runs against the cache-side LFs."""
    paths = _build_skip_upload_tree(tmp_path, lf_count=1)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["stages_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
    )
    numbers = {r.number for r in report.checks}
    # #09 (thumbnails under long_forms) and #13 (youtube_uploads_unchanged)
    # are skipped pre-upload.  #01 (lf_count under long_forms) is replaced
    # with a cache-side equivalent, so the CHECK itself may keep the same
    # number but must not require long_forms/.
    assert 9 not in numbers
    assert 13 not in numbers


def test_post_upload_mode_fails_when_long_forms_missing(tmp_path: Path):
    """Regression for current behaviour under live upload — post_upload
    mode still runs all 15 checks and flags the missing long_forms/ dir."""
    paths = _build_skip_upload_tree(tmp_path, lf_count=1)
    # No long_forms dir was built. In post_upload mode this MUST fail.
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["stages_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.POST_UPLOAD,
    )
    # 12 surviving original checks (#1-6, #8-13) + content-fidelity #15
    # + #19 stop_motion + #20 places_verified = 15 checks (#7/#14/#16 retired;
    # #17/#18 removed with the shorts CSV Drive-link checks); expect
    # several failures.
    assert len(report.checks) == 15
    assert report.failed_count >= 3


def test_post_upload_mode_passes_when_everything_present(tmp_path: Path):
    """Happy-path regression: with a full delivery tree, post_upload mode
    returns the 12 surviving original checks + content-fidelity check #15
    + #19 stop_motion (no-op PASS, no persisted stop mask)
    + #20 places_verified (no state → skipped PASS) = 15 as PASS
    (#7/#14/#16 retired; #17/#18 removed with the shorts CSV Drive-link
    checks)."""
    paths = _build_valid_output(tmp_path, lf_count=2)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.POST_UPLOAD,
    )
    assert len(report.checks) == 15
    failures = [r for r in report.checks if not r.passed]
    assert failures == [], [f"{r.number} {r.name}: {r.detail}" for r in failures]


def test_auto_mode_picks_skip_upload_when_state_says_so(tmp_path: Path):
    """Mode=AUTO + state.skip_upload=True → skip_upload_audit path."""
    paths = _build_skip_upload_tree(tmp_path, lf_count=1)
    state = make_state(skip_upload=True)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["stages_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.AUTO,
        state=state,
    )
    assert report.failed_count == 0


def test_auto_mode_defaults_to_post_upload_when_no_state(tmp_path: Path):
    """Mode=AUTO and no state → post_upload (preserves existing default
    behaviour so old callers are unaffected)."""
    paths = _build_valid_output(tmp_path, lf_count=1)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.AUTO,
    )
    # Delivery tree is full → post_upload path returns the 12 surviving
    # original checks (#1-6, #8-13) + content-fidelity #15 + #19 + #20 = 15 checks
    # (#7, #14, #16, #17, #18 retired/removed).
    assert len(report.checks) == 15


def test_cli_mode_flag_forces_skip_upload(tmp_path: Path):
    """CLI ``--mode skip_upload_audit`` overrides auto-detection."""
    paths = _build_skip_upload_tree(tmp_path, lf_count=1)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "validate_pipeline_output.py"),
            "--project",
            PROJECT,
            "--output-base",
            str(paths["output_base"]),
            "--state-dir",
            str(paths["stages_dir"]),
            "--snapshot-dir",
            str(paths["snapshot_dir"]),
            "--run-start",
            paths["run_start"].isoformat(),
            "--mode",
            "skip_upload_audit",
            "--min-lf-duration",
            "25",
            "--max-lf-duration",
            "35",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "audit passed (upload not attempted)" in proc.stdout.lower()


# ---------------------------------------------------------------------------
# check_schedule_length — scope-aware (1-LF 1-week smoke must pass)
# ---------------------------------------------------------------------------


def _write_state(
    state_dir: Path,
    lf_dates: list[str],
    shorts_dates: list[str] | None = None,
    identity_times: list[str] | None = None,
) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    drive_schedule: dict = {
        "lf_slots": [{"lf_index": i, "date": d} for i, d in enumerate(lf_dates, 1)],
    }
    if shorts_dates is not None:
        drive_schedule["shorts_slots"] = [
            {"short_index": i, "date": d} for i, d in enumerate(shorts_dates, 1)
        ]
    if identity_times is not None:
        drive_schedule["identity_times"] = identity_times
    (state_dir / "pipeline_state.json").write_text(json.dumps({"drive_schedule": drive_schedule}))


def test_check_schedule_length_accepts_one_lf_one_week_smoke(tmp_path: Path):
    """Regression: 1-day smoke produces 1 LF + 14 shorts spanning a week.

    Previous hard-coded ``len(dates) < 2`` rejected this valid schedule.
    """
    state_dir = tmp_path / "stages"
    shorts = []
    for day_offset in range(7):
        d = (datetime(2026, 4, 24, tzinfo=timezone.utc) + timedelta(days=day_offset)).isoformat()
        shorts.append(d)
        shorts.append(d)
    _write_state(state_dir, lf_dates=["2026-04-24T11:00:00+00:00"], shorts_dates=shorts)

    result = v.check_schedule_length(state_dir, expected_weeks=1, expected_lf_min=1)
    assert result.passed, result.detail


def test_check_schedule_length_fails_when_lf_slots_below_expected_lf_min(tmp_path: Path):
    """Scheduler under-produced LF slots for a multi-week run."""
    state_dir = tmp_path / "stages"
    _write_state(state_dir, lf_dates=["2026-04-01T11:00:00+00:00"])

    result = v.check_schedule_length(state_dir, expected_weeks=4, expected_lf_min=4)
    assert not result.passed
    assert "1" in result.detail and "4" in result.detail


def test_check_schedule_length_span_uses_combined_lf_and_shorts(tmp_path: Path):
    """Week-long span is satisfied by shorts even when LFs share one date."""
    state_dir = tmp_path / "stages"
    base = datetime(2026, 4, 24, tzinfo=timezone.utc)
    shorts = [(base + timedelta(days=i)).isoformat() for i in range(7)]
    _write_state(state_dir, lf_dates=[base.isoformat()], shorts_dates=shorts)

    result = v.check_schedule_length(state_dir, expected_weeks=1, expected_lf_min=1)
    assert result.passed, result.detail


def test_check_schedule_length_preserves_multi_week_span_enforcement(tmp_path: Path):
    """Multi-week run with too-short span still fails."""
    state_dir = tmp_path / "stages"
    lf_dates = [
        "2026-04-01T11:00:00+00:00",
        "2026-04-03T11:00:00+00:00",
    ]
    _write_state(state_dir, lf_dates=lf_dates, shorts_dates=lf_dates)

    result = v.check_schedule_length(state_dir, expected_weeks=4, expected_lf_min=2)
    assert not result.passed
    assert "span" in result.detail.lower()


def test_check_schedule_length_interim_cadence_one_lf_two_shorts(tmp_path: Path):
    """Interim cadence (--shorts-target 2): 1 LF Sun + shorts Mon, Tue spans
    2d. That is the whole schedule, not a truncated week, so it must not
    fail with 'span 2d < 6d'."""
    state_dir = tmp_path / "stages"
    _write_state(
        state_dir,
        lf_dates=["2030-09-22"],
        shorts_dates=["2030-09-23", "2030-09-24"],  # real state: date-only
    )

    result = v.check_schedule_length(state_dir, expected_weeks=1, expected_lf_min=1)
    assert result.passed, result.detail


def test_check_schedule_length_interim_cadence_collapsed_still_fails(tmp_path: Path):
    """Same 3 slots all booked on one day is still a broken schedule."""
    state_dir = tmp_path / "stages"
    _write_state(
        state_dir,
        lf_dates=["2030-09-22"],
        shorts_dates=["2030-09-22", "2030-09-22"],
    )

    result = v.check_schedule_length(state_dir, expected_weeks=1, expected_lf_min=1)
    assert not result.passed
    assert "span" in result.detail.lower()


def test_check_schedule_length_interim_cadence_lf_shares_first_short_day(tmp_path: Path):
    """Interim cadence (``--shorts-target 2``): LF Tue 16:00,
    shorts Tue 18:00 + Wed 14:00 — the run-wide identity pair
    ``["14:00:00", "18:00:00"]`` posts up to two shorts a day, so 2 shorts
    need one calendar day, not two. A slots−1 cap would still demand 2d and
    the publish pass would exit "failed DoD validation" after uploading
    (``combined schedule span 1d < required 2d``)."""
    state_dir = tmp_path / "stages"
    _write_state(
        state_dir,
        lf_dates=["2030-09-24T16:00:00-04:00"],
        shorts_dates=["2030-09-24T18:00:00-04:00", "2030-09-25T14:00:00-04:00"],
        identity_times=["14:00:00", "18:00:00"],
    )

    result = v.check_schedule_length(state_dir, expected_weeks=1, expected_lf_min=1)
    assert result.passed, result.detail


def test_check_schedule_length_collapsed_week_of_shorts_still_fails(tmp_path: Path):
    """14 shorts with two posting times a day need 7 days; all on one day is
    a broken schedule even though the per-day pair is honoured."""
    state_dir = tmp_path / "stages"
    _write_state(
        state_dir,
        lf_dates=["2030-09-24"],
        shorts_dates=["2030-09-24"] * 14,
        identity_times=["14:00:00", "18:00:00"],
    )

    result = v.check_schedule_length(state_dir, expected_weeks=1, expected_lf_min=1)
    assert not result.passed
    assert "span" in result.detail.lower()
    assert "6d" in result.detail


def test_check_schedule_length_missing_state_file(tmp_path: Path):
    result = v.check_schedule_length(tmp_path / "nope", expected_weeks=1, expected_lf_min=1)
    assert not result.passed
    assert "pipeline_state.json" in result.detail


# ---------------------------------------------------------------------------
# DoD validator must read finals from the
# day-first cache paths in POST_UPLOAD mode, not just the legacy
# ``<project>/long_forms/`` tree. Day-first runs canonically populate
# ``state["regular_lf_paths"]`` with absolute paths under
# ``<stages>/day_*/regular_lfs/`` and never copy into ``long_forms/``.
# Stale path expectations would otherwise report false FAILs.
# ---------------------------------------------------------------------------


def _build_dayfirst_post_upload_tree(tmp_path: Path, lf_count: int = 2) -> dict:
    """Build a day-first POST_UPLOAD layout: LFs live ONLY under
    ``<stages>/day_*/regular_lfs/`` (not under ``<project>/long_forms/``),
    plus a full delivery-tree set of post-upload sidecars (thumbnails,
    youtube_uploads.json, consolidated shorts CSV, per-LF metadata).
    """
    paths = _build_valid_output(tmp_path, lf_count=lf_count)
    # Move every LF out of long_forms/ into the cache-side day_first layout.
    base_date = datetime(2026, 4, 1)
    moved: list[Path] = []
    for i in range(1, lf_count + 1):
        src = paths["long_forms"] / f"lf_{i:02d}.mp4"
        day = (base_date + timedelta(days=i - 1)).strftime("%Y-%m-%d")
        dst_dir = paths["state_dir"] / f"day_{day}" / "regular_lfs"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"lf_{i:02d}.mp4"
        src.rename(dst)
        moved.append(dst)
    # long_forms/ now empty — verify so the test's premise is real.
    assert not list(paths["long_forms"].iterdir())
    paths["dayfirst_lfs"] = moved
    return paths


def test_post_upload_resolves_lfs_from_state_regular_lf_paths(tmp_path: Path):
    """With state['regular_lf_paths'] pointing at day-first cache LFs
    and an empty <project>/long_forms/, POST_UPLOAD mode must find the LFs
    (no false-FAIL on lf_count / lf_files_size / lf_durations / etc.).
    """
    paths = _build_dayfirst_post_upload_tree(tmp_path, lf_count=2)
    state = make_state(
        skip_upload=False,
        regular_lf_paths=[str(p) for p in paths["dayfirst_lfs"]],
    )
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.POST_UPLOAD,
        state=state,
    )
    # The 6 LF-list-driven checks (#1–6) must pass; metadata check #8 still
    # lands on project_dir sidecars (which the fixture wrote).
    lf_check = next(r for r in report.checks if r.number == 1)
    assert lf_check.passed, lf_check.detail
    assert "state" in lf_check.detail
    for n in (2, 3, 4, 5, 6):
        r = next(c for c in report.checks if c.number == n)
        assert r.passed, f"check #{n} {r.name} failed: {r.detail}"


def test_post_upload_falls_back_to_legacy_long_forms_when_state_empty(tmp_path: Path):
    """Regression: legacy callers without state['regular_lf_paths'] must
    keep working against ``<project>/long_forms/lf_NN.mp4``."""
    paths = _build_valid_output(tmp_path, lf_count=2)
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.POST_UPLOAD,
        state=None,
    )
    lf_check = next(r for r in report.checks if r.number == 1)
    assert lf_check.passed, lf_check.detail
    assert "legacy" in lf_check.detail
    # No regression in total check count (15: #7/#14/#16/#17/#18 gone, #19 + #20 added).
    assert len(report.checks) == 15


def test_post_upload_glob_fallback_to_dayfirst_cache(tmp_path: Path):
    """When neither state['regular_lf_paths'] nor long_forms/ has
    LFs, fall back to globbing ``<state_dir>/day_*/regular_lfs/lf_NN.mp4``
    so an orchestrator that died before writing state still audits.
    """
    paths = _build_dayfirst_post_upload_tree(tmp_path, lf_count=1)
    # No state → must glob the cache.
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.POST_UPLOAD,
        state=None,
    )
    lf_check = next(r for r in report.checks if r.number == 1)
    assert lf_check.passed, lf_check.detail
    assert "day_first_glob" in lf_check.detail


def test_post_upload_no_lfs_fail_message_names_path_scheme(tmp_path: Path):
    """When no LFs are found anywhere, the failure message must
    name which path scheme was tried so the operator can debug quickly.
    """
    paths = _build_skip_upload_tree(tmp_path, lf_count=1)
    # Delete the cache LF too — nothing exists anywhere.
    for _, lf_path in paths["lfs"]:
        lf_path.unlink()
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["stages_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.POST_UPLOAD,
        state=None,
    )
    lf_check = next(r for r in report.checks if r.number == 1)
    assert not lf_check.passed
    # Detail must identify the scheme so an operator knows what was tried.
    assert "none" in lf_check.detail
    assert "path scheme" in lf_check.detail


def test_resolve_post_upload_lfs_state_priority(tmp_path: Path):
    """Unit test for the resolution ladder: state wins over legacy wins
    over day_first_glob."""
    project_dir = tmp_path / "proj"
    long_forms = project_dir / "long_forms"
    long_forms.mkdir(parents=True)
    legacy_lf = long_forms / "lf_01.mp4"
    legacy_lf.write_bytes(b"x")

    state_dir = tmp_path / "stages"
    state_dir.mkdir()
    cache_lf_dir = state_dir / "day_2026-04-01" / "regular_lfs"
    cache_lf_dir.mkdir(parents=True)
    cache_lf = cache_lf_dir / "lf_01.mp4"
    cache_lf.write_bytes(b"x")

    # state set → state wins
    state = make_state(regular_lf_paths=[str(cache_lf)])
    lfs, scheme = v.resolve_post_upload_lfs(project_dir, state, state_dir)
    assert scheme == "state"
    assert lfs == [(1, cache_lf)]

    # state empty → legacy wins
    lfs, scheme = v.resolve_post_upload_lfs(project_dir, {}, state_dir)
    assert scheme == "legacy"
    assert lfs == [(1, legacy_lf)]

    # legacy gone → day_first_glob wins
    legacy_lf.unlink()
    lfs, scheme = v.resolve_post_upload_lfs(project_dir, {}, state_dir)
    assert scheme == "day_first_glob"
    assert lfs == [(1, cache_lf)]

    # nothing left → none
    cache_lf.unlink()
    lfs, scheme = v.resolve_post_upload_lfs(project_dir, {}, state_dir)
    assert scheme == "none"
    assert lfs == []


# ---------------------------------------------------------------------------
# Day-first / full-drive: check_shorts_count must consume state["shorts_per_lf"]
# ---------------------------------------------------------------------------


def test_check_shorts_count_state_aware_passes_for_day_first(tmp_path: Path):
    """Day-first runs upload shorts directly from ``_stages/day_*/shorts/``;
    nothing is copied to ``project_dir/shorts/lf_NN/`` except per-LF social
    JSON. ``state["shorts_per_lf"]`` is the authoritative source — the check
    must trust it when present."""
    project_dir = tmp_path / "out" / PROJECT
    (project_dir / "shorts" / "lf_01").mkdir(parents=True)
    (project_dir / "shorts" / "lf_01" / "social_posts.json").write_text("{}")
    cache_shorts = tmp_path / "cache" / "_stages" / "day_2025-09-14" / "shorts"
    cache_shorts.mkdir(parents=True)
    paths = []
    for i in range(1, 15):
        p = cache_shorts / f"short_{i:02d}.mp4"
        p.write_bytes(b"\x00\x00\x00\x18ftyp" + b"x" * 32)
        paths.append(str(p))
    state = make_state(shorts_per_lf={"1": paths})
    res = v.check_shorts_count(project_dir, state=state)
    assert res.passed, res.detail
    assert "1 LFs" in res.detail


def test_check_shorts_count_state_aware_fails_when_path_missing(tmp_path: Path):
    project_dir = tmp_path / "out" / PROJECT
    (project_dir / "shorts").mkdir(parents=True)
    state = make_state(shorts_per_lf={"1": [str(tmp_path / "missing.mp4")]})
    res = v.check_shorts_count(project_dir, state=state)
    assert not res.passed
    assert "missing" in res.detail


def test_check_shorts_count_legacy_fallback_when_no_state(tmp_path: Path):
    """Non-day-first / legacy callers still hit the directory-based check."""
    project_dir = tmp_path / "out" / PROJECT
    lf_dir = project_dir / "shorts" / "lf_01"
    lf_dir.mkdir(parents=True)
    (lf_dir / "short_01.mp4").write_bytes(b"dummy")
    (lf_dir / "shorts_rendered.json").write_text(json.dumps({"num_rendered": 1}))
    res = v.check_shorts_count(project_dir, state=None)
    assert res.passed, res.detail


# ---------------------------------------------------------------------------
# check_recording_date_per_lf — canonical "Mon YYYY" Filmed format
# ---------------------------------------------------------------------------


def _write_youtube_with_filmed(project_dir: Path, idx: int, filmed_line: str) -> Path:
    """Write a minimal ``{project}_lfNN_youtube.json`` whose description carries
    the given ``📅 Filmed:`` line, and return a synthetic LF path tuple."""
    project_dir.mkdir(parents=True, exist_ok=True)
    desc = f"Highlights from a drive.\n\n📅 Filmed: {filmed_line}\n\n#scenic"
    (project_dir / f"{PROJECT}_lf{idx:02d}_youtube.json").write_text(
        json.dumps({"title": "t", "description": desc})
    )
    return project_dir / f"lf_{idx:02d}.mp4"


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Sep 2025", (2025, 9, None)),
        ("September 2025", (2025, 9, None)),
        ("2025-09-14", (2025, 9, 14)),
        ("September 14, 2025", (2025, 9, 14)),
        ("garbage", None),
        ("", None),
    ],
)
def test_parse_filmed_date_accepts_canonical_and_legacy(value, expected):
    assert v._parse_filmed_date(value) == expected


@pytest.mark.parametrize(
    "actual,expected,match",
    [
        # canonical Mon YYYY vs ISO state -> month granularity
        ("Sep 2025", "2025-09-14", True),
        ("Oct 2025", "2025-09-14", False),
        # legacy ISO actual keeps full day-level discrimination
        ("2025-09-14", "2025-09-14", True),
        ("2025-09-13", "2025-09-14", False),
        # unparseable actual -> None (real defect)
        ("soon", "2025-09-14", None),
    ],
)
def test_filmed_dates_match_precision_aware(actual, expected, match):
    assert v._filmed_dates_match(actual, expected) is match


def test_recording_date_passes_mon_year_against_iso_state(tmp_path: Path):
    """Regression: description renders canonical ``Mon YYYY`` while state holds
    the ISO day — the check must compare at month granularity and PASS."""
    project_dir = tmp_path / PROJECT
    lf_path = _write_youtube_with_filmed(project_dir, 1, "Sep 2025")
    state = {"per_lf_recording_date": {1: "2025-09-14"}}
    res = v.check_recording_date_per_lf(project_dir, [(1, lf_path)], PROJECT, state)
    assert res.passed, res.detail


def test_recording_date_fails_on_wrong_month(tmp_path: Path):
    project_dir = tmp_path / PROJECT
    lf_path = _write_youtube_with_filmed(project_dir, 1, "Oct 2025")
    state = {"per_lf_recording_date": {1: "2025-09-14"}}
    res = v.check_recording_date_per_lf(project_dir, [(1, lf_path)], PROJECT, state)
    assert not res.passed
    assert "Oct 2025" in res.detail and "2025-09-14" in res.detail


def test_recording_date_iso_wrong_day_still_fails(tmp_path: Path):
    """A legacy day-bearing ISO description retains day-level discrimination."""
    project_dir = tmp_path / PROJECT
    lf_path = _write_youtube_with_filmed(project_dir, 1, "2025-09-13")
    state = {"per_lf_recording_date": {1: "2025-09-14"}}
    res = v.check_recording_date_per_lf(project_dir, [(1, lf_path)], PROJECT, state)
    assert not res.passed
    assert "2025-09-13" in res.detail


def test_recording_date_fails_on_unparseable_filmed_value(tmp_path: Path):
    project_dir = tmp_path / PROJECT
    lf_path = _write_youtube_with_filmed(project_dir, 1, "soon")
    state = {"per_lf_recording_date": {1: "2025-09-14"}}
    res = v.check_recording_date_per_lf(project_dir, [(1, lf_path)], PROJECT, state)
    assert not res.passed
    assert "unparseable" in res.detail


# ---------------------------------------------------------------------------
# Check #19 stop_motion (parked-short backstop)
# ---------------------------------------------------------------------------


def _write_stop_mask(state_dir: Path, date: str, mask) -> None:
    import numpy as np

    state_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(state_dir / f"stop_mask_{date}.npz", mask=np.asarray(mask, dtype=bool))


def _write_shorts_audit(state_dir: Path, date: str, shorts: list[dict]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"shorts_ost_audit_{date}.json").write_text(
        json.dumps({"date": date, "shorts": shorts})
    )


def test_check_stop_motion_noops_without_artifacts(tmp_path: Path):
    """No stop mask / shorts audit on disk → no-op PASS (GPS-less/legacy)."""
    res = v.check_stop_motion(tmp_path / "proj", tmp_path / "stages")
    assert res.passed is True
    assert "skipped" in res.detail


def test_check_stop_motion_fails_majority_parked_short(tmp_path: Path):
    """A shipped short whose window is majority-masked stopped → FAIL."""
    import numpy as np

    date = "2026-04-01"
    state_dir = tmp_path / "stages"
    # 40 s day, seconds [10, 30) all stopped.
    mask = np.zeros(40, dtype=bool)
    mask[10:30] = True
    _write_stop_mask(state_dir, date, mask)
    _write_shorts_audit(state_dir, date, [{"rank": 1, "start_sec": 10.0, "end_sec": 30.0}])
    res = v.check_stop_motion(tmp_path / "proj", state_dir)
    assert res.passed is False
    assert "rank 1" in res.detail


def test_check_stop_motion_passes_moving_short(tmp_path: Path):
    """A shipped short whose window is unmasked (moving) → PASS."""
    import numpy as np

    date = "2026-04-01"
    state_dir = tmp_path / "stages"
    mask = np.zeros(40, dtype=bool)  # nothing stopped
    _write_stop_mask(state_dir, date, mask)
    _write_shorts_audit(state_dir, date, [{"rank": 1, "start_sec": 10.0, "end_sec": 30.0}])
    res = v.check_stop_motion(tmp_path / "proj", state_dir)
    assert res.passed is True
    assert "majority moving" in res.detail


# Superlong parts (sl_NN.mp4) — DoD coverage + per-artifact duration ceiling


def test_check_lf_durations_no_superlong_args_unchanged():
    """Regression: byte-identical output with the original signature."""
    lfs = [(1, Path("lf_01.mp4"))]
    probes = {1: {"format": {"duration": "1700.0"}}}
    result = v.check_lf_durations(probes, lfs, 480.0, 1800.0)
    assert result == v.CheckResult(3, "lf_duration", True, "all 1 LFs within bounds")


def test_check_lf_durations_superlong_passes_own_ceiling_regular_still_fails():
    """A 4h superlong part passes its own ceiling; a 4h regular LF still fails its own."""
    four_hours = 14400.0
    probes = {
        1: {"format": {"duration": str(four_hours)}},
        -1: {"format": {"duration": str(four_hours)}},
    }
    result = v.check_lf_durations(
        probes,
        [(1, Path("lf_01.mp4"))],
        480.0,
        1800.0,
        superlong_parts=[(-1, Path("sl_01.mp4"))],
        superlong_max_dur=four_hours * 1.08,
    )
    assert result.passed is False
    assert "lf_01" in result.detail
    assert "sl_01" not in result.detail


def test_discover_state_superlong_parts(tmp_path: Path):
    """Sorted by part_index; negative synthetic indices; skips a record
    whose path is missing on disk."""
    assert v.discover_state_superlong_parts(None) == []
    part1, part2 = tmp_path / "sl_01.mp4", tmp_path / "sl_02.mp4"
    part1.write_bytes(b"x")
    part2.write_bytes(b"x")
    missing = tmp_path / "sl_03.mp4"  # never written
    records = [
        {"path": str(part2), "part_index": 2},
        {"path": str(part1), "part_index": 1},
        {"path": str(missing), "part_index": 3},
    ]
    state = make_state(superlong_parts={"2026-04-01": {"part_count": 3, "records": records}})
    result = v.discover_state_superlong_parts(state)
    assert [p for _idx, p in result] == [part1, part2]
    indices = [idx for idx, _p in result]
    assert all(idx < 0 for idx in indices) and len(set(indices)) == len(indices)


def test_superlong_part_missing_audio_fails_check_audio_stream(tmp_path: Path):
    """A video-only superlong part now fails DoD #05 (previously invisible)."""
    paths = _build_valid_output(tmp_path, lf_count=1)
    sl_path = paths["state_dir"] / "day_2026-04-01" / "superlong" / "sl_01.mp4"
    _make_video_only_sl(sl_path)
    state = make_state(
        superlong_parts={
            "2026-04-01": {
                "part_count": 1,
                "records": [{"path": str(sl_path), "part_index": 1, "part_count": 1}],
            }
        },
    )
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        superlong_max_lf_duration=15552.0,
        mode=v.Mode.POST_UPLOAD,
        state=state,
    )
    audio_check = next(r for r in report.checks if r.number == 5)
    assert audio_check.passed is False
    assert "sl_01" in audio_check.detail


# ── #20 places_verified — audit-pass visibility for the publish place gate ──
# A published short may name only the place its own clip's GPS verified. The
# publish pass refuses otherwise (assert_all_places_verified); this check runs
# in both modes so the --skip-upload audit catches it before the relaunch.
# titles.yaml is an optional override, so its absence is never a failure.


def _place_sidecar(stages: Path, entries: dict[int, dict]) -> list[Path]:
    day = "2026-09-14"
    (stages / f"day_{day}" / "shorts").mkdir(parents=True, exist_ok=True)
    (stages / f"shorts_ost_audit_{day}.json").write_text(
        json.dumps({"shorts": [{"rank": r, **e} for r, e in entries.items()]})
    )
    return [stages / f"day_{day}" / "shorts" / f"short_{r:02d}.mp4" for r in entries]


_GOOD_PLACE = {
    "ost_text": "Birch County, Virginia",
    "regions": ["Birch County, Virginia"],
    "place_source": "clip_gps",
    "start_sec": 10.0,
    "end_sec": 40.0,
}


def test_check_places_verified_passes_when_every_short_verified(tmp_path: Path):
    shorts = _place_sidecar(tmp_path / "stages", {1: _GOOD_PLACE, 2: _GOOD_PLACE})
    state = make_state(shorts_per_lf={1: shorts[:1], 2: shorts[1:]})
    res = v.check_places_verified(tmp_path / "proj", state)
    assert res.number == 20 and res.name == "places_verified"
    assert res.passed, res.detail


def test_check_places_verified_fails_naming_lf_label_short(tmp_path: Path):
    shorts = _place_sidecar(
        tmp_path / "stages", {1: _GOOD_PLACE, 2: {**_GOOD_PLACE, "place_source": "lf_label"}}
    )
    res = v.check_places_verified(tmp_path / "proj", make_state(shorts_per_lf={1: shorts}))
    assert not res.passed
    assert "short_02.mp4" in res.detail and "short_01.mp4" not in res.detail


def test_check_places_verified_fails_on_empty_regions(tmp_path: Path):
    shorts = _place_sidecar(tmp_path / "stages", {1: {**_GOOD_PLACE, "regions": []}})
    res = v.check_places_verified(tmp_path / "proj", make_state(shorts_per_lf={1: shorts}))
    assert not res.passed
    assert "short_01.mp4" in res.detail and "regions" in res.detail


def test_check_places_verified_skips_legacy_state(tmp_path: Path):
    res = v.check_places_verified(tmp_path / "proj", None)
    assert res.passed and "skipped" in res.detail
    res = v.check_places_verified(tmp_path / "proj", make_state())
    assert res.passed and "skipped" in res.detail


def test_check_places_verified_absent_titles_yaml_is_not_a_failure(tmp_path: Path):
    project_dir = tmp_path / "proj"
    shorts = _place_sidecar(tmp_path / "stages", {1: _GOOD_PLACE})
    assert not (project_dir / "social" / "titles.yaml").exists()
    res = v.check_places_verified(project_dir, {"shorts_per_lf": {"1": [str(shorts[0])]}})
    assert res.passed, res.detail


def test_skip_upload_audit_mode_runs_places_verified_check(tmp_path: Path):
    """The audit pass must surface an unverified place, not just the publish pass."""
    paths = _build_valid_output(tmp_path, lf_count=1)
    shorts = _place_sidecar(tmp_path / "stages", {1: {**_GOOD_PLACE, "place_source": "lf_label"}})
    report = v.run_validation(
        project=PROJECT,
        output_base=paths["output_base"],
        snapshot_dir=paths["snapshot_dir"],
        state_dir=paths["state_dir"],
        run_start=paths["run_start"],
        expected_lf_min=1,
        expected_weeks=1,
        min_lf_duration=25.0,
        max_lf_duration=35.0,
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
        state=make_state(shorts_per_lf={1: shorts}),
    )
    check = next(r for r in report.checks if r.number == 20)
    assert check.name == "places_verified"
    assert not check.passed and "short_01.mp4" in check.detail
