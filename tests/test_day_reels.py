"""Day reels from Drive (publishing/day_reels.py): request -> privacy blur -> queue."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gomoveshift.publishing import day_reels
from gomoveshift.publishing.dispatch import PRIVACY_REQUIRED_CLASSES
from gomoveshift.publishing.queue import open_queue_db

CAPTION = "Up on Peakeen Ridge, Sheep's Head. #Ireland #CountyCork #WestCork #SheepsHead"
VIDEO_ID = "1-XmsePdmWUzVZ5EVTXZbzet"


def _request(**over):
    doc = {
        "schema": day_reels.REQUEST_SCHEMA,
        "video_file_id": VIDEO_ID,
        "caption": CAPTION,
        "recording_date": "2026-11-25",
        "platforms": ["facebook", "instagram"],
    }
    doc.update(over)
    return json.dumps(doc)


def _verified(**over):
    classes = {
        name: {"state": "verified", "reason": f"{name} model resolved and ran"}
        for name in PRIVACY_REQUIRED_CLASSES
    }
    for name, state in over.items():
        classes[name] = {"state": state, "reason": "model missing"}
    return lambda blur: classes


class FakeDrive:
    def __init__(self, requests):
        self.requests = dict(requests)
        self.downloads = []

    def list_requests(self):
        return sorted(self.requests)

    def read_request(self, name):
        return self.requests[name]

    def download(self, file_id, dest_dir):
        self.downloads.append(file_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / "20261125_121049.mp4"
        path.write_bytes(b"phone reel with gps tags")
        return path


def _renderer(calls):
    def render(source, output, mask):
        calls.append((Path(source), Path(output), Path(mask)))
        output.write_bytes(b"blurred reel")
        return {"masked_frames": 12, "face_detections": 3, "plate_detections": 1}

    return render


PROBE = {"width": 1080, "height": 1920, "duration": "21.0"}


@pytest.fixture(autouse=True)
def _fake_ffprobe(monkeypatch):
    probe = dict(PROBE)

    def run(cmd, **kw):
        stream = {"codec_type": "video", "codec_name": "h264"}
        stream.update(width=probe["width"], height=probe["height"])
        body = json.dumps({"format": {"duration": probe["duration"]}, "streams": [stream]})
        return subprocess.CompletedProcess(cmd, 0, body, "")

    monkeypatch.setattr(subprocess, "run", run)
    return probe


def _run(tmp_path, drive, *, privacy=None, renderer=None, apply=True, location_free=True):
    calls = []
    outcomes = day_reels.process_requests(
        drive,
        state_dir=tmp_path / "state",
        storage_path=tmp_path / "queue",
        apply=apply,
        now_utc="2026-11-25T23:00:00+00:00",
        privacy_reporter=privacy or _verified(),
        renderer=renderer or _renderer(calls),
        location_check=lambda path: location_free,
        gpu_lock=day_reels.no_gpu_lock,
    )
    return outcomes, calls


def _rows(tmp_path):
    conn = open_queue_db(tmp_path / "queue")
    try:
        return conn.execute(
            "SELECT platform, project_name, scheduled_at, sidecar_path FROM deliveries "
            "ORDER BY platform"
        ).fetchall()
    finally:
        conn.close()


# --- request parsing -------------------------------------------------------


def test_parse_request_accepts_a_valid_request():
    request = day_reels.parse_request("day10.json", _request())
    assert request.video_file_id == VIDEO_ID
    assert request.caption == CAPTION
    assert request.platforms == ("facebook", "instagram")
    assert request.at is None


@pytest.mark.parametrize(
    "over, needle",
    [
        ({"schema": "other/v1"}, "schema"),
        ({"video_file_id": "../../etc"}, "video_file_id"),
        ({"video_file_id": "--config=x"}, "video_file_id"),
        ({"caption": ""}, "caption"),
        ({"caption": "x" * 2201}, "caption"),
        ({"recording_date": "25/11/2026"}, "recording_date"),
        ({"platforms": []}, "platforms"),
        ({"platforms": ["tiktok"]}, "platforms"),
        ({"journal_url": "http://gomoveshift.com/x"}, "journal_url"),
        ({"journal_url": "https://evil.example/gomoveshift.com/"}, "journal_url"),
        ({"at": "2026-11-25T10:00:00"}, "at"),
    ],
)
def test_parse_request_refuses_bad_fields(over, needle):
    with pytest.raises(ValueError, match=needle):
        day_reels.parse_request("day10.json", _request(**over))


def test_parse_request_refuses_non_json():
    with pytest.raises(ValueError, match="JSON"):
        day_reels.parse_request("day10.json", "not json")


# --- processing ------------------------------------------------------------


def test_verified_render_enqueues_blurred_reel_with_exact_caption(tmp_path):
    drive = FakeDrive({"day10.json": _request()})
    outcomes, calls = _run(tmp_path, drive)

    assert [o.state for o in outcomes] == ["enqueued"]
    assert len(calls) == 1
    rows = _rows(tmp_path)
    assert [r[0] for r in rows] == ["facebook", "instagram", "youtube"]
    assert {r[1] for r in rows} == {f"day-reel-{VIDEO_ID}"}
    assert rows[0][2].startswith("2026-11-25T23:00:00")

    sidecar = json.loads(Path(rows[0][3]).read_text())
    record = sidecar["shorts"][0]
    assert record["payloads"]["instagram"] == {"caption": CAPTION}
    assert record["payloads"]["facebook"] == {"description": CAPTION}
    # The queued media is the blurred render, never the downloaded original.
    assert Path(record["media"]["path"]) == calls[0][1]
    assert Path(record["media"]["path"]).read_bytes() == b"blurred reel"
    assert not calls[0][0].exists(), "downloaded original is deleted after the render"

    attestation = json.loads(
        (Path(rows[0][3]).parent / "day_asset_privacy_attestation.json").read_text()
    )
    assert {c["state"] for c in attestation["classes"].values()} == {"verified"}


def test_unverified_model_holds_without_download_or_render(tmp_path):
    drive = FakeDrive({"day10.json": _request()})
    outcomes, calls = _run(tmp_path, drive, privacy=_verified(vehicle_occupants="inconclusive"))

    assert [o.state for o in outcomes] == ["held"]
    assert "vehicle_occupants" in outcomes[0].reason
    assert drive.downloads == [] and calls == []
    assert _rows(tmp_path) == []
    # Transient: nothing recorded, so the next hourly run tries again.
    second, _ = _run(tmp_path, drive)
    assert [o.state for o in second] == ["enqueued"]


def test_failed_privacy_render_is_refused_and_never_queued(tmp_path):
    def failing(source, output, mask):
        raise ValueError("vehicle cabin coverage could not be verified")

    drive = FakeDrive({"day10.json": _request()})
    outcomes, _ = _run(tmp_path, drive, renderer=failing)

    assert [o.state for o in outcomes] == ["refused"]
    assert "cabin" in outcomes[0].reason
    assert _rows(tmp_path) == []
    again, calls = _run(tmp_path, drive)
    assert [o.state for o in again] == ["done"] and calls == []


def test_location_metadata_left_in_render_is_refused(tmp_path):
    drive = FakeDrive({"day10.json": _request()})
    outcomes, _ = _run(tmp_path, drive, location_free=False)
    assert [o.state for o in outcomes] == ["refused"]
    assert "location" in outcomes[0].reason
    assert _rows(tmp_path) == []


def test_invalid_request_is_refused_once(tmp_path):
    drive = FakeDrive({"bad.json": _request(platforms=["tiktok"])})
    outcomes, calls = _run(tmp_path, drive)
    assert [o.state for o in outcomes] == ["refused"]
    assert drive.downloads == [] and calls == []


def test_a_handled_request_is_never_processed_twice(tmp_path):
    drive = FakeDrive({"day10.json": _request()})
    _run(tmp_path, drive)
    outcomes, calls = _run(tmp_path, drive)
    assert [o.state for o in outcomes] == ["done"]
    assert calls == [] and drive.downloads == [VIDEO_ID]
    assert len(_rows(tmp_path)) == 3


def test_dry_run_reads_requests_but_downloads_and_writes_nothing(tmp_path):
    drive = FakeDrive({"day10.json": _request()})
    outcomes, calls = _run(tmp_path, drive, apply=False)
    assert [o.state for o in outcomes] == ["would-process"]
    assert drive.downloads == [] and calls == []
    assert not (tmp_path / "state").exists()


def test_busy_gpu_leaves_the_request_pending(tmp_path):
    def busy():
        raise day_reels.GpuLockBusyError("held by night render")

    drive = FakeDrive({"day10.json": _request()})
    outcomes = day_reels.process_requests(
        drive,
        state_dir=tmp_path / "state",
        storage_path=tmp_path / "queue",
        apply=True,
        now_utc="2026-11-25T23:00:00+00:00",
        privacy_reporter=_verified(),
        renderer=_renderer([]),
        location_check=lambda path: True,
        gpu_lock=busy,
    )
    assert [o.state for o in outcomes] == ["held"]
    assert "GPU" in outcomes[0].reason
    assert drive.downloads == []


# --- rclone reader -----------------------------------------------------------


def test_rclone_drive_is_read_only_and_keeps_the_token_out_of_argv(tmp_path):
    seen = []

    def runner(argv, **kwargs):
        seen.append((argv, kwargs["env"]))
        if argv[1] == "lsjson":
            out = json.dumps(
                [
                    {"Name": "day10.json", "IsDir": False},
                    {"Name": "notes.txt", "IsDir": False},
                    {"Name": "old", "IsDir": True},
                ]
            )
        elif argv[1] == "cat":
            out = _request()
        else:
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            (Path(argv[-1]) / "clip.mp4").write_bytes(b"x")
            out = ""
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    reply = json.dumps({"secret": "t"})
    drive = day_reels.RcloneDrive(folder_id="FOLDER123", token=reply, runner=runner)
    assert drive.list_requests() == ["day10.json"]
    assert "caption" in drive.read_request("day10.json")
    path = drive.download(VIDEO_ID, tmp_path / "dl")
    assert path.name == "clip.mp4"

    for argv, env in seen:
        assert all("secret" not in part for part in argv)
        assert env["RCLONE_CONFIG_GMSDAYREELS_SCOPE"] == "drive.readonly"
        assert env["RCLONE_CONFIG_GMSDAYREELS_ROOT_FOLDER_ID"] == "FOLDER123"
    assert seen[-1][0][1:4] == ["backend", "copyid", "gmsdayreels:"]


def test_authorize_token_is_parsed_from_rclone_output():
    out = (
        "Paste the following into your remote machine --->\n"
        '{"access_token":"a","refresh_token":"r"}\n'
        "<---End paste\n"
    )
    assert day_reels.parse_authorize_output(out) == '{"access_token":"a","refresh_token":"r"}'
    with pytest.raises(ValueError):
        day_reels.parse_authorize_output("no token here")


# --- YouTube Shorts ----------------------------------------------------------


def _record(tmp_path):
    sidecar = Path(_rows(tmp_path)[0][3])
    return json.loads(sidecar.read_text())["shorts"][0]


def test_every_reel_also_goes_to_youtube_as_a_short(tmp_path):
    drive = FakeDrive(
        {"day10.json": _request(journal_url="https://gomoveshift.com/journal/day-10/")}
    )
    outcomes, calls = _run(tmp_path, drive)

    assert [o.state for o in outcomes] == ["enqueued"]
    assert "youtube" in outcomes[0].reason
    youtube = _record(tmp_path)["payloads"]["youtube"]
    assert youtube["title"] == "Up on Peakeen Ridge, Sheep's Head."
    assert youtube["description"] == (
        CAPTION + "\n\nRead the full day: https://gomoveshift.com/journal/day-10/\n\n#Shorts"
    )
    assert youtube["tags"] == ["Ireland", "CountyCork", "WestCork", "SheepsHead"]
    # The YouTube upload guard admits only canonical short names.
    assert calls[0][1].name == "short_01.mp4"
    # Facebook and Instagram keep the caption byte for byte.
    record = _record(tmp_path)
    assert record["payloads"]["instagram"] == {"caption": CAPTION}
    assert record["payloads"]["facebook"] == {"description": CAPTION}


def test_youtube_listed_in_the_request_is_queued_once(tmp_path):
    drive = FakeDrive({"day10.json": _request(platforms=["instagram", "youtube"])})
    _run(tmp_path, drive)
    assert [r[0] for r in _rows(tmp_path)] == ["instagram", "youtube"]


@pytest.mark.parametrize(
    "probe, needle",
    [({"width": 1920, "height": 1080}, "landscape"), ({"duration": "181.0"}, "180s")],
)
def test_a_reel_that_is_not_a_short_stays_off_youtube(tmp_path, _fake_ffprobe, probe, needle):
    _fake_ffprobe.update(probe)
    drive = FakeDrive({"day10.json": _request()})
    outcomes, _ = _run(tmp_path, drive)
    assert [o.state for o in outcomes] == ["enqueued"]
    assert needle in outcomes[0].reason
    assert [r[0] for r in _rows(tmp_path)] == ["facebook", "instagram"]


def test_youtube_title_is_the_first_line_without_hashtags():
    assert day_reels.youtube_title("Cliffs at <Slieve League> #Donegal", "2026-12-01") == (
        "Cliffs at Slieve League"
    )
    assert day_reels.youtube_title("#Ireland #Donegal", "2026-12-01") == (
        "GoMoveShift day reel, 01 Dec 2026"
    )
    long = day_reels.youtube_title("word " * 40, "2026-12-01")
    assert len(long) <= 100 and long.endswith("\u2026")


def test_youtube_description_keeps_an_existing_shorts_tag_once():
    text = day_reels.youtube_description("Rain on the Gap #Shorts", None)
    assert text == "Rain on the Gap #Shorts"


def test_the_queued_short_passes_the_youtube_upload_gates(tmp_path):
    """What the drain checks before ``videos.insert``: the filename guard and
    the final-text review (a GPS-less Short has none to refuse)."""
    import re

    from gomoveshift.config import DriveConfig
    from gomoveshift.publishing.plan_io import load_publish_plan
    from gomoveshift.publishing.youtube_adapter import short_review_refusal

    drive = FakeDrive({"day10.json": _request()})
    _run(tmp_path, drive)
    youtube_row = [r for r in _rows(tmp_path) if r[0] == "youtube"][0]
    record = load_publish_plan(Path(youtube_row[3])).shorts[0]

    assert re.match(DriveConfig().youtube_upload_filename_pattern, Path(record.media.path).name)
    assert short_review_refusal(record) is None
    assert record.payloads.youtube.title == "Up on Peakeen Ridge, Sheep's Head."
