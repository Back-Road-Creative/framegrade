"""Day reels from Drive (publishing/day_reels.py): request -> privacy blur -> queue."""

from __future__ import annotations

import base64
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


def _run(
    tmp_path,
    drive,
    *,
    privacy=None,
    renderer=None,
    apply=True,
    location_free=True,
    journal_link=lambda recording_date: None,
):
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
        journal_link=journal_link,
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


def test_authorize_token_is_unwrapped_from_rclone_base64_reply():
    # rclone v1.60 answers a base64 config blob with a base64 (URL-safe, unpadded)
    # {"token": "<token JSON>"} -- captured from a real run against a fake token server.
    inner = '{"access_token":"FAKEACCESS","token_type":"Bearer","refresh_token":"FAKEREFRESH"}'
    wrapped = base64.urlsafe_b64encode(json.dumps({"token": inner}).encode()).decode().rstrip("=")
    out = f"Paste the following into your remote machine --->\n{wrapped}\n<---End paste\n"
    assert day_reels.parse_authorize_output(out) == inner
    junk = base64.urlsafe_b64encode(b"[1, 2]").decode()
    with pytest.raises(ValueError) as caught:
        day_reels.parse_authorize_output(
            f"Paste the following into your remote machine --->\n{junk}\n<---End paste"
        )
    assert junk not in str(caught.value)


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


# --- Journal day link --------------------------------------------------------

TRIPS = json.dumps(
    {
        "trips": [
            {
                "slug": "ireland-2026-11",
                "title": "Ireland",
                "start": "2026-11-16",
                "days": 46,
                "timezone": "Europe/Dublin",
                "finished": False,
            }
        ]
    }
)
DAY_10 = "https://gomoveshift.com/journeys/ireland-2026-11/#day-10"


@pytest.mark.parametrize(
    "recording_date, expected",
    [
        ("2026-11-25", DAY_10),
        ("2026-11-16", "https://gomoveshift.com/journeys/ireland-2026-11/#day-1"),
        ("2026-12-31", "https://gomoveshift.com/journeys/ireland-2026-11/#day-46"),
        ("2026-11-15", None),
        ("2027-01-01", None),
        ("25/11/2026", None),
    ],
)
def test_journal_day_link_numbers_days_from_the_trip_start(recording_date, expected):
    assert day_reels.journal_day_link(TRIPS, recording_date) == expected


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        json.dumps({"trips": "x"}),
        json.dumps({"trips": [{"slug": "Bad Slug", "start": "2026-11-16", "days": 46}]}),
        json.dumps({"trips": [{"slug": "a", "start": "2026-13-01", "days": 46}]}),
        json.dumps({"trips": [{"slug": "a", "start": "2026-11-16", "days": "46"}]}),
        json.dumps({"trips": [{"slug": "a", "start": "2026-11-16", "days": True}]}),
    ],
)
def test_journal_day_link_is_none_for_a_bad_trip_list(text):
    assert day_reels.journal_day_link(text, "2026-11-25") is None


def test_journal_day_link_is_none_when_two_trips_claim_the_day():
    trips = json.loads(TRIPS)["trips"]
    both = json.dumps({"trips": trips + [{**trips[0], "slug": "other-trip"}]})
    assert day_reels.journal_day_link(both, "2026-11-25") is None


def test_unreachable_trip_list_gives_no_link(monkeypatch):
    class Refusing:
        def open(self, *args, **kwargs):
            raise OSError("no route to host")

    monkeypatch.setattr(day_reels.urllib.request, "build_opener", lambda *h: Refusing())
    assert day_reels.fetch_journal_trips() is None
    assert day_reels.lookup_journal_link("2026-11-25") is None


def test_a_short_without_a_journal_url_links_its_journal_day(tmp_path):
    asked = []

    def link(recording_date):
        asked.append(recording_date)
        return DAY_10

    outcomes, _ = _run(tmp_path, FakeDrive({"day10.json": _request()}), journal_link=link)
    assert [o.state for o in outcomes] == ["enqueued"]
    assert asked == ["2026-11-25"]
    youtube = _record(tmp_path)["payloads"]["youtube"]
    assert f"Read the full day: {DAY_10}" in youtube["description"]
    # The Facebook and Instagram caption stays the owner's words only.
    assert _record(tmp_path)["payloads"]["instagram"] == {"caption": CAPTION}


def test_the_request_journal_url_wins_over_the_lookup(tmp_path):
    own = "https://gomoveshift.com/journeys/ireland-2026-11/#day-9"

    def link(recording_date):
        raise AssertionError("the lookup must not run when the request names a page")

    drive = FakeDrive({"day10.json": _request(journal_url=own)})
    _run(tmp_path, drive, journal_link=link)
    assert f"Read the full day: {own}" in _record(tmp_path)["payloads"]["youtube"]["description"]


def test_no_journal_day_means_no_link_line(tmp_path):
    _run(tmp_path, FakeDrive({"day10.json": _request()}))
    description = _record(tmp_path)["payloads"]["youtube"]["description"]
    assert "Read the full day" not in description and description.endswith("#Shorts")


def test_a_queue_refusal_is_recorded_and_later_requests_still_run(tmp_path, monkeypatch):
    real = day_reels.enqueue_from_plan_sidecar
    seen = []

    def enqueue(conn, sidecar_path):
        seen.append(sidecar_path)
        if len(seen) == 1:
            raise day_reels.DeliveryPlanConflictError("an ACTIVE delivery already exists")
        return real(conn, sidecar_path)

    monkeypatch.setattr(day_reels, "enqueue_from_plan_sidecar", enqueue)
    drive = FakeDrive(
        {
            "day10.json": _request(),
            "day11.json": _request(
                video_file_id="2-OtherClipIdForDay11xx", recording_date="2026-11-26"
            ),
        }
    )
    outcomes, _ = _run(tmp_path, drive)
    assert [(o.name, o.state) for o in outcomes] == [
        ("day10.json", "refused"),
        ("day11.json", "enqueued"),
    ]
    assert "queue refused the reel" in outcomes[0].reason
    # Recorded, so the next hourly run does not retry it.
    again, _ = _run(tmp_path, drive)
    assert [o.state for o in again] == ["done", "done"]


def test_a_second_reel_of_the_same_footage_is_refused_not_queued(tmp_path):
    # Two Drive files, one render (the fake renderer writes the same bytes):
    # the same content under a second day-reel project name is refused.
    other = _request(video_file_id="2-OtherClipIdForDay11xx", recording_date="2026-11-26")
    drive = FakeDrive({"day10.json": _request(), "day11.json": other})
    outcomes, _ = _run(tmp_path, drive)
    assert [(o.name, o.state) for o in outcomes] == [
        ("day10.json", "enqueued"),
        ("day11.json", "refused"),
    ]
    assert "duplicate_content" in outcomes[1].reason
    assert {row["project_name"] for row in _rows(tmp_path)} == {f"day-reel-{VIDEO_ID}"}


EMPTY = "{}"
OWN = ("own-client.apps.googleusercontent.com", "own-client-value")


def _rclone(stdout="[]", rc=0, stderr="", seen=None):
    def runner(argv, **kwargs):
        if seen is not None:
            seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, rc, stdout=stdout, stderr=stderr)

    return runner


def test_rclone_drive_uses_the_pipelines_own_google_client_outside_argv():
    seen = []
    drive = day_reels.RcloneDrive(
        folder_id="FOLDER123",
        token=EMPTY,
        client_id=OWN[0],
        client_secret=OWN[1],
        runner=_rclone(seen=seen),
    )
    drive.list_requests()
    argv, kwargs = seen[0]
    assert kwargs["env"]["RCLONE_CONFIG_GMSDAYREELS_CLIENT_ID"] == OWN[0]
    assert kwargs["env"]["RCLONE_CONFIG_GMSDAYREELS_CLIENT_SECRET"] == OWN[1]
    assert all(OWN[1] not in part for part in argv)


def test_an_old_sign_in_without_a_client_keeps_rclones_default(monkeypatch):
    monkeypatch.delenv("RCLONE_CONFIG_GMSDAYREELS_CLIENT_ID", raising=False)
    seen = []
    day_reels.RcloneDrive(
        folder_id="FOLDER123", token=EMPTY, runner=_rclone(seen=seen)
    ).list_requests()
    assert "RCLONE_CONFIG_GMSDAYREELS_CLIENT_ID" not in seen[0][1]["env"]


def test_listing_and_reading_give_up_fast():
    seen = []
    drive = day_reels.RcloneDrive(folder_id="FOLDER123", token=EMPTY, runner=_rclone(seen=seen))
    drive.list_requests()
    drive.read_request("day10.json")
    for argv, kwargs in seen:
        assert kwargs["timeout"] <= 60
        flags = dict(zip(argv[2::], argv[3::]))
        assert flags["--low-level-retries"] == "2"
        assert flags["--contimeout"] == "15s"


THROTTLED = (
    "googleapi: Error 403: Quota exceeded for quota metric 'Queries' and limit "
    "'Previous quota: Requests per minute' of service 'drive.googleapis.com', rateLimitExceeded"
)


def test_a_throttled_listing_says_google_is_throttling():
    drive = day_reels.RcloneDrive(
        folder_id="FOLDER123",
        token=EMPTY,
        runner=_rclone(rc=1, stderr=f"Failed to lsjson: {THROTTLED}"),
    )
    with pytest.raises(day_reels.DriveReadError, match="Google is throttling Drive reads"):
        drive.list_requests()


def test_a_timeout_while_throttled_says_google_is_throttling():
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(
            argv, kwargs["timeout"], stderr=f"DEBUG : pacer: {THROTTLED}".encode()
        )

    drive = day_reels.RcloneDrive(folder_id="FOLDER123", token=EMPTY, runner=runner)
    with pytest.raises(day_reels.DriveReadError, match="Google is throttling Drive reads"):
        drive.list_requests()


def test_a_plain_timeout_still_says_it_timed_out():
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    drive = day_reels.RcloneDrive(folder_id="FOLDER123", token=EMPTY, runner=runner)
    with pytest.raises(day_reels.DriveReadError, match="timed out"):
        drive.list_requests()


# --- tracking record ---------------------------------------------------------


def test_a_queued_reel_gets_a_tracking_record_its_publish_can_update(tmp_path):
    from gomoveshift.tracking import TrackingStore

    drive = FakeDrive({"day10.json": _request()})
    outcomes, calls = _run(tmp_path, drive)
    assert [o.state for o in outcomes] == ["enqueued"]

    store = TrackingStore()
    project = f"day-reel-{VIDEO_ID}"
    record = store.load(project)
    assert [s.index for s in record.shorts] == [0]
    assert record.shorts[0].content.duration_sec == 21.0
    assert Path(record.shorts[0].file) == calls[0][1]

    # The publish hook's one write into tracking now finds the reel.
    store.update_platform_id(project, 0, "youtube", "abc123")
    assert store.load(project).shorts[0].platform_ids == {"youtube": "abc123"}


def test_a_tracking_failure_never_undoes_a_queued_reel(tmp_path):
    def broken(*args):
        raise OSError("disk full")

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
        gpu_lock=day_reels.no_gpu_lock,
        journal_link=lambda recording_date: None,
        register_tracking=broken,
    )
    assert [o.state for o in outcomes] == ["enqueued"]
    assert "tracking record not written: disk full" in outcomes[0].reason
    assert len(_rows(tmp_path)) == 3
