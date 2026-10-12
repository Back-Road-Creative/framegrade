"""The finished film is measured, not assumed.

Every check here mirrors a manual review of a finished render: is the file
the resolution, frame rate, colour tagging and audio format that was asked
for, how many frames did it get, and is any stretch of it black or frozen.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import delivery
from gomoveshift.pipeline.route_story import film as film_module

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _probe(**overrides) -> dict:
    video = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 1920,
        "height": 1080,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "30/1",
        "nb_frames": "300",
        "color_range": "tv",
        "color_space": "bt709",
        "color_primaries": "bt709",
        "color_transfer": "bt709",
    }
    audio = {
        "codec_type": "audio",
        "codec_name": "aac",
        "sample_rate": "48000",
        "channels": 2,
    }
    video.update(overrides.pop("video", {}))
    audio.update(overrides.pop("audio", {}))
    streams = overrides.pop("streams", [video, audio])
    return {"streams": streams, "format": {"duration": "10.000000"}}


def _runner(probe: dict, scan_stderr: str = ""):
    calls: list[list[str]] = []

    def run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[0] == "ffprobe":
            return SimpleNamespace(returncode=0, stdout=json.dumps(probe), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr=scan_stderr)

    run.calls = calls
    return run


def _check(probe: dict, scan_stderr: str = "", **spec):
    spec = {"width": 1920, "height": 1080, "fps": 30, **spec}
    return delivery.check_delivery(Path("film.mp4"), runner=_runner(probe, scan_stderr), **spec)


class TestAnOnSpecFilmPasses:
    def test_no_issues_and_the_measured_facts_are_recorded(self):
        report = _check(_probe())

        assert report["issues"] == []
        assert report["video"]["width"] == 1920
        assert report["video"]["fps"] == 30.0
        assert report["video"]["frames"] == 300
        assert report["audio"] == {"codec": "aac", "sample_rate": 48000, "channels": 2}
        assert report["duration_s"] == 10.0
        assert report["black"] == [] and report["frozen"] == []

    def test_the_whole_file_is_scanned_not_a_sample(self):
        runner = _runner(_probe())

        delivery.check_delivery(Path("film.mp4"), width=1920, height=1080, fps=30, runner=runner)

        scan = next(cmd for cmd in runner.calls if cmd[0] == "ffmpeg")
        assert "-ss" not in scan and "-t" not in scan
        assert any("blackdetect" in part and "freezedetect" in part for part in scan)


class TestAnOffSpecFilmIsNamed:
    @pytest.mark.parametrize(
        ("overrides", "needle"),
        [
            ({"video": {"width": 1280, "height": 720}}, "1280x720"),
            ({"video": {"r_frame_rate": "24/1"}}, "24"),
            ({"video": {"pix_fmt": "yuv444p"}}, "yuv444p"),
            ({"video": {"color_primaries": ""}}, "color_primaries"),
            ({"video": {"color_range": "pc"}}, "color_range"),
            ({"audio": {"sample_rate": "44100"}}, "44100"),
            ({"audio": {"channels": 1}}, "channels"),
            ({"audio": {"codec_name": "mp3"}}, "mp3"),
        ],
    )
    def test_each_departure_from_the_spec_is_an_issue(self, overrides, needle):
        report = _check(_probe(**overrides))

        assert any(needle in issue for issue in report["issues"]), report["issues"]

    def test_a_film_with_no_audio_stream_is_an_issue(self):
        probe = _probe()
        probe["streams"] = probe["streams"][:1]

        assert any("audio" in issue for issue in _check(probe)["issues"])

    def test_a_social_sized_route_story_short_with_no_audio_stream_is_an_issue(self):
        # The route-story social cut (delivered
        # at the route_story_short rendition's 1080x1920) keeps its
        # attenuated audio -- a delivered file missing that stream is a
        # contract violation, same as any other off-spec film.
        probe = _probe(video={"width": 1080, "height": 1920})
        probe["streams"] = probe["streams"][:1]

        report = _check(probe, width=1080, height=1920)

        assert any("audio" in issue for issue in report["issues"])

    def test_a_frame_count_that_disagrees_with_the_duration_is_an_issue(self):
        report = _check(_probe(video={"nb_frames": "250"}))

        assert any("250" in issue and "frames" in issue for issue in report["issues"])

    def test_a_file_that_does_not_probe_is_an_issue_not_a_pass(self):
        def run(cmd, **kwargs):
            return SimpleNamespace(returncode=1, stdout="", stderr="Invalid data")

        report = delivery.check_delivery(
            Path("film.mp4"), width=1920, height=1080, fps=30, runner=run
        )

        assert any("ffprobe" in issue for issue in report["issues"])


class TestBlackAndFrozenStretches:
    _SCAN = (
        "[blackdetect @ 0x1] black_start:4.5 black_end:5.5 black_duration:1\n"
        "[freezedetect @ 0x2] lavfi.freezedetect.freeze_start: 7\n"
        "[freezedetect @ 0x2] lavfi.freezedetect.freeze_duration: 2.5\n"
        "[freezedetect @ 0x2] lavfi.freezedetect.freeze_end: 9.5\n"
    )

    def test_a_black_stretch_is_an_issue_with_its_time(self):
        report = _check(_probe(), self._SCAN)

        assert report["black"] == [{"start_s": 4.5, "duration_s": 1.0}]
        assert any("black" in issue and "4.5" in issue for issue in report["issues"])

    def test_a_frozen_stretch_is_recorded_but_is_not_an_issue(self):
        """A held title card is a choice; a reader of the receipt judges it."""
        report = _check(_probe(), self._SCAN)

        assert report["frozen"] == [{"start_s": 7.0, "duration_s": 2.5}]
        assert not any("frozen" in issue or "freeze" in issue for issue in report["issues"])

    def test_a_freeze_that_runs_to_the_end_of_the_film_is_still_recorded(self):
        scan = "[freezedetect @ 0x2] lavfi.freezedetect.freeze_start: 8\n"

        assert _check(_probe(), scan)["frozen"] == [{"start_s": 8.0, "duration_s": 2.0}]


@needs_ffmpeg
def test_a_real_encode_is_measured_end_to_end(tmp_path):
    film = tmp_path / "film.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=2",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", str(film),
        ],
        check=True,
    )  # fmt: skip

    report = delivery.check_delivery(film, width=320, height=180, fps=30)

    assert report["issues"] == []
    assert report["video"]["frames"] == 60
    assert report["black"] == [] and report["frozen"] == []


def _main_argv(tmp_path: Path) -> list[str]:
    gpx = tmp_path / "track.gpx"
    gpx.write_text(
        """<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    photos = tmp_path / "photos"
    photos.mkdir()
    national, local = tmp_path / "national.jpg", tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    return [
        "--gpx", str(gpx), "--recorded-on", "2026-09-16", "--photos-dir", str(photos),
        "--national-raster", str(national), "--national-credit", "Test overview",
        "--local-raster", str(local), "--local-credit", "Test local",
        "--local-bounds", "-80.1", "34.9", "-79.9", "35.1",
        "--output", str(tmp_path / "film.mp4"), "--receipt", str(tmp_path / "receipt.json"),
        "--width", "640", "--height", "360", "--fps", "24",
    ]  # fmt: skip


class TestTheFilmCommandChecksWhatItDelivered:
    def _run(self, tmp_path, monkeypatch, issues: list[str]):
        seen: dict = {}

        def fake_check(path, *, width, height, fps):
            seen.update(path=Path(path), width=width, height=height, fps=fps)
            return {"issues": issues, "video": {"frames": 1}}

        monkeypatch.setattr(
            film_module,
            "render_route_story_film",
            lambda **kwargs: Path(kwargs["output"]).write_bytes(b"fake film bytes"),
        )
        monkeypatch.setattr(
            film_module, "encode_final", lambda s, d: Path(d).write_bytes(Path(s).read_bytes())
        )
        monkeypatch.setattr(film_module, "check_delivery", fake_check)
        code = film_module.main(_main_argv(tmp_path))
        receipt = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
        return code, receipt, seen

    def test_the_delivered_file_is_checked_against_the_requested_export(
        self, tmp_path, monkeypatch
    ):
        code, receipt, seen = self._run(tmp_path, monkeypatch, issues=[])

        assert code == 0
        assert seen == {"path": tmp_path / "film.mp4", "width": 640, "height": 360, "fps": 24}
        assert receipt["delivery_check"] == {"issues": [], "video": {"frames": 1}}

    def test_an_off_spec_film_fails_the_command_and_still_writes_the_receipt(
        self, tmp_path, monkeypatch, capsys
    ):
        code, receipt, _ = self._run(tmp_path, monkeypatch, issues=["video is 1280x720"])

        assert code == 2
        assert receipt["delivery_check"]["issues"] == ["video is 1280x720"]
        assert "video is 1280x720" in capsys.readouterr().err
