"""exiftool capture reader: argv shape, batching, mapping, degradation."""

import json
import math
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline.photos import capture_meta as cm


class FakeRun:
    def __init__(self, rows_for=None, returncode=0, raises=None, stdout=None):
        self.calls = []
        self.argfiles = []
        self.rows_for = rows_for or (lambda paths: [{"SourceFile": p} for p in paths])
        self.returncode = returncode
        self.raises = raises
        self.stdout = stdout

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        argfile = argv[argv.index("-@") + 1]
        paths = Path(argfile).read_text(encoding="utf-8").splitlines()
        self.argfiles.append(paths)
        if self.raises:
            raise self.raises
        out = self.stdout if self.stdout is not None else json.dumps(self.rows_for(paths))
        return subprocess.CompletedProcess(argv, self.returncode, out, "")


def test_argv_shape_argfile_no_shell(tmp_path):
    run = FakeRun()
    p = tmp_path / "a.nef"
    cm.read_capture_records([p], exiftool="/x/exiftool", run=run)
    argv, kwargs = run.calls[0]
    assert argv[:7] == ["/x/exiftool", "-json", "-n", "-fast", "-charset", "filename=utf8", "-@"]
    assert "-ShutterCount" in argv and str(p) not in argv
    assert not kwargs.get("shell")
    assert run.argfiles[0] == [str(p)]


def test_batching_splits_450_into_three(tmp_path):
    run = FakeRun()
    paths = [tmp_path / f"{i}.nef" for i in range(450)]
    recs, warns = cm.read_capture_records(paths, exiftool="/x", run=run)
    assert len(run.calls) == 3 and len(recs) == 450 and warns == []


def test_json_mapped_by_source_file_and_fields(tmp_path):
    p = tmp_path / "a.nef"
    row = {
        "SourceFile": str(tmp_path / "." / "a.nef"),
        "Make": " NIKON ",
        "Model": "Z50_2",
        "SerialNumber": 123,
        "ShutterCount": 42,
        "DateTimeOriginal": "2024:01:02 03:04:05",
        "SubSecTimeOriginal": 7,
        "FocusDistance": 1.5,
        "Lens": "NIKKOR 24",
        "FocalLength": 24,
        "FNumber": 4,
        "ISO": 100,
        "ExposureTime": 0.01,
    }
    recs, _ = cm.read_capture_records([p], exiftool="/x", run=FakeRun(lambda _: [row]))
    r = recs[p]
    assert (r.make, r.model, r.serial, r.shutter_count) == ("NIKON", "Z50_2", "123", 42)
    assert (r.datetime_original, r.subsec) == ("2024:01:02 03:04:05", "7")
    assert (r.focus_distance_m, r.lens_model, r.focal_length_mm) == (1.5, "NIKKOR 24", 24.0)
    assert (r.f_number, r.iso, r.exposure_s) == (4.0, 100.0, 0.01)


def test_serial_and_lens_fallback_tags(tmp_path):
    p = tmp_path / "a.nef"
    row = {"SourceFile": str(p), "InternalSerialNumber": "I9", "LensID": "L1"}
    r = cm.read_capture_records([p], exiftool="/x", run=FakeRun(lambda _: [row]))[0][p]
    assert (r.serial, r.lens_model) == ("I9", "L1")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("inf", math.inf),
        ("Infinity", math.inf),
        (2.5, 2.5),
        ("junk", None),
        (-1, None),
        (None, None),
    ],
)
def test_focus_distance_values(tmp_path, raw, expected):
    p = tmp_path / "a.nef"
    row = {"SourceFile": str(p), "FocusDistance": raw}
    r = cm.read_capture_records([p], exiftool="/x", run=FakeRun(lambda _: [row]))[0][p]
    assert r.focus_distance_m == expected


def test_garbage_numerics_are_none(tmp_path):
    p = tmp_path / "a.nef"
    row = {
        "SourceFile": str(p),
        "ShutterCount": "n/a",
        "ISO": "x",
        "FNumber": True,
        "FocalLength": float("nan"),
        "Make": [1],
        "Model": "  ",
    }
    r = cm.read_capture_records([p], exiftool="/x", run=FakeRun(lambda _: [row]))[0][p]
    assert (r.shutter_count, r.iso, r.f_number, r.focal_length_mm) == (None,) * 4
    assert (r.make, r.model) == (None, None)


def test_missing_exiftool_falls_back_with_warning(tmp_path, monkeypatch):
    from gomoveshift.pipeline.photos import exif_gps

    class Exif:
        iso = 200.0
        exposure_s = 0.5

    monkeypatch.setattr(exif_gps, "extract_exif_capture_metadata", lambda p: Exif())
    monkeypatch.setattr(exif_gps, "extract_exif_focal_length", lambda p: 35.0)
    monkeypatch.setattr(exif_gps, "extract_exif_aperture", lambda p: 2.8)
    monkeypatch.setattr(cm, "exiftool_path", lambda: None)
    p = tmp_path / "a.jpg"
    recs, warns = cm.read_capture_records([p])
    assert warns == [cm.MISSING_WARNING]
    r = recs[p]
    assert (r.iso, r.exposure_s, r.focal_length_mm, r.f_number) == (200.0, 0.5, 35.0, 2.8)
    assert (r.serial, r.shutter_count, r.focus_distance_m) == (None, None, None)


def test_missing_exiftool_no_paths_no_warning():
    assert cm.read_capture_records([], exiftool=None) == ({}, [])


@pytest.mark.parametrize(
    "fake",
    [
        FakeRun(raises=subprocess.TimeoutExpired("x", 1)),
        FakeRun(returncode=2),
        FakeRun(stdout="not json"),
        FakeRun(stdout='{"a": 1}'),
        FakeRun(raises=OSError("boom")),
    ],
)
def test_failed_batch_warns_and_falls_back(tmp_path, monkeypatch, fake):
    from gomoveshift.pipeline.photos import exif_gps

    monkeypatch.setattr(
        exif_gps,
        "extract_exif_capture_metadata",
        lambda p: type("E", (), {"iso": 50.0, "exposure_s": None})(),
    )
    monkeypatch.setattr(exif_gps, "extract_exif_focal_length", lambda p: None)
    monkeypatch.setattr(exif_gps, "extract_exif_aperture", lambda p: None)
    p = tmp_path / "a.jpg"
    recs, warns = cm.read_capture_records([p], exiftool="/x", run=fake)
    assert recs[p].iso == 50.0
    assert len(warns) == 1 and "batch 1" in warns[0]


def test_path_missing_from_output(tmp_path):
    a, b = tmp_path / "a.nef", tmp_path / "b.nef"
    run = FakeRun(lambda _: [{"SourceFile": str(a), "Make": "N"}, "junk", {"Make": "no source"}])
    recs, warns = cm.read_capture_records([a, b], exiftool="/x", run=run)
    assert recs[a].make == "N"
    assert recs[b] == cm.CaptureRecord(path=b)
    assert warns == [f"exiftool returned nothing for {b}"]


def test_default_exiftool_path_used(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "exiftool_path", lambda: "/found/exiftool")
    run = FakeRun()
    cm.read_capture_records([tmp_path / "a"], run=run)
    assert run.calls[0][0][0] == "/found/exiftool"


def test_resolve_survives_oserror(monkeypatch):
    def boom(self, *a, **k):
        raise OSError

    monkeypatch.setattr(Path, "resolve", boom)
    assert cm._resolve("x") == Path("x")


def test_real_exiftool_on_tiny_jpeg(tmp_path):
    if cm.exiftool_path() is None:
        pytest.skip("exiftool not installed")
    from PIL import Image

    p = tmp_path / "t.jpg"
    Image.new("RGB", (4, 4)).save(p)
    recs, warns = cm.read_capture_records([p])
    assert warns == [] and recs[p].path == p
