"""Route Story runs inside a drive as a sibling deliverable.

``attach_trail_film`` takes a day's ``Deliverables`` and, when the day has a
GPX and a prepared imagery bundle, runs prepare-lookup -> proof -> verify ->
film with the LF as the placed source clip. Anything missing is a loud, recorded
skip -- never silence, never a failed drive.
"""

from __future__ import annotations

import inspect
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline import media_api
from gomoveshift.pipeline.drive_phases import phase_bc_day_first, trail_film
from gomoveshift.pipeline.media_api import (
    DayArtifacts,
    Deliverables,
    TrailFilm,
    deliverables_to_state,
)
from gomoveshift.pipeline.route_story import proof
from gomoveshift.pipeline.route_story.preflight import (
    TerrainRendererUnavailableError,
    probe_3d_renderer,
)
from gomoveshift.pipeline.route_story.verify import VerifyReport

DAY = "2026-05-01"
#: The fake GPX covers 14:00-20:00 UTC on DAY.
SPAN = (
    datetime(2026, 5, 1, 14, tzinfo=timezone.utc),
    datetime(2026, 5, 1, 20, tzinfo=timezone.utc),
)


class _Steps:
    """Stand-ins for the four Route Story steps; ``calls`` records the order."""

    def __init__(
        self,
        tmp_path: Path,
        *,
        bundle=True,
        verify_ok=True,
        film_rc=0,
        photo_times=None,
        probe_error=None,
    ):
        self.calls: list[str] = []
        #: bundles the 3D preflight probed, kept apart from ``calls`` (the step order).
        self.probed: list[Path] = []
        self._probe_error = probe_error
        self.film_argv: list[str] = []
        #: photo file name -> zoned capture instant (absent = no zoned time).
        self.photo_times: dict[str, datetime] = photo_times or {}
        self.bundle_dir = tmp_path / "bundle"
        self.bundle_dir.mkdir(exist_ok=True)
        self._bundle, self._verify_ok, self._film_rc = bundle, verify_ok, film_rc

    def steps(self) -> trail_film.TrailSteps:
        def find_bundle(gpx, *, recorded_on):
            self.calls.append(f"bundle:{gpx.name}:{recorded_on.isoformat()}")
            return self.bundle_dir if self._bundle else None

        def render_proof(gpx, bundle_root, output, *, recorded_on):
            self.calls.append("proof")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"proof")
            receipt = Path(str(output) + ".receipt.json")
            receipt.write_text("{}")
            return receipt

        def verify(receipt):
            self.calls.append("verify")
            report = VerifyReport(kind="proof")
            report.add("ok_check", self._verify_ok, "" if self._verify_ok else "sha mismatch")
            return report

        def film(argv):
            self.calls.append("film")
            self.film_argv = list(argv)
            out = Path(argv[argv.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"film")
            return self._film_rc

        def probe_3d(bundle_root):
            self.probed.append(bundle_root)
            if self._probe_error is not None:
                raise self._probe_error

        return trail_film.TrailSteps(
            find_bundle=find_bundle,
            probe_3d=probe_3d,
            render_proof=render_proof,
            verify=verify,
            film=film,
            gpx_span=lambda gpx, recorded_on: SPAN,
            capture_utc=lambda photo: self.photo_times.get(photo.name),
        )


def _profile(tmp_path: Path, photos_dir: "Path | None" = None, *, with_photos=True) -> Path:
    path = tmp_path / "trip.yaml"
    photos = photos_dir if photos_dir is not None else tmp_path / "photos"
    if photos_dir is None:
        photos.mkdir(exist_ok=True)
    path.write_text(
        'national-credit: "NASA"\nlocal-credit: "USGS"\nlocal-bounds: [-83.6, 35.2, -83.3, 35.3]\n'
        'national-raster: "/r/n.jpg"\nlocal-raster: "/r/l.jpg"\n'
        + (f'photos-dir: "{photos}"\n' if with_photos else "")
        + 'renderer: "scene"\n'
    )
    return path


def _day(tmp_path: Path, day: str = DAY, lfs: int = 1) -> Deliverables:
    final = tmp_path / f"day_{day}" / "final"
    final.mkdir(parents=True, exist_ok=True)
    paths = []
    for n in range(1, lfs + 1):
        p = final / f"lf_{n:02d}.mp4"
        p.write_bytes(b"lf")
        paths.append(p)
    return Deliverables(
        date=day,
        master_path=None,
        regular_lf_paths=paths,
        regular_lf_indices=list(range(1, lfs + 1)),
    )


def _state(tmp_path: Path, *dates: str):
    dates = dates or (DAY,)
    return SimpleNamespace(
        episode_dates=list(dates),
        gpx_files=[str(tmp_path / f"{d}.gpx") for d in dates],
        source_dir=str(tmp_path / "src"),
    )


def _cfg(
    tmp_path: Path, *, trail=None, profile=True, photos_dir: "Path | None" = None, set_profile=None
):
    path = set_profile or (_profile(tmp_path, photos_dir) if profile else "")
    return SimpleNamespace(drive=SimpleNamespace(trail=trail, trail_profile=str(path)))


def _argv_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _utc(hour: int, day: int = 1) -> datetime:
    return datetime(2026, 5, day, hour, tzinfo=timezone.utc)


def test_trail_film_gets_only_that_days_photos(tmp_path):
    pool = tmp_path / "all_photos"
    pool.mkdir()
    times = {
        "on_trail.jpg": _utc(15),
        "at_summit.NEF": _utc(19),
        "before_start.jpg": _utc(11),  # 3 h before: inside the film's bookend window
        "pre_dawn.jpg": _utc(2),  # 12 h before: a different part of the trip
        "next_day.jpg": _utc(15, day=2),
        "last_week.jpg": _utc(15) - timedelta(days=7),
    }
    for name in [*times, "undated.jpg", "notes.txt"]:
        (pool / name).write_bytes(name.encode())
    fake = _Steps(tmp_path, photo_times=times)
    d = _day(tmp_path)

    trail_film.attach_trail_film(
        d, _state(tmp_path), _cfg(tmp_path, photos_dir=pool), steps=fake.steps()
    )

    assert d.trail_skips == [] and len(d.trail_films) == 1
    staged = Path(_argv_value(fake.film_argv, "--photos-dir"))
    # The film is not pointed at the drive-wide folder, but at this day's slice of it.
    assert staged != pool
    assert staged == d.trail_films[0].path.parent / f"photos_{DAY}"
    assert sorted(p.name for p in staged.iterdir()) == [
        "at_summit.NEF",
        "before_start.jpg",
        "on_trail.jpg",
    ]
    assert (staged / "on_trail.jpg").read_bytes() == b"on_trail.jpg"


def test_trail_film_photo_pool_defaults_to_the_drive_source_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("mine.jpg", "other_day.jpg"):
        (src / name).write_bytes(b"x")
    fake = _Steps(tmp_path, photo_times={"mine.jpg": _utc(16), "other_day.jpg": _utc(16, day=3)})
    cfg = _cfg(tmp_path, set_profile=_profile(tmp_path, with_photos=False))
    d = _day(tmp_path)

    trail_film.attach_trail_film(d, _state(tmp_path), cfg, steps=fake.steps())

    staged = Path(_argv_value(fake.film_argv, "--photos-dir"))
    assert [p.name for p in staged.iterdir()] == ["mine.jpg"]


def test_trail_film_restages_on_a_retry_and_skips_a_missing_photo_dir(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "a.jpg").write_bytes(b"a")
    fake = _Steps(tmp_path, photo_times={"a.jpg": _utc(16)})
    d = _day(tmp_path)
    cfg = _cfg(tmp_path, photos_dir=pool)
    trail_film.attach_trail_film(d, _state(tmp_path), cfg, steps=fake.steps())
    stage = Path(_argv_value(fake.film_argv, "--photos-dir"))
    (stage / "stale.jpg").write_bytes(b"left by an earlier attempt")

    again = _day(tmp_path)
    trail_film.attach_trail_film(again, _state(tmp_path), cfg, steps=fake.steps())
    assert [p.name for p in stage.iterdir()] == ["a.jpg"]

    missing = _day(tmp_path, "2026-05-09")
    state = _state(tmp_path, "2026-05-09")
    trail_film.attach_trail_film(
        missing, state, _cfg(tmp_path, photos_dir=tmp_path / "gone"), steps=fake.steps()
    )
    assert missing.trail_films == [] and "not found" in missing.trail_skips[0]


def test_gpx_span_is_the_selected_days_first_and_last_fix(tmp_path):
    gpx = tmp_path / "day.gpx"
    pts = "".join(
        f'<trkpt lat="35.{i}" lon="-83.{i}"><time>2026-05-01T{h}:00:00Z</time></trkpt>'
        for i, h in enumerate(("14", "15", "20"))
    )
    gpx.write_text(
        '<gpx version="1.1" creator="t" xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        + pts
        + "</trkseg></trk></gpx>"
    )
    assert trail_film._gpx_span(gpx, date(2026, 5, 1)) == SPAN


def test_trail_skip_retried_on_resume(tmp_path):
    day = _day(tmp_path)
    reconstructed = DayArtifacts(
        date=DAY, master_path=None,
        regular_lf_paths=list(day.regular_lf_paths), regular_lf_indices=[1],
    )  # fmt: skip
    day.trail_skips = [f"trail film skipped for {DAY}: no prepared imagery bundle"]
    state = _state(tmp_path)
    state.per_day_completed = {
        DAY: {"master_path": None, "deliverables": deliverables_to_state(day)}
    }
    saves: list[int] = []

    def resume(fake, cfg):
        return phase_bc_day_first._resume_trail_films(
            state, cfg, [reconstructed], tmp_path, lambda st: saves.append(1), steps=fake.steps()
        )

    # Cause still present (no bundle): the recorded skip stands, nothing renders.
    still = _Steps(tmp_path, bundle=False)
    (d,) = resume(still, _cfg(tmp_path))
    assert still.calls == [f"bundle:{DAY}.gpx:{DAY}"] and saves == []
    assert d.trail_films == [] and "no prepared imagery bundle" in d.trail_skips[0]

    # Cause resolved: only the trail step runs for the day (no encode, no LF render).
    fixed = _Steps(tmp_path)
    (d,) = resume(fixed, _cfg(tmp_path))
    assert [c for c in fixed.calls if not c.startswith("bundle")] == ["proof", "verify", "film"]
    assert d.trail_skips == [] and [f.lf_index for f in d.trail_films] == [1]
    assert d.regular_lf_paths == day.regular_lf_paths  # the day itself is untouched
    record = state.per_day_completed[DAY]
    assert record["master_path"] is None  # the day's own entry is not rewritten
    assert record["deliverables"]["trail_skips"] == []
    assert record["deliverables"]["trail_films"][0]["path"] == str(d.trail_films[0].path)
    assert saves == [1]

    # The film is now recorded: a further resume restores it and renders nothing.
    settled = _Steps(tmp_path)
    (d,) = resume(settled, _cfg(tmp_path))
    assert settled.calls == [] and len(d.trail_films) == 1 and d.trail_skips == []


def test_resume_leaves_a_no_trail_day_and_a_vanished_film_alone(tmp_path):
    day = _day(tmp_path)
    day.trail_skips = ["trail film skipped: whatever"]
    day.trail_films = [TrailFilm(lf_index=1, path=tmp_path / "gone.mp4")]
    state = _state(tmp_path)
    state.per_day_completed = {
        DAY: {"master_path": None, "deliverables": deliverables_to_state(day)}
    }
    off = _Steps(tmp_path)
    (d,) = phase_bc_day_first._resume_trail_films(
        state, _cfg(tmp_path, trail=False), [day], tmp_path, None, steps=off.steps()
    )
    assert off.calls == []
    # The vanished film is not restored; the recorded skip is.
    assert d.trail_films == [] and d.trail_skips == ["trail film skipped: whatever"]


def test_drive_trail_renders_film_per_lf_day(tmp_path):
    fake = _Steps(tmp_path)
    state = _state(tmp_path, "2026-05-01", "2026-05-02")
    out = []
    for day in ("2026-05-01", "2026-05-02"):
        d = _day(tmp_path, day, lfs=2)
        trail_film.attach_trail_film(d, state, _cfg(tmp_path), steps=fake.steps())
        out.append(d)

    # Both days got their own film, in step order, from that day's GPX.
    assert fake.calls == [
        "bundle:2026-05-01.gpx:2026-05-01", "proof", "verify", "film",
        "bundle:2026-05-02.gpx:2026-05-02", "proof", "verify", "film",
    ]  # fmt: skip
    for d in out:
        assert d.trail_skips == []
        (film,) = d.trail_films
        assert isinstance(film, TrailFilm)
        assert film.path.is_file()
        assert film.path.parent.parent == d.regular_lf_paths[0].parent.parent
        # The LF is the placed source clip: the day's first LF, by index.
        assert film.lf_index == 1
    argv = fake.film_argv
    assert argv[argv.index("--source-video") + 1] == str(out[1].regular_lf_paths[0])
    assert argv[argv.index("--gpx") + 1].endswith("2026-05-02.gpx")
    assert argv[argv.index("--recorded-on") + 1] == "2026-05-02"
    assert argv[argv.index("--renderer") + 1] == "scene"
    assert argv[argv.index("--bundle") + 1] == str(fake.bundle_dir)
    # Imagery comes from the profile; the day's own values override it.
    assert argv[argv.index("--national-raster") + 1] == "/r/n.jpg"


def test_trail_preflight_refuses_without_3d_renderer(tmp_path, caplog):
    """No browser / no terrain tile: refuse before the proof or the film encode."""
    fake = _Steps(
        tmp_path, probe_error=TerrainRendererUnavailableError("playwright is not installed")
    )
    d = _day(tmp_path)
    with caplog.at_level(logging.WARNING):
        trail_film.attach_trail_film(
            d, _state(tmp_path), _cfg(tmp_path, trail=True), steps=fake.steps()
        )

    assert d.trail_films == []
    assert fake.probed == [fake.bundle_dir]  # probed the day's own bundle
    assert fake.calls == [f"bundle:{DAY}.gpx:{DAY}"]  # no proof, verify or film ran
    (reason,) = d.trail_skips
    assert "3D terrain renderer" in reason and "route-story-3d" in reason
    assert "playwright is not installed" in reason
    # --trail asked for the film: the refusal is an error, not a warning.
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_trail_preflight_runs_once_before_encode_when_renderer_works(tmp_path):
    fake = _Steps(tmp_path)
    d = _day(tmp_path)
    trail_film.attach_trail_film(d, _state(tmp_path), _cfg(tmp_path), steps=fake.steps())
    assert fake.probed == [fake.bundle_dir]
    assert len(d.trail_films) == 1


def test_no_trail_does_not_probe_the_renderer(tmp_path):
    fake = _Steps(tmp_path)
    d = _day(tmp_path)
    trail_film.attach_trail_film(
        d, _state(tmp_path), _cfg(tmp_path, trail=False), steps=fake.steps()
    )
    assert fake.probed == []


def test_probe_3d_renderer_wraps_any_launch_failure(tmp_path):
    """The real probe: an injected capture factory stands in for the browser."""

    class _Boom:
        def __enter__(self):
            raise RuntimeError("chromium missing")

        def __exit__(self, *exc):
            return False

    with pytest.raises(TerrainRendererUnavailableError, match="chromium missing"):
        probe_3d_renderer(tmp_path, capture_factory=lambda root, w, h: _Boom())

    entered = []

    class _Ok:
        def __enter__(self):
            entered.append(1)
            return self

        def __exit__(self, *exc):
            return False

    probe_3d_renderer(tmp_path, capture_factory=lambda root, w, h: _Ok())
    assert entered == [1]


def test_drive_trail_skips_loudly_without_bundle(tmp_path, caplog):
    fake = _Steps(tmp_path, bundle=False)
    d = _day(tmp_path)
    with caplog.at_level(logging.WARNING):
        trail_film.attach_trail_film(d, _state(tmp_path), _cfg(tmp_path), steps=fake.steps())

    assert d.trail_films == []
    assert fake.calls == [f"bundle:{DAY}.gpx:{DAY}"]  # nothing past the lookup ran
    (reason,) = d.trail_skips
    assert DAY in reason and "route-story-prepare" in reason
    loud = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("route-story-prepare" in r.getMessage() for r in loud)


def test_trail_skips_loudly_without_gpx(tmp_path, caplog):
    fake = _Steps(tmp_path)
    d = _day(tmp_path)
    state = SimpleNamespace(episode_dates=[DAY], gpx_files=[None], source_dir=str(tmp_path))
    with caplog.at_level(logging.WARNING):
        trail_film.attach_trail_film(d, state, _cfg(tmp_path), steps=fake.steps())
    assert fake.calls == []
    assert "no GPX" in d.trail_skips[0]
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_no_trail_flag_is_silent_and_runs_nothing(tmp_path, caplog):
    fake = _Steps(tmp_path)
    d = _day(tmp_path)
    with caplog.at_level(logging.WARNING):
        trail_film.attach_trail_film(
            d, _state(tmp_path), _cfg(tmp_path, trail=False), steps=fake.steps()
        )
    assert fake.calls == [] and d.trail_films == [] and d.trail_skips == []
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_trail_skips_without_profile(tmp_path):
    fake = _Steps(tmp_path)
    d = _day(tmp_path)
    trail_film.attach_trail_film(
        d, _state(tmp_path), _cfg(tmp_path, profile=False), steps=fake.steps()
    )
    assert "film.main" not in fake.calls and "film" not in fake.calls
    assert "--trail-profile" in d.trail_skips[0]


def test_failed_verify_withholds_the_film(tmp_path):
    fake = _Steps(tmp_path, verify_ok=False)
    d = _day(tmp_path)
    trail_film.attach_trail_film(d, _state(tmp_path), _cfg(tmp_path), steps=fake.steps())
    assert fake.calls[-1] == "verify"
    assert d.trail_films == []
    assert "ok_check" in d.trail_skips[0] and "sha mismatch" in d.trail_skips[0]


def test_failed_film_is_recorded_not_raised(tmp_path):
    fake = _Steps(tmp_path, film_rc=3)
    d = _day(tmp_path)
    trail_film.attach_trail_film(d, _state(tmp_path), _cfg(tmp_path), steps=fake.steps())
    assert d.trail_films == []
    assert "exit 3" in d.trail_skips[0]


def test_step_exception_is_recorded_not_raised(tmp_path):
    fake = _Steps(tmp_path)
    steps = fake.steps()

    def boom(*a, **k):
        raise RuntimeError("no browser")

    steps = trail_film.TrailSteps(
        find_bundle=steps.find_bundle,
        render_proof=boom,
        verify=steps.verify,
        film=steps.film,
        probe_3d=steps.probe_3d,
    )
    d = _day(tmp_path)
    trail_film.attach_trail_film(d, _state(tmp_path), _cfg(tmp_path), steps=steps)
    assert "no browser" in d.trail_skips[0]


def test_trail_outcome_round_trips_through_state(tmp_path):
    d = _day(tmp_path)
    d.trail_films = [TrailFilm(lf_index=1, path=tmp_path / "t.mp4", receipt=tmp_path / "r.json")]
    d.trail_skips = ["x"]
    back = media_api.deliverables_from_state(media_api.deliverables_to_state(d))
    assert back.trail_films == d.trail_films
    assert back.trail_skips == ["x"]
    # A record written before trail films has neither key.
    legacy = media_api.deliverables_to_state(_day(tmp_path))
    legacy.pop("trail_films"), legacy.pop("trail_skips")
    assert media_api.deliverables_from_state(legacy).trail_films == []


def test_drive_parser_trail_flags():
    parse = _build_parser().parse_args
    base = ["drive", "/src"]
    assert parse(base).trail is None
    assert parse([*base, "--trail"]).trail is True
    assert parse([*base, "--no-trail"]).trail is False
    assert parse([*base, "--trail-profile", "p.yaml"]).trail_profile == "p.yaml"


def test_phase_bc_attaches_trail_film_before_recording_the_day():
    src = inspect.getsource(phase_bc_day_first)
    body = src[src.index("def _on_day_complete") :]
    body = body[: body.index("# Build per-date GPS")]
    assert body.index("attach_trail_film(") < body.index("_record_day_and_save(")


def test_proof_with_receipt_writes_receipt_beside_the_clip(tmp_path, monkeypatch):
    seen = {}

    def fake_render(gpx, bundle, output, *, recorded_on, fps, width, height, workers):
        seen.update(width=width, workers=workers, recorded_on=recorded_on)
        return {"schema": "route-story-proof-receipt/2"}

    monkeypatch.setattr(proof, "render_proof_clip", fake_render)
    out = tmp_path / "p.mp4"
    receipt = proof.render_proof_with_receipt(
        tmp_path / "t.gpx", tmp_path, out, recorded_on=date(2026, 5, 1), fps=30.0,
        width=640, height=360, workers=2,
    )  # fmt: skip
    assert receipt == Path(str(out) + ".receipt.json")
    assert receipt.is_file() and seen["width"] == 640 and seen["workers"] == 2


@pytest.mark.parametrize("name", ["trail_film"])
def test_trail_film_module_stays_off_publishing(name):
    text = Path(inspect.getsourcefile(trail_film)).read_text()
    assert "gomoveshift.cli" not in text and "from ...cli" not in text
