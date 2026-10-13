"""Route Story runs inside a drive as a sibling deliverable.

``attach_trail_film`` takes a day's ``Deliverables`` and, when the day has a
GPX and a prepared imagery bundle, runs prepare-lookup -> proof -> verify ->
film with the LF as the placed source clip. Anything missing is a loud, recorded
skip -- never silence, never a failed drive.
"""

from __future__ import annotations

import inspect
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.cli.parser import _build_parser
from gomoveshift.pipeline import media_api
from gomoveshift.pipeline.drive_phases import phase_bc_day_first, trail_film
from gomoveshift.pipeline.media_api import Deliverables, TrailFilm
from gomoveshift.pipeline.route_story import proof
from gomoveshift.pipeline.route_story.verify import VerifyReport

DAY = "2026-05-01"


class _Steps:
    """Stand-ins for the four Route Story steps; ``calls`` records the order."""

    def __init__(self, tmp_path: Path, *, bundle=True, verify_ok=True, film_rc=0):
        self.calls: list[str] = []
        self.film_argv: list[str] = []
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

        return trail_film.TrailSteps(
            find_bundle=find_bundle, render_proof=render_proof, verify=verify, film=film
        )


def _profile(tmp_path: Path) -> Path:
    path = tmp_path / "trip.yaml"
    path.write_text(
        'national-credit: "NASA"\nlocal-credit: "USGS"\nlocal-bounds: [-83.6, 35.2, -83.3, 35.3]\n'
        'national-raster: "/r/n.jpg"\nlocal-raster: "/r/l.jpg"\n'
        'photos-dir: "/p"\nrenderer: "scene"\n'
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


def _cfg(tmp_path: Path, *, trail=None, profile=True):
    return SimpleNamespace(
        drive=SimpleNamespace(trail=trail, trail_profile=str(_profile(tmp_path)) if profile else "")
    )


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
        find_bundle=steps.find_bundle, render_proof=boom, verify=steps.verify, film=steps.film
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
