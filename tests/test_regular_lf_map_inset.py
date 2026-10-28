"""Opt-in ``--map-inset route`` on regular long-form videos.

Off is today's path (same ``apply_ost`` kwargs, same OST fingerprint, no state).
On renders a route-only inset clip per LF from an operator-prepared offline
bundle and composites it inside the OST encode. Insufficient inputs are a
reported refusal BEFORE any encode, never a silent fallback to off.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from _state_helpers import make_state

from gomoveshift.pipeline import cleanup, day_pipeline
from gomoveshift.pipeline.map_inset import InsetReceipt, MapInsetRefused
from gomoveshift.pipeline.ost_applier import MapInsetOverlay
from gomoveshift.pipeline.route_claims import RouteClaimRefusedError
from gomoveshift.pipeline.route_story.route_privacy import PrivacyZone, RouteTrim

DATE = "2026-02-20"
FRAME = (1920, 1080)
RECT = (38, 800, 384, 288)


def _keycut_receipt(spans=((0.0, 60.0),)) -> dict:
    out, rows = 0.0, []
    for i, (s, e) in enumerate(spans):
        rows.append(
            {
                "index": i,
                "requested_indices": [i],
                "merged_start": s,
                "merged_end": e,
                "source_start": s,
                "source_end": e,
                "method": "reencode",
                "reason": "keyframes_unknown",
                "output_start": out,
                "output_end": out + (e - s),
                "duration_source": "planned",
            }
        )
        out += e - s
    return {
        "schema": "keycut-assembly-receipt/1",
        "provenance": "planned",
        "keycut_version": None,
        "master": {"path": "m.mp4", "size_bytes": None},
        "time_base": "seconds",
        "max_gap_sec": 2.0,
        "requested": [[s, e] for s, e in spans],
        "spans": rows,
        "output": {"path": "o.mp4", "duration": out},
    }


def _inset_receipt(wall: float = 1.5) -> InsetReceipt:
    return InsetReceipt(
        bundle_manifest_sha256="ab" * 32,
        privacy={"route_trim": {"start_m": 150.0, "end_m": 150.0}},
        frame_size=FRAME,
        inset_rect=RECT,
        fps=10.0,
        frame_count=750,
        hidden_frames={"insert": 150, "crossfade": 0, "gap": 0},
        credits=("Test Imagery Agency",),
        wall_time_s=wall,
    )


def _drive(**kw) -> SimpleNamespace:
    base = dict(
        brand_logo_path=None,
        shorts_target_count=9,
        shorts_min_gap_sec=60.0,
        skip_audio=False,
        montage_max_duration_sec=15.0,
        gps_captions="off",
        regular_lf_lower_third_duration_sec=8.0,
        regular_lf_lower_third_min_gap_sec=45.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _on(bundle: Path, **kw) -> SimpleNamespace:
    opts = dict(
        map_inset="route",
        map_inset_bundle=str(bundle),
        map_inset_trim_start_m=150.0,
        map_inset_trim_end_m=150.0,
        map_inset_privacy_zones=[],
    )
    opts.update(kw)
    return _drive(**opts)


def _gps(spans=((0.0, 60.0),)) -> dict:
    """``{date: [(t_master, lat, lon)]}``, a fix every 2 s inside each span."""
    pts = []
    for lo, hi in spans:
        t = lo
        while t <= hi:
            pts.append((t, 35.2 + 0.0002 * t, -83.6 + 0.0004 * t))
            t += 2.0
    return {DATE: pts}


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_text('{"x": 1}', encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _bundle_verifies():
    with patch("gomoveshift.pipeline.route_story.bundle.verify_bundle", return_value=[]) as v:
        yield v


def _probe(path, *a, **k):
    return SimpleNamespace(duration=315.0 if "wm" in str(path) else 300.0)


class Run:
    """One run_day_pipeline call with every heavy step stubbed."""

    def __init__(self):
        self.apply_calls: list[dict] = []
        self.render_calls: list[dict] = []
        self.extract_calls = 0
        self.encode_calls = 0
        self.state = None
        self.clip_at_burn = None
        self.extract_kwargs: dict = {}
        self.error: BaseException | None = None
        self.spliced = 0


def _run(
    tmp_path,
    drive,
    *,
    gps="default",
    sidecar_on_call=1,
    render_error=None,
    receipt=None,
    state_kw=None,
    inserts_by_lf=None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ep = tmp_path / "ep.mp4"
    ep.write_bytes(b"\x00")
    for n in ("master", "seg", "wm"):
        (tmp_path / f"{n}.mp4").write_bytes(b"\x00")
    lf_plan = SimpleNamespace(
        index=1,
        segments=[SimpleNamespace(source_file_index=0, start=0.0, end=60.0)],
        montage_peaks=[],
    )
    day_batch = SimpleNamespace(
        date=DATE, episodes=[SimpleNamespace(index=1, concat_path=ep, total_duration=300.0)]
    )
    state = make_state(ost_audit_confirmed=True, **(state_kw or {}))
    run = Run()
    run.state = state

    def fake_extract(**kwargs):
        run.extract_calls += 1
        run.extract_kwargs = kwargs
        sidecar = day_pipeline.receipt_sidecar_path(kwargs["output_path"])
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.unlink(missing_ok=True)
        if kwargs.get("receipt_sidecar") and run.extract_calls >= sidecar_on_call:
            sidecar.write_text(json.dumps(_keycut_receipt()), encoding="utf-8")
        return tmp_path / "seg.mp4"

    def fake_apply_ost(**kwargs):
        run.apply_calls.append(kwargs)
        if "map_inset" in kwargs:
            run.clip_at_burn = kwargs["map_inset"].clip_path.exists()
        kwargs["output_path"].write_bytes(b"\x00")
        return kwargs["output_path"]

    def fake_render(bundle, segments, timeline, output, frame_size, **options):
        run.render_calls.append(
            dict(
                bundle=bundle,
                segments=segments,
                timeline=timeline,
                output=output,
                frame_size=frame_size,
                options=options,
            )
        )
        if render_error is not None:
            raise render_error
        Path(output).write_bytes(b"\x00")
        return receipt or _inset_receipt()

    gps_by_date = _gps() if gps == "default" else gps
    with (
        patch.object(
            day_pipeline,
            "encode_day_master",
            side_effect=lambda *a, **k: (
                setattr(run, "encode_calls", run.encode_calls + 1) or tmp_path / "master.mp4"
            ),
        ),
        patch.object(day_pipeline, "extract_and_concat_segments", side_effect=fake_extract),
        patch.object(day_pipeline, "prepend_montage", return_value=tmp_path / "wm.mp4"),
        patch.object(day_pipeline, "apply_ost", side_effect=fake_apply_ost),
        patch.object(day_pipeline, "render_route_inset", side_effect=fake_render),
        patch.object(day_pipeline, "get_video_dimensions", return_value=FRAME),
        patch.object(day_pipeline, "plan_shorts", return_value=[]),
        patch.object(day_pipeline, "compute_day_file_boundaries", return_value=[(0.0, 300.0)]),
        patch.object(day_pipeline, "probe_media", side_effect=_probe),
        contextlib.ExitStack() as stack,
    ):
        if inserts_by_lf:  # placeholder inserts: record the splice, keep no records
            stack.enter_context(patch.object(day_pipeline, "insert_records", return_value={}))
            stack.enter_context(
                patch.object(
                    day_pipeline,
                    "splice_day_lf_inserts",
                    side_effect=lambda *a, **k: setattr(run, "spliced", run.spliced + 1),
                )
            )
        try:
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=None,
                output_dir=tmp_path / "out",
                location="Alphaville",
                config=SimpleNamespace(drive=drive),
                lf_plans=[lf_plan],
                gps_points_by_date=gps_by_date,
                state=state,
                inserts_by_lf=inserts_by_lf,
            )
        except BaseException as exc:  # noqa: BLE001 -- the tests inspect it
            run.error = exc
    run.apply_calls = [c for c in run.apply_calls if c.get("layout") == "regular_lf"]
    return run


def _fp(tmp_path) -> str:
    return (tmp_path / "out" / f"day_{DATE}" / "regular_lfs" / "lf_01.ost.sha256").read_text()


# -- off is today's path ---------------------------------------------------------------


def test_off_matches_a_config_that_predates_the_flag(tmp_path):
    legacy = _run(tmp_path / "a", _drive())  # no map_inset attributes at all
    off = _run(tmp_path / "b", _drive(map_inset="off"))
    assert legacy.error is None and off.error is None
    assert len(off.apply_calls) == 1
    assert "map_inset" not in off.apply_calls[0]
    assert off.apply_calls[0].keys() == legacy.apply_calls[0].keys()
    assert _fp(tmp_path / "a") == _fp(tmp_path / "b")
    assert off.render_calls == []
    assert off.state.get("regular_lf_map_inset") is None


def test_off_does_not_ask_the_extractor_for_a_sidecar(tmp_path):
    run = _run(tmp_path, _drive())
    assert run.error is None and run.extract_calls == 1
    assert "receipt_sidecar" not in run.extract_kwargs


# -- refusals before any encode --------------------------------------------------------


def test_route_without_a_bundle_refuses_before_encode(tmp_path):
    run = _run(tmp_path, _on(tmp_path, map_inset_bundle=None))
    assert isinstance(run.error, RouteClaimRefusedError)
    assert "no --map-inset-bundle" in str(run.error)
    assert "nothing was encoded" in str(run.error)
    assert run.encode_calls == 0


def test_bundle_that_fails_verification_refuses_before_encode(tmp_path, bundle, _bundle_verifies):
    _bundle_verifies.return_value = ["imagery/10/1/2.jpg is missing"]
    run = _run(tmp_path, _on(bundle))
    assert isinstance(run.error, RouteClaimRefusedError)
    assert "imagery/10/1/2.jpg is missing" in str(run.error)
    assert run.encode_calls == 0


def test_missing_bundle_directory_refuses(tmp_path):
    run = _run(tmp_path, _on(tmp_path / "nope"))
    assert isinstance(run.error, RouteClaimRefusedError) and run.encode_calls == 0
    assert "manifest.json" in str(run.error)


@pytest.mark.parametrize(
    "start, end", [(0.0, 150.0), (150.0, 0.0), (0.0, 0.0), (-5.0, 150.0)], ids=str
)
def test_untrimmed_endpoint_refuses_before_encode(tmp_path, bundle, start, end):
    run = _run(tmp_path, _on(bundle, map_inset_trim_start_m=start, map_inset_trim_end_m=end))
    assert isinstance(run.error, RouteClaimRefusedError) and run.encode_calls == 0
    assert "trim" in str(run.error)


@pytest.mark.parametrize("gps", [None, {}, {DATE: []}, {"2026-01-01": [(0.0, 1.0, 2.0)]}], ids=str)
def test_day_without_a_usable_gps_trace_refuses_before_encode(tmp_path, bundle, gps):
    run = _run(tmp_path, _on(bundle), gps=gps)
    assert isinstance(run.error, RouteClaimRefusedError) and run.encode_calls == 0
    assert "GPS" in str(run.error)


def test_every_reason_is_listed_in_one_error(tmp_path):
    run = _run(
        tmp_path,
        _on(tmp_path, map_inset_bundle=None, map_inset_trim_start_m=0.0, map_inset_trim_end_m=0.0),
        gps=None,
    )
    msg = str(run.error)
    assert "no --map-inset-bundle" in msg and "trim" in msg and "GPS" in msg
    assert run.encode_calls == 0 and run.apply_calls == [] and run.render_calls == []


def test_only_fixes_with_valid_coordinates_count_as_a_trace(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle), gps={DATE: [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]})
    assert isinstance(run.error, RouteClaimRefusedError) and run.encode_calls == 0


# -- on: render, composite, record -----------------------------------------------------


def test_on_renders_one_clip_and_composites_it_at_the_receipt_rect(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle, map_inset_privacy_zones=["35.2,-83.6,200"]))
    assert run.error is None
    (call,) = run.render_calls
    assert call["frame_size"] == FRAME
    assert Path(call["bundle"]) == bundle
    assert call["output"].name == "lf_01_map_inset.mov"
    assert call["options"]["trim"] == RouteTrim(start_m=150.0, end_m=150.0)
    assert call["options"]["zones"] == [PrivacyZone(35.2, -83.6, 200.0)]
    assert call["options"]["workdir"] is not None
    (apply_call,) = run.apply_calls
    overlay = apply_call["map_inset"]
    assert isinstance(overlay, MapInsetOverlay)
    assert overlay.clip_path == call["output"]
    assert (overlay.x, overlay.y) == RECT[:2]


def test_timeline_is_the_keycut_receipt_shifted_by_the_probed_montage_lead(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle))
    timeline = run.render_calls[0]["timeline"]
    # lead = 315 - 300 = 15 s of montage with no source, then the 60 s receipt span
    assert timeline.duration == pytest.approx(75.0)
    (pos,) = timeline.output_to_source(15.5)
    assert pos.time == pytest.approx(0.5)
    assert not timeline.output_to_source(5.0)


def test_route_points_are_master_seconds_split_at_gps_gaps(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle), gps=_gps(spans=((0.0, 20.0), (40.0, 60.0))))
    segments = run.render_calls[0]["segments"]
    assert [len(s) for s in segments] == [11, 11]
    assert segments[0][0].time_s == 0.0 and segments[1][0].time_s == 40.0
    assert segments[1][0].latitude == pytest.approx(35.2 + 0.0002 * 40.0)


def test_on_records_the_receipt_for_review(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle))
    rec = run.state.get("regular_lf_map_inset")[DATE]["1"]
    assert rec == _inset_receipt().to_dict()


def test_the_clip_exists_when_the_ost_burn_composites_it(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle))
    assert run.error is None and run.clip_at_burn is True
    assert run.render_calls[0]["output"].parent == run.apply_calls[0]["output_path"].parent


def test_photo_inserts_keep_the_map_inset(tmp_path, bundle):
    # Stills are spliced into the burned LF, so the map travels with its frames
    # and the inserted stills simply carry none: inserts never block an inset.
    run = _run(tmp_path, _on(bundle), inserts_by_lf={1: [object()]})
    assert run.error is None and len(run.render_calls) == 1
    assert isinstance(run.apply_calls[0]["map_inset"], MapInsetOverlay)
    assert run.spliced == 1  # the inserts still went in, after the burn


def test_inset_refusal_stops_the_render_naming_the_lf(tmp_path, bundle):
    run = _run(
        tmp_path,
        _on(bundle),
        render_error=MapInsetRefused(["marker coverage 0.10 is below the minimum 0.50"]),
    )
    assert isinstance(run.error, RuntimeError)
    assert "LF 1" in str(run.error) and "marker coverage 0.10" in str(run.error)
    assert run.apply_calls == []  # never shipped without its map


# -- fingerprint -----------------------------------------------------------------------


def test_fingerprint_differs_from_off_and_follows_every_option(tmp_path, bundle):
    off = _run(tmp_path / "off", _drive())
    base = _run(tmp_path / "base", _on(bundle))
    again = _run(tmp_path / "again", _on(bundle))
    trim = _run(tmp_path / "trim", _on(bundle, map_inset_trim_start_m=300.0))
    zone = _run(tmp_path / "zone", _on(bundle, map_inset_privacy_zones=["35.2,-83.6,200"]))
    route = _run(tmp_path / "route", _on(bundle), gps=_gps(spans=((0.0, 30.0), (36.0, 60.0))))
    for r in (off, base, again, trim, zone, route):
        assert r.error is None
    fps = {k: _fp(tmp_path / k) for k in ("off", "base", "again", "trim", "zone", "route")}
    assert fps["base"] == fps["again"]
    assert len({fps[k] for k in ("off", "base", "trim", "zone", "route")}) == 5


def test_changed_bundle_content_re_burns(tmp_path, bundle):
    first = _run(tmp_path, _on(bundle))
    (bundle / "manifest.json").write_text('{"x": 2}', encoding="utf-8")
    second = _run(tmp_path, _on(bundle))
    assert len(first.apply_calls) == 1 and len(second.apply_calls) == 1


def test_an_unchanged_resume_keeps_the_burned_final(tmp_path, bundle):
    first = _run(tmp_path, _on(bundle))
    second = _run(tmp_path, _on(bundle))
    assert len(first.apply_calls) == 1 and second.apply_calls == []
    assert second.render_calls == []


def test_toggling_off_after_on_re_burns(tmp_path, bundle):
    _run(tmp_path, _on(bundle))
    off = _run(tmp_path, _drive())
    assert len(off.apply_calls) == 1 and "map_inset" not in off.apply_calls[0]


# -- the per-LF receipt sidecar --------------------------------------------------------


def test_on_asks_the_extractor_for_the_receipt_sidecar(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle))
    assert run.error is None and run.extract_calls == 1


def test_missing_sidecar_re_extracts_the_concat_once(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle), sidecar_on_call=2)
    assert run.error is None
    assert run.extract_calls == 2
    assert len(run.apply_calls) == 1 and "map_inset" in run.apply_calls[0]


def test_sidecar_that_never_appears_stops_the_render(tmp_path, bundle):
    run = _run(tmp_path, _on(bundle), sidecar_on_call=99)
    assert isinstance(run.error, RuntimeError) and "LF 1" in str(run.error)
    assert run.extract_calls == 2 and run.apply_calls == [] and run.render_calls == []


def _cut_stub(out: Path, receipt: dict):
    class _R:
        def to_dict(self):
            return receipt

    def cut(*a, **kw):
        out.write_bytes(b"concat")
        return out, _R()

    return cut


def _segment():
    return type("S", (), {"source_file_index": 0, "start_time": 1.0, "end_time": 5.0})()


def test_extractor_writes_the_receipt_beside_the_concat_atomically(tmp_path, monkeypatch):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"m")
    out = tmp_path / "lf_01_segments.mp4"
    receipt = _keycut_receipt(((1.0, 5.0),))
    monkeypatch.setattr(
        day_pipeline, "extract_and_concat_ranges_with_receipt", _cut_stub(out, receipt)
    )
    result = day_pipeline.extract_and_concat_segments(
        master, [_segment()], [(0.0, 100.0)], out, tmp_path / "work", receipt_sidecar=True
    )
    assert result == out
    sidecar = tmp_path / "lf_01_segments.mp4.receipt.json"
    assert json.loads(sidecar.read_text()) == receipt
    assert sorted(p.name for p in tmp_path.iterdir() if "receipt" in p.name) == [sidecar.name]


def test_extractor_writes_no_sidecar_unless_asked(tmp_path, monkeypatch):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"m")
    out = tmp_path / "lf_01_segments.mp4"
    monkeypatch.setattr(
        day_pipeline,
        "extract_and_concat_ranges_with_receipt",
        _cut_stub(out, _keycut_receipt(((1.0, 5.0),))),
    )
    day_pipeline.extract_and_concat_segments(
        master, [_segment()], [(0.0, 100.0)], out, tmp_path / "work"
    )
    assert not list(tmp_path.glob("*.receipt.json"))


def test_a_failed_cut_or_bad_receipt_leaves_no_sidecar(tmp_path, monkeypatch):
    master = tmp_path / "master.mp4"
    master.write_bytes(b"m")
    out = tmp_path / "lf_01_segments.mp4"
    stale = tmp_path / "lf_01_segments.mp4.receipt.json"
    stale.write_text("{}")
    bad = _keycut_receipt(((1.0, 5.0),))
    bad["spans"][0]["method"] = "teleport"
    monkeypatch.setattr(day_pipeline, "extract_and_concat_ranges_with_receipt", _cut_stub(out, bad))
    assert (
        day_pipeline.extract_and_concat_segments(
            master, [_segment()], [(0.0, 100.0)], out, tmp_path / "work", receipt_sidecar=True
        )
        is None
    )
    assert not out.exists() and not stale.exists()

    stale.write_text("{}")
    monkeypatch.setattr(
        day_pipeline,
        "extract_and_concat_ranges_with_receipt",
        lambda *a, **k: (None, None),
    )
    assert (
        day_pipeline.extract_and_concat_segments(
            master, [_segment()], [(0.0, 100.0)], out, tmp_path / "work", receipt_sidecar=True
        )
        is None
    )
    assert not stale.exists()


def test_purge_removes_the_sidecar_and_the_inset_clip(tmp_path):
    lf_dir = tmp_path / "day" / "regular_lfs"
    lf_dir.mkdir(parents=True)
    (lf_dir / "lf_01.mp4").write_bytes(b"final")
    names = [
        "lf_01_segments.mp4",
        "lf_01_segments.mp4.receipt.json",
        "lf_01_map_inset.mov",
        "lf_01_map_inset.mov.receipt.json",
    ]
    for n in names:
        (lf_dir / n).write_bytes(b"x")
    cleanup.purge_lf_intermediates(tmp_path / "day", 1)
    assert sorted(p.name for p in lf_dir.iterdir()) == ["lf_01.mp4"]


# -- real ffmpeg: render, composite, keep ten bits ---------------------------------------

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


def _can_main10() -> bool:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return False
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=128x128:d=0.1", "-pix_fmt",
         "yuv420p10le", "-c:v", "libx265", "-f", "null", "-"],
        capture_output=True,
    )  # fmt: skip
    return r.returncode == 0


@pytest.mark.skipif(not _can_main10(), reason="ffmpeg/libx265 main10 not available")
def test_rendered_clip_composites_through_apply_ost_at_ten_bits(tmp_path, monkeypatch):
    from test_map_inset import _make_bundle, _route

    from gomoveshift.pipeline import ost_applier
    from gomoveshift.pipeline.edit_timeline import EditTimeline, load_receipt
    from gomoveshift.pipeline.map_inset import render_route_inset

    monkeypatch.setattr("gomoveshift.pipeline.color.detect_nvenc", lambda: False)
    bundle = _make_bundle(tmp_path / "bundle")
    w, h = 1280, 360
    src = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=gray:s={w}x{h}:r=30:d=2",
         "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le", str(src)],
        check=True, capture_output=True,
    )  # fmt: skip
    timeline = EditTimeline.from_receipt(load_receipt(_keycut_receipt(((10.0, 12.0),))))
    clip = tmp_path / "lf_01_map_inset.mov"
    receipt = render_route_inset(
        bundle, [_route()], timeline, clip, (w, h), workdir=tmp_path / "work",
        trim=RouteTrim(start_m=50.0, end_m=50.0), fps=10.0,
    )  # fmt: skip
    x, y = receipt.inset_rect[:2]
    out = tmp_path / "out.mp4"
    result = ost_applier.apply_ost(
        src, out, "Longwood", "regular_lf", prefer_nvenc=False,
        map_inset=MapInsetOverlay(clip_path=clip, x=x, y=y),
    )  # fmt: skip
    assert result == out and out.exists()
    pix = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert pix == "yuv420p10le"
