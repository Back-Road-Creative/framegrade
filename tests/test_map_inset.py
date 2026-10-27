"""Route-only map inset: privacy first, one fixed framing, marker per output frame."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline import map_inset
from gomoveshift.pipeline.edit_timeline import EditTimeline, Insert, load_receipt
from gomoveshift.pipeline.map_inset import (
    InsetReceipt,
    MapInsetRefused,
    marker_state,
    plan_route_inset,
    render_inset_frame,
    render_route_inset,
)
from gomoveshift.pipeline.route_story import bundle as bundle_mod
from gomoveshift.pipeline.route_story.metrics import RoutePoint, distance_m
from gomoveshift.pipeline.route_story.route_privacy import PrivacyZone, RouteTrim

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)

CREDIT = "Test Imagery Agency"
MARKER_RGB = (0xF5, 0xB7, 0x3B)
FRAME = (1920, 1080)


def _make_bundle(root: Path, *, labels: list[dict] | None = None) -> Path:
    """A 3x3 block of z10 imagery tiles around lon -83.5 / lat 35.2, dark so text stands out."""
    x0, y0 = (int(v) for v in bundle_mod.lonlat_to_tile(-83.5, 35.2, 10))
    rng = np.random.default_rng(7)
    for x in range(x0 - 1, x0 + 2):
        for y in range(y0 - 1, y0 + 2):
            pixels = rng.integers(20, 90, size=(256, 256, 3), dtype=np.uint8)
            path = root / "imagery" / "10" / str(x) / f"{y}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels, "RGB").save(path, quality=95)
    if labels is not None:
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item["lon"], item["lat"]]},
                "properties": {"name": item["name"], "kind": "town"},
            }
            for item in labels
        ]
        (root / "labels.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
        )
    bundle_mod.write_manifest(
        root,
        {
            "imagery": {
                "bands": [
                    {
                        "source": "test_band",
                        "provenance": {"attribution": CREDIT},
                        "written_zoom_range": {"minzoom": 10, "maxzoom": 10},
                    }
                ]
            }
        },
    )
    assert bundle_mod.verify_bundle(root) == []
    return root


def _point(t: float, lon0: float = -83.60, lat0: float = 35.20) -> RoutePoint:
    return RoutePoint(time_s=t, latitude=lat0 + 0.0002 * t, longitude=lon0 + 0.0004 * t)


def _route(start: float = 0.0, end: float = 100.0, step: float = 2.0, **kw) -> list[RoutePoint]:
    count = int(round((end - start) / step))
    return [_point(start + step * i, **kw) for i in range(count + 1)]


def _receipt(spans):
    out = 0.0
    rows = []
    for i, (s, e) in enumerate(spans):
        rows.append(
            {
                "index": i,
                "requested_indices": [0],
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
    return load_receipt(
        {
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
    )


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    return _make_bundle(tmp_path / "bundle")


def _plan(bundle: Path, tmp_path: Path, segments, timeline, **kw):
    return plan_route_inset(bundle, segments, timeline, FRAME, workdir=tmp_path / "work", **kw)


def _digest(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _pixel_at(plan, frame: Image.Image, lat: float, lon: float):
    x, y = plan.project(lat, lon)
    return frame.getpixel((round(x), round(y)))[:3]


def test_cut_marker_follows_source_time_not_output_time(bundle, tmp_path):
    timeline = EditTimeline.from_segments([(0.0, 30.0), (50.0, 80.0)])
    plan = _plan(bundle, tmp_path, [_route()], timeline)
    state = marker_state(plan, 30.0)
    assert state.visible
    assert state.source_time_s == pytest.approx(50.0)
    at_50, at_30 = _point(50.0), _point(30.0)
    assert state.latitude == pytest.approx(at_50.latitude)
    assert state.longitude == pytest.approx(at_50.longitude)
    frame = render_inset_frame(plan, 30.0)
    assert _pixel_at(plan, frame, at_50.latitude, at_50.longitude) == MARKER_RGB
    assert _pixel_at(plan, frame, at_30.latitude, at_30.longitude) != MARKER_RGB


def test_insert_crossfade_and_gap_hide_the_marker_and_are_counted(bundle, tmp_path):
    # insert hold: output 10..13 has no source
    insert = EditTimeline.from_receipt(_receipt([(0.0, 30.0)]), inserts=[Insert(10.0, 3.0)])
    plan = _plan(bundle, tmp_path, [_route()], insert)
    assert marker_state(plan, 11.0).hidden_reason == "insert"
    assert render_inset_frame(plan, 11.0).tobytes() == plan.base.tobytes()
    assert plan.hidden_counts == {"insert": 30, "crossfade": 0, "gap": 0}

    # crossfade: the seam overlaps for 4 s
    fade = EditTimeline.from_segments([(0.0, 30.0), (50.0, 80.0)], crossfade=4.0)
    plan = _plan(bundle, tmp_path, [_route()], fade)
    assert marker_state(plan, 28.0).hidden_reason == "crossfade"
    assert render_inset_frame(plan, 28.0).tobytes() == plan.base.tobytes()
    assert plan.hidden_counts["crossfade"] == 40

    # GPS gap between two segments, source 40..60
    segments = [_route(0.0, 40.0), _route(60.0, 100.0)]
    plan = _plan(bundle, tmp_path, segments, EditTimeline.from_segments([(0.0, 100.0)]))
    assert marker_state(plan, 50.0).hidden_reason == "gap"
    assert render_inset_frame(plan, 50.0).tobytes() == plan.base.tobytes()
    assert plan.hidden_counts["gap"] == 199  # t in (40, 60) at 10 fps
    assert marker_state(plan, 39.0).visible and marker_state(plan, 61.0).visible


def test_trim_keeps_removed_start_out_of_the_framing(bundle, tmp_path):
    timeline = EditTimeline.from_segments([(0.0, 100.0)])
    route = _route()
    open_plan = _plan(bundle, tmp_path, [route], timeline)
    plan = _plan(bundle, tmp_path, [route], timeline, trim=RouteTrim(start_m=1000.0))
    removed = [p for p in route if p.time_s < 20.0]
    west, south, east, north = plan.extent
    for point in removed:
        assert not (south <= point.latitude <= north and west <= point.longitude <= east)
        x, y = plan.project(point.latitude, point.longitude)
        assert not (0 <= x < plan.size[0] and 0 <= y < plan.size[1])
    # without the trim the start is framed, so the trim is what moved the window
    assert open_plan.extent[0] < west
    # and the marker is hidden while the car is still inside the trimmed stretch
    assert marker_state(plan, 5.0).hidden_reason == "gap"


def test_zone_removes_points_from_the_extent_and_the_line(bundle, tmp_path):
    timeline = EditTimeline.from_segments([(0.0, 100.0)])
    route = _route()
    centre = RoutePoint(0.0, route[-1].latitude, route[-1].longitude)
    zone = PrivacyZone(centre.latitude, centre.longitude, 800.0)
    open_plan = _plan(bundle, tmp_path, [route], timeline)
    plan = _plan(bundle, tmp_path, [route], timeline, zones=[zone])
    removed = [p for p in route if distance_m(p, centre) <= 800.0]
    assert removed
    assert plan.extent[2] < open_plan.extent[2]
    kept_east = max(p.longitude for s in plan.segments for p in s)
    assert kept_east < min(p.longitude for p in removed)
    assert marker_state(plan, 99.0).hidden_reason == "gap"


def test_labels_inside_a_zone_are_not_drawn(tmp_path):
    here = _point(50.0)
    spot = (here.latitude + 0.004, here.longitude)
    root = _make_bundle(
        tmp_path / "bundle", labels=[{"name": "Elsewhere Town", "lat": spot[0], "lon": spot[1]}]
    )
    timeline = EditTimeline.from_segments([(0.0, 100.0)])

    def white_in_body(plan) -> bool:
        w, h = plan.size
        body = np.asarray(plan.base.crop((0, 0, w, h - 40)))[..., :3]
        return bool((body == 255).all(axis=-1).any())

    assert white_in_body(_plan(root, tmp_path, [_route()], timeline))
    zone = PrivacyZone(spot[0], spot[1], 300.0)
    assert not white_in_body(_plan(root, tmp_path, [_route()], timeline, zones=[zone]))


def test_labels_beside_a_trimmed_start_are_not_drawn(tmp_path):
    start = _point(0.0)
    root = _make_bundle(
        tmp_path / "bundle",
        labels=[
            {"name": "Trailhead", "lat": start.latitude, "lon": start.longitude},
            {"name": "Far Town", "lat": _point(60.0).latitude, "lon": _point(60.0).longitude},
        ],
    )
    timeline = EditTimeline.from_segments([(0.0, 100.0)])
    plan = _plan(root, tmp_path, [_route()], timeline, trim=RouteTrim(start_m=1000.0))
    removed = [p for p in _route() if p.time_s < 20.0]
    kept = map_inset._safe_labels(map_inset._load_labels(root), (), removed)
    assert [label.name for label in kept] == ["Far Town"]
    assert plan.extent[0] > start.longitude


def test_frames_are_byte_identical_across_renders(bundle, tmp_path):
    timeline = EditTimeline.from_segments([(0.0, 30.0), (50.0, 80.0)])
    first = _plan(bundle, tmp_path / "a", [_route()], timeline)
    second = _plan(bundle, tmp_path / "b", [_route()], timeline)
    assert _digest(first.base) == _digest(second.base)
    for t in (0.0, 12.3, 30.0, 55.0):
        assert _digest(render_inset_frame(first, t)) == _digest(render_inset_frame(second, t))


def test_credit_is_drawn_and_recorded(bundle, tmp_path):
    plan = _plan(bundle, tmp_path, [_route()], EditTimeline.from_segments([(0.0, 20.0)]))
    assert plan.credits == (CREDIT,)
    w, h = plan.size
    strip = np.asarray(plan.base.crop((0, h - 40, w, h)))[..., :3]
    body = np.asarray(plan.base.crop((0, 0, w, h - 40)))[..., :3]
    assert (strip == 255).all(axis=-1).any()  # credit text
    assert not (body == 255).all(axis=-1).any()  # nothing else in the base is pure white


@pytest.mark.parametrize(
    "corner, expect",
    [
        ("bottom-left", (38, 1080 - 38 - 288)),
        ("bottom-right", (1920 - 38 - 384, 1080 - 38 - 288)),
        ("top-left", (38, 38)),
        ("top-right", (1920 - 38 - 384, 38)),
    ],
)
def test_inset_rect_for_every_corner(bundle, tmp_path, corner, expect):
    plan = _plan(
        bundle, tmp_path, [_route()], EditTimeline.from_segments([(0.0, 20.0)]), corner=corner
    )
    assert plan.rect == (*expect, 384, 288)


def test_rejects_an_unknown_corner(bundle, tmp_path):
    with pytest.raises(ValueError):
        _plan(bundle, tmp_path, [_route()], EditTimeline.from_segments([(0.0, 20.0)]), corner="mid")


def _refused(bundle, tmp_path, segments, timeline, **kw) -> str:
    out = tmp_path / "out" / "inset.mov"
    out.parent.mkdir()
    with pytest.raises(MapInsetRefused) as caught:
        render_route_inset(bundle, segments, timeline, out, FRAME, workdir=tmp_path / "work", **kw)
    assert list(out.parent.iterdir()) == []  # no clip, no receipt, no temp file
    return str(caught.value)


def test_refuses_a_missing_bundle_before_writing(tmp_path):
    timeline = EditTimeline.from_segments([(0.0, 20.0)])
    assert "bundle" in _refused(tmp_path / "nope", tmp_path, [_route()], timeline)


def test_refuses_a_bundle_that_fails_verification(bundle, tmp_path):
    tile = next((bundle / "imagery").rglob("*.jpg"))
    tile.write_bytes(tile.read_bytes() + b"x")
    timeline = EditTimeline.from_segments([(0.0, 20.0)])
    assert "checksum mismatch" in _refused(bundle, tmp_path, [_route()], timeline)


def test_refuses_when_privacy_leaves_no_route(bundle, tmp_path):
    timeline = EditTimeline.from_segments([(0.0, 20.0)])
    message = _refused(bundle, tmp_path, [_route()], timeline, trim=RouteTrim(start_m=1_000_000.0))
    assert "privacy" in message and "route" in message


def test_refuses_imagery_that_does_not_cover_the_route(bundle, tmp_path):
    timeline = EditTimeline.from_segments([(0.0, 20.0)])
    assert "imagery" in _refused(bundle, tmp_path, [_route(lon0=-80.0)], timeline)


def test_refuses_low_marker_coverage(bundle, tmp_path):
    short = _route(0.0, 20.0)  # only 20 s of a 100 s timeline has a position
    message = _refused(bundle, tmp_path, [short], EditTimeline.from_segments([(0.0, 100.0)]))
    assert "coverage" in message


def test_refusal_lists_every_reason(bundle, tmp_path):
    tile = next((bundle / "imagery").rglob("*.jpg"))
    tile.write_bytes(tile.read_bytes() + b"x")
    timeline = EditTimeline.from_segments([(0.0, 20.0)])
    message = _refused(bundle, tmp_path, [_route()], timeline, trim=RouteTrim(start_m=1_000_000.0))
    assert "checksum mismatch" in message and "privacy" in message


def test_refuses_trimmed_points_that_the_framing_cannot_exclude(bundle, tmp_path):
    # a loop that returns to its start: the trimmed start sits inside the kept route's box
    loop = [
        RoutePoint(float(i), 35.20 + 0.01 * np.sin(i / 10.0), -83.60 + 0.01 * np.cos(i / 10.0))
        for i in range(0, 64)
    ]
    timeline = EditTimeline.from_segments([(0.0, 63.0)])
    message = _refused(bundle, tmp_path, [loop], timeline, trim=RouteTrim(start_m=150.0))
    assert "privacy" in message and "framing" in message


def test_min_coverage_is_configurable(bundle, tmp_path):
    short = _route(0.0, 20.0)
    timeline = EditTimeline.from_segments([(0.0, 30.0)])
    assert _plan(bundle, tmp_path, [short], timeline, min_coverage=0.5).frame_count == 300
    with pytest.raises(MapInsetRefused):
        _plan(bundle, tmp_path, [short], timeline, min_coverage=0.9)


@needs_ffmpeg
def test_writes_an_alpha_clip_and_a_receipt(bundle, tmp_path):
    out = tmp_path / "out" / "inset.mov"
    out.parent.mkdir()
    timeline = EditTimeline.from_segments([(0.0, 1.0), (50.0, 51.0)])
    receipt = render_route_inset(
        bundle,
        [_route()],
        timeline,
        out,
        (640, 360),
        fps=10.0,
        trim=RouteTrim(start_m=100.0),
        workdir=tmp_path / "work",
    )
    assert isinstance(receipt, InsetReceipt)
    assert sorted(p.name for p in out.parent.iterdir()) == ["inset.mov", "inset.mov.receipt.json"]
    probe = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt,width,height,nb_read_frames",
                "-of",
                "json",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["streams"][0]
    assert probe["pix_fmt"] in {"argb", "rgba", "bgra", "abgr", "yuva420p", "yuva444p"}
    assert (probe["width"], probe["height"]) == (128, 96)
    assert int(probe["nb_read_frames"]) == receipt.frame_count == 20
    receipt_path = out.parent / "inset.mov.receipt.json"
    assert InsetReceipt.load(receipt_path) == receipt
    assert receipt.frame_size == (640, 360)
    assert receipt.inset_rect == (13, 360 - 13 - 96, 128, 96)
    assert receipt.fps == 10.0
    assert receipt.credits == (CREDIT,)
    manifest = (bundle / "manifest.json").read_bytes()
    assert receipt.bundle_manifest_sha256 == hashlib.sha256(manifest).hexdigest()
    assert receipt.privacy["route_trim"]["start_m"] == 100.0
    # the 100 m trim removes the first points, so the first second of source has no position
    assert receipt.hidden_frames == {"insert": 0, "crossfade": 0, "gap": 10}
    assert receipt.wall_time_s >= 0
    assert str(tmp_path) not in receipt_path.read_text()


def test_receipt_load_rejects_a_wrong_schema(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schema": "something-else/1"}), encoding="utf-8")
    with pytest.raises(ValueError):
        InsetReceipt.load(path)
