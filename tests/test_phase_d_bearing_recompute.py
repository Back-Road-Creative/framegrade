"""Phase D re-derives the ``🧭 Heading:`` from every episode an LF spans.

A drive encoded before the multi-episode bearing fix carries a heading
computed from its representative GPX alone (Beaufort 2026-02-20 stored
``"NE"`` for an out-and-back; Ace Basin 2026-02-21 would store ``"W"`` for
a northbound LF). The publish pass recomputes it from the persisted
per-episode span map, so the fix reaches drives already encoded.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.drive_phases.phase_d_metadata import _publish_time_bearing


def _gpx(path, points):
    rows = "".join(
        f'<trkpt lat="{lat}" lon="{lon}"><time>2026-02-20T15:{i:02d}:00Z</time></trkpt>'
        for i, (lat, lon) in enumerate(points)
    )
    path.write_text(
        '<?xml version="1.0"?><gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">'
        f"<trk><trkseg>{rows}</trkseg></trk></gpx>"
    )
    return str(path)


def _out_and_back(tmp_path):
    out = _gpx(tmp_path / "ep01.gpx", [(32.4231, -80.4675), (32.36, -80.55), (32.3048, -80.6413)])
    back = _gpx(tmp_path / "ep02.gpx", [(32.3049, -80.6412), (32.36, -80.55), (32.4137, -80.4794)])
    return out, back


def test_out_and_back_heading_dropped(tmp_path):
    out, back = _out_and_back(tmp_path)
    state = SimpleNamespace(
        per_lf_bearing={1: "NE"},
        per_lf_gpx_path={1: back},
        per_lf_gpx_episode_paths={1: {out: None, back: None}},
    )
    assert _publish_time_bearing(state) == {}


def test_one_way_heading_spans_every_episode(tmp_path):
    north = _gpx(tmp_path / "ep01.gpx", [(40.0, -75.0), (40.25, -75.0), (40.5, -75.0)])
    east = _gpx(tmp_path / "ep02.gpx", [(40.5, -75.0), (40.5, -74.7), (40.5, -74.4)])
    state = SimpleNamespace(
        per_lf_bearing={1: "E"},
        per_lf_gpx_path={1: east},
        per_lf_gpx_episode_paths={1: {north: None, east: None}},
    )
    assert _publish_time_bearing(state) == {1: "NE"}


def test_missing_gpx_keeps_stored_heading(tmp_path):
    state = SimpleNamespace(
        per_lf_bearing={"1": "NE"},
        per_lf_gpx_path={1: str(tmp_path / "gone02.gpx")},
        per_lf_gpx_episode_paths={1: {str(tmp_path / "gone01.gpx"): None}},
    )
    assert _publish_time_bearing(state) == {1: "NE"}


def test_no_episode_map_keeps_stored_heading():
    state = SimpleNamespace(
        per_lf_bearing={1: "N"}, per_lf_gpx_path={}, per_lf_gpx_episode_paths=None
    )
    assert _publish_time_bearing(state) == {1: "N"}
