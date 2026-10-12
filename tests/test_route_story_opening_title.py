"""The opening title card and the high-point callout.

Both surfaces name real things: the title names places the track actually ran
near and the day it was recorded, the callout reports an elevation the GPS
actually recorded. Neither may invent a name when the evidence is absent --
the film's whole claim is that everything on screen came off the track.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from PIL import Image

from gomoveshift.pipeline.route_story import assembly, hud
from gomoveshift.pipeline.route_story.metrics import RoutePoint

WIDTH = 1920
HEIGHT = 1080


def _place(name: str, latitude: float, longitude: float) -> SimpleNamespace:
    return SimpleNamespace(name=name, latitude=latitude, longitude=longitude, kind="town")


def _blank() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), (3, 12, 17))


def _ink(image: Image.Image) -> int:
    """How many pixels differ from the flat backdrop this frame started as."""
    reference = _blank()
    return sum(1 for a, b in zip(image.getdata(), reference.getdata()) if a != b)


def _ink_weight(image: Image.Image) -> int:
    """How FAR the frame moved from its backdrop -- the same pixels drawn at
    half opacity score half as much, which pixel counts cannot distinguish."""
    reference = _blank()
    return sum(
        abs(a - b)
        for pixel, ref in zip(image.getdata(), reference.getdata())
        for a, b in zip(pixel, ref)
    )


def test_the_default_title_names_the_places_the_track_ran_between() -> None:
    labels = (
        _place("Bryson City", 35.4301, -83.4471),
        _place("Robbinsville", 35.3226, -83.8074),
        _place("Asheville", 35.5951, -82.5515),
    )
    title, subtitle = hud.resolve_opening_title(
        None,
        labels,
        start=(35.4295, -83.4460),
        end=(35.3230, -83.8060),
        recorded_on=date(2026, 9, 16),
    )

    assert title == "Bryson City → Robbinsville"
    assert subtitle == "16 September 2026"


def test_a_track_that_starts_and_ends_in_one_place_names_it_once() -> None:
    labels = (_place("Bryson City", 35.4301, -83.4471),)
    title, _ = hud.resolve_opening_title(
        None,
        labels,
        start=(35.4295, -83.4460),
        end=(35.4310, -83.4480),
        recorded_on=date(2026, 9, 16),
    )

    assert title == "Bryson City"


def test_a_place_too_far_from_the_track_is_never_named_in_the_title() -> None:
    """The labels file carries every name in the regional bbox, most of them
    nowhere near the ride. Naming the nearest one regardless would caption a
    Smokies ride with a town two counties away -- an invented claim, which is
    exactly what this film may not make. No close place means no place line,
    and the date alone carries the card.
    """
    labels = (_place("Charlotte", 35.2271, -80.8431),)
    title, subtitle = hud.resolve_opening_title(
        None,
        labels,
        start=(35.4295, -83.4460),
        end=(35.3230, -83.8060),
        recorded_on=date(2026, 9, 16),
    )

    assert title == ""
    assert subtitle == "16 September 2026"


def test_the_operators_own_title_is_used_verbatim() -> None:
    """``--title`` is the operator's own words for the trip. A derived place
    line may not edit, append to, or reorder them."""
    labels = (_place("Bryson City", 35.4301, -83.4471),)
    title, subtitle = hud.resolve_opening_title(
        "Tail of the Dragon, the long way",
        labels,
        start=(35.4295, -83.4460),
        end=(35.3230, -83.8060),
        recorded_on=date(2026, 9, 16),
    )

    assert title == "Tail of the Dragon, the long way"
    assert subtitle == "16 September 2026"


def test_a_track_with_no_known_date_still_gets_its_place_line() -> None:
    labels = (_place("Bryson City", 35.4301, -83.4471),)
    title, subtitle = hud.resolve_opening_title(
        None,
        labels,
        start=(35.4295, -83.4460),
        end=(35.4310, -83.4480),
        recorded_on=None,
    )

    assert title == "Bryson City"
    assert subtitle == ""


def test_the_opening_title_is_off_the_screen_before_the_camera_settles() -> None:
    """The opening camera descends to the start of the route and the film's
    first map leg begins the instant it lands. A title still burning in at
    that hand-off covers the route the leg exists to show, so it is gone by
    the time the descent ends -- not merely dimmed.
    """
    mid = _blank()
    hud.draw_opening_title(mid, 0.5, "Bryson City → Robbinsville", "16 September 2026")
    assert _ink(mid) > 0

    settled = _blank()
    hud.draw_opening_title(settled, 1.0, "Bryson City → Robbinsville", "16 September 2026")
    assert _ink(settled) == 0


def test_the_opening_title_fades_up_rather_than_cutting_in() -> None:
    """The very first frame of the descent carries no title at all, and the
    frames just after it carry a partial one -- a card that cut in at full
    opacity on frame zero would read as a slate, not a film opening."""
    first = _blank()
    hud.draw_opening_title(first, 0.0, "Bryson City", "16 September 2026")
    assert _ink(first) == 0

    rising = _blank()
    hud.draw_opening_title(rising, hud.OPENING_TITLE_FADE_IN_END / 2, "Bryson City", "16 Sep")
    held = _blank()
    hud.draw_opening_title(held, 0.5, "Bryson City", "16 Sep")
    assert 0 < _ink_weight(rising) < _ink_weight(held)


def test_an_empty_title_draws_nothing_at_all() -> None:
    frame = _blank()
    hud.draw_opening_title(frame, 0.5, "", "")
    assert _ink(frame) == 0


def _climb() -> list[RoutePoint]:
    return [
        RoutePoint(time_s=0.0, latitude=35.43, longitude=-83.44, elevation_m=520.0),
        RoutePoint(time_s=600.0, latitude=35.40, longitude=-83.55, elevation_m=1538.0),
        RoutePoint(time_s=1200.0, latitude=35.32, longitude=-83.80, elevation_m=610.0),
    ]


def test_the_high_point_callout_reports_the_elevation_the_gps_recorded() -> None:
    callout = hud.high_point_callout(_climb())

    assert callout is not None
    time_s, text = callout
    assert time_s == 600.0
    # 1538 m is 5046 ft. The callout rounds to the foot and never restates the
    # metre value the track carried -- one unit, the one the dashboard uses.
    assert text == "HIGH POINT · 5,046 ft"


def test_a_track_with_no_elevation_data_gets_no_high_point_callout() -> None:
    flat = [
        RoutePoint(time_s=0.0, latitude=35.43, longitude=-83.44),
        RoutePoint(time_s=600.0, latitude=35.40, longitude=-83.55),
    ]

    assert hud.high_point_callout(flat) is None


def test_a_high_point_that_is_the_first_or_last_point_is_still_a_high_point() -> None:
    descent = [
        RoutePoint(time_s=0.0, latitude=35.43, longitude=-83.44, elevation_m=1538.0),
        RoutePoint(time_s=600.0, latitude=35.40, longitude=-83.55, elevation_m=520.0),
    ]
    callout = hud.high_point_callout(descent)

    assert callout is not None
    assert callout[0] == 0.0


def _jpeg_ink(path) -> bool:
    """True when a saved frame carries drawn pixels, not just JPEG noise.

    Frames land as quality-95 JPEG, so a "flat" backdrop is not one exact
    colour -- comparing colour counts would call ringing artefacts a badge.
    """
    backdrop = (3, 12, 17)
    return any(
        max(abs(channel - reference) for channel, reference in zip(pixel, backdrop)) > 40
        for pixel in Image.open(path).convert("RGB").getdata()
    )


class _FlatRenderer:
    """Yields one flat frame per requested time, recording the times asked for."""

    def __init__(self) -> None:
        self.times: list[float] = []

    def render_leg(self, segments, times, width, height):
        for time_s in times:
            self.times.append(time_s)
            yield Image.new("RGB", (width, height), (3, 12, 17))


def test_the_high_point_callout_is_drawn_only_on_the_frames_around_it(tmp_path) -> None:
    """The callout is a moment on the route, not a permanent overlay: it
    appears as the rider reaches the summit and is gone again afterwards.
    Drawing it on every frame of the leg would turn a landmark into chrome.
    """
    renderer = _FlatRenderer()
    frames = tmp_path / "frames"
    frames.mkdir()

    assembly._append_map_leg(
        frames,
        0,
        [_climb()],
        renderer,
        0.0,
        1200.0,
        frame_count=12,
        width=640,
        height=360,
        callout=(600.0, "HIGH POINT · 5,046 ft"),
    )

    written = sorted(frames.glob("*.jpg"))
    assert len(written) == 12
    inked = [index for index, path in enumerate(written) if _jpeg_ink(path)]
    assert inked, "the callout never appeared on any frame of the leg"
    # Every frame carrying it sits inside the callout's own window; frames
    # outside it are the untouched map.
    for index in inked:
        assert abs(renderer.times[index] - 600.0) <= hud.HIGH_POINT_CALLOUT_SECONDS
    assert len(inked) < 12


def test_a_leg_with_no_callout_renders_untouched_map_frames(tmp_path) -> None:
    renderer = _FlatRenderer()
    frames = tmp_path / "frames"
    frames.mkdir()

    assembly._append_map_leg(
        frames, 0, [_climb()], renderer, 0.0, 1200.0, frame_count=4, width=640, height=360
    )

    for path in sorted(frames.glob("*.jpg")):
        assert not _jpeg_ink(path)
