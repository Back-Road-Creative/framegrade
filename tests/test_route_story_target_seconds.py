"""``--target-seconds`` cuts the film to a delivered length (e.g. the 90 s clip).

The map is the elastic part: intro, recap and every still keep their own
seconds, and the map legs share whatever is left under the same honest,
globally proportional time compression ``--map-seconds`` already uses.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_story import film
from gomoveshift.pipeline.route_story.assembly import MAP_SETTLE_SECONDS
from gomoveshift.pipeline.route_story.still_selection import (
    PHOTO_SECONDS_FLOOR,
    resolve_photo_seconds,
    solve_map_seconds,
)


def _total(*, map_seconds, photo_seconds, kept_count, mapped_count, intro=12.0, recap=6.0):
    return (
        intro
        + recap
        + map_seconds
        + kept_count * photo_seconds
        + mapped_count * (MAP_SETTLE_SECONDS)
    )


@pytest.mark.parametrize("kept_count, mapped_count", [(12, 12), (12, 10), (3, 3), (0, 0)])
def test_the_solved_film_lands_on_the_target(kept_count, mapped_count) -> None:
    map_seconds, photo_seconds = solve_map_seconds(
        target_seconds=90.0,
        kept_count=kept_count,
        mapped_count=mapped_count,
        requested_photo_seconds=2.5,
        intro_seconds=12.0,
        recap_seconds=6.0,
    )
    assert map_seconds > 0
    total = _total(
        map_seconds=map_seconds,
        photo_seconds=photo_seconds,
        kept_count=kept_count,
        mapped_count=mapped_count,
    )
    assert total == pytest.approx(90.0)


def test_the_photo_seconds_are_the_ones_pacing_would_pick_for_that_map() -> None:
    map_seconds, photo_seconds = solve_map_seconds(
        target_seconds=90.0,
        kept_count=4,
        mapped_count=4,
        requested_photo_seconds=6.0,
        intro_seconds=12.0,
        recap_seconds=6.0,
    )
    pacing = resolve_photo_seconds(
        requested_seconds=6.0,
        kept_count=4,
        intro_seconds=12.0,
        map_seconds=map_seconds,
        recap_seconds=6.0,
    )
    assert photo_seconds == pytest.approx(pacing["used_seconds"])
    assert photo_seconds >= PHOTO_SECONDS_FLOOR


def test_a_placed_video_counts_toward_the_target() -> None:
    plain, _ = solve_map_seconds(
        target_seconds=90.0,
        kept_count=12,
        mapped_count=12,
        requested_photo_seconds=2.5,
        intro_seconds=12.0,
        recap_seconds=6.0,
    )
    with_video, _ = solve_map_seconds(
        target_seconds=90.0,
        kept_count=12,
        mapped_count=12,
        requested_photo_seconds=2.5,
        intro_seconds=12.0,
        recap_seconds=6.0,
        extra_seconds=14.9,
    )
    assert plain - with_video == pytest.approx(14.9)


def test_a_target_the_fixed_scenes_already_fill_is_refused() -> None:
    with pytest.raises(ValueError, match="target"):
        solve_map_seconds(
            target_seconds=40.0,
            kept_count=12,
            mapped_count=12,
            requested_photo_seconds=2.5,
            intro_seconds=12.0,
            recap_seconds=6.0,
        )


def _argv(tmp_path, *extra: str) -> list[str]:
    return [
        "--gpx",
        str(tmp_path / "t.gpx"),
        "--recorded-on",
        "2026-09-16",
        "--photos-dir",
        str(tmp_path),
        "--national-raster",
        str(tmp_path / "n.tif"),
        "--national-credit",
        "n",
        "--local-raster",
        str(tmp_path / "l.tif"),
        "--local-credit",
        "l",
        "--local-bounds",
        "-84",
        "35",
        "-83",
        "36",
        "--output",
        str(tmp_path / "out.mp4"),
        *extra,
    ]


# Both refusals must come from the flag's own check -- a SystemExit for a
# missing required argument would pass these vacuously, so the message is read.
def test_target_and_map_seconds_are_mutually_exclusive(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        film.main(_argv(tmp_path, "--target-seconds", "90", "--map-seconds", "30"))
    assert "--target-seconds" in capsys.readouterr().err


def test_a_non_positive_target_is_refused(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        film.main(_argv(tmp_path, "--target-seconds", "0"))
    assert "--target-seconds must be positive" in capsys.readouterr().err
