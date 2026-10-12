"""``FilmPreset``/``PRESETS``: the ``full``/``social`` editorial defaults.

``full`` must reproduce film.py's pre-preset literals exactly -- an operator
who never passes ``--preset`` sees byte-identical behavior.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_story.film import (
    DEFAULT_INTRO_SECONDS,
    DEFAULT_RECAP_SECONDS,
    DELIVERY_HEIGHT,
    DELIVERY_WIDTH,
)
from gomoveshift.pipeline.route_story.presets import (
    PRESETS,
    FilmTooLongError,
    apply_preset,
    capped_still_count,
    check_ceiling,
)


def test_full_preset_matches_film_pys_pre_preset_defaults() -> None:
    full = PRESETS["full"]
    assert full.width == DELIVERY_WIDTH
    assert full.height == DELIVERY_HEIGHT
    assert full.intro_seconds == DEFAULT_INTRO_SECONDS
    assert full.recap_seconds == DEFAULT_RECAP_SECONDS
    assert full.photo_seconds == 2.5
    assert full.map_seconds is None
    assert full.max_photos is None
    assert full.max_seconds is None


def test_social_preset_is_vertical_and_capped() -> None:
    social = PRESETS["social"]
    assert (social.width, social.height) == (1080, 1920)
    assert social.max_seconds == 90.0
    assert social.max_photos is not None
    # The social cut is delivered by encode_final's own audio
    # policy -- it must NOT resolve to the drive-shorts rendition, which
    # strips audio (-an).
    assert social.rendition == "route_story_short"


def test_apply_preset_fills_only_unset_fields() -> None:
    social = PRESETS["social"]
    width, height, photo_seconds, map_seconds = apply_preset(
        social, width=None, height=None, photo_seconds=None, map_seconds=None
    )
    assert (width, height) == (1080, 1920)
    assert photo_seconds == social.photo_seconds
    assert map_seconds == social.map_seconds


def test_apply_preset_explicit_values_override_the_preset() -> None:
    social = PRESETS["social"]
    width, height, photo_seconds, map_seconds = apply_preset(
        social, width=1920, height=1080, photo_seconds=3.0, map_seconds=10.0
    )
    assert (width, height) == (1920, 1080)
    assert photo_seconds == 3.0
    assert map_seconds == 10.0


def test_full_preset_never_caps_still_count() -> None:
    full = PRESETS["full"]
    assert capped_still_count(full, cap_limit=40) == 40


def test_social_preset_caps_still_count_at_the_lower_of_the_two() -> None:
    social = PRESETS["social"]
    assert capped_still_count(social, cap_limit=40) == social.max_photos
    assert capped_still_count(social, cap_limit=3) == 3


def test_check_ceiling_is_a_noop_without_a_ceiling() -> None:
    check_ceiling(PRESETS["full"], planned_seconds=10_000.0)


def test_check_ceiling_refuses_a_plan_over_the_social_ceiling() -> None:
    with pytest.raises(FilmTooLongError, match="90"):
        check_ceiling(PRESETS["social"], planned_seconds=91.0)


def test_check_ceiling_allows_a_plan_at_or_under_the_ceiling() -> None:
    check_ceiling(PRESETS["social"], planned_seconds=90.0)


# --- main() CLI wiring: argparse-level checks only, no fixtures/render needed ---


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


def test_an_unknown_preset_name_is_refused(tmp_path, capsys) -> None:
    from gomoveshift.pipeline.route_story import film

    with pytest.raises(SystemExit):
        film.main(_argv(tmp_path, "--preset", "bogus"))
    assert "--preset" in capsys.readouterr().err


def test_preset_social_still_honors_the_target_seconds_map_seconds_exclusivity(
    tmp_path, capsys
) -> None:
    # Regression guard: the preset must be resolved AFTER this exclusivity
    # check, never before -- a preset fill happening first would make
    # --map-seconds look operator-supplied on every social run.
    from gomoveshift.pipeline.route_story import film

    with pytest.raises(SystemExit):
        film.main(
            _argv(tmp_path, "--preset", "social", "--target-seconds", "90", "--map-seconds", "30")
        )
    assert "--target-seconds" in capsys.readouterr().err
