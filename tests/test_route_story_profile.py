"""Route Story launch profile -> film.py argv mapping (console's in-process door)."""

from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story import profile as profile_module
from gomoveshift.pipeline.route_story.profile import (
    ProfileError,
    build_film_argv,
    film_parser,
    load_profile,
    unverified_values,
)


def test_load_profile_reads_a_yaml_mapping(tmp_path):
    path = tmp_path / "trip.yaml"
    path.write_text("gpx: track.gpx\noutput: film.mp4\n", encoding="utf-8")

    assert load_profile(path) == {"gpx": "track.gpx", "output": "film.mp4"}


def test_load_profile_refuses_a_missing_file(tmp_path):
    with pytest.raises(ProfileError, match="not found"):
        load_profile(tmp_path / "missing.yaml")


def test_load_profile_refuses_a_non_mapping(tmp_path):
    path = tmp_path / "trip.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ProfileError, match="mapping"):
        load_profile(path)


def test_unknown_profile_key_rejected_before_launch():
    parser = film_parser()

    with pytest.raises(ProfileError, match="not-a-real-flag"):
        build_film_argv(parser, {"not-a-real-flag": "value"})


def test_bool_true_maps_to_a_bare_flag():
    parser = film_parser()

    argv = build_film_argv(parser, {"no-best-pick": True})

    assert argv == ["--no-best-pick"]


def test_bool_false_omits_the_flag():
    parser = film_parser()

    argv = build_film_argv(parser, {"no-best-pick": False})

    assert argv == []


def test_append_action_repeats_the_flag_per_item():
    parser = film_parser()

    argv = build_film_argv(parser, {"keep": ["a.jpg", "b.jpg"]})

    assert argv == ["--keep", "a.jpg", "--keep", "b.jpg"]


def test_nargs_list_appends_all_values_after_one_flag():
    parser = film_parser()

    argv = build_film_argv(parser, {"local-bounds": [-80.1, 35.0, -79.9, 35.2]})

    assert argv == ["--local-bounds", "-80.1", "35.0", "-79.9", "35.2"]


def test_pacing_mode_key_maps_to_the_flag():
    parser = film_parser()

    argv = build_film_argv(parser, {"pacing-mode": "recorded"})

    assert argv == ["--pacing-mode", "recorded"]


def test_regional_triple_length_mismatch_rejected():
    parser = film_parser()

    with pytest.raises(ProfileError, match="same length"):
        build_film_argv(
            parser,
            {
                "regional-raster": ["a.tif"],
                "regional-credit": ["c1", "c2"],
            },
        )


def test_regional_triple_builds_repeated_flags():
    parser = film_parser()

    argv = build_film_argv(
        parser,
        {
            "regional-raster": ["a.tif", "b.tif"],
            "regional-credit": ["c1", "c2"],
            "regional-bounds": [[-1, -2, -3, -4], [1, 2, 3, 4]],
        },
    )

    assert argv == [
        "--regional-raster",
        "a.tif",
        "--regional-raster",
        "b.tif",
        "--regional-credit",
        "c1",
        "--regional-credit",
        "c2",
        "--regional-bounds",
        "-1",
        "-2",
        "-3",
        "-4",
        "--regional-bounds",
        "1",
        "2",
        "3",
        "4",
    ]


def test_unverified_values_detects_the_placeholder_prefix():
    argv = ["--photos-dir", "UNVERIFIED-example/photos", "--output", "film.mp4"]

    assert unverified_values(argv) == ["UNVERIFIED-example/photos"]


def test_unverified_values_empty_when_nothing_is_a_placeholder():
    assert unverified_values(["--output", "film.mp4"]) == []


def test_scene_bundle_miss_names_the_prepare_command(tmp_path, monkeypatch):
    parser = film_parser()
    monkeypatch.setattr(profile_module, "find_prepared_bundle", lambda *a, **kw: None)

    with pytest.raises(ProfileError, match="route-story-prepare trip.gpx --recorded-on 2026-09-16"):
        build_film_argv(
            parser,
            {"renderer": "scene", "gpx": "trip.gpx", "recorded-on": "2026-09-16"},
        )


def test_scene_bundle_hit_adds_bundle_flag(tmp_path, monkeypatch):
    parser = film_parser()
    monkeypatch.setattr(
        profile_module,
        "find_prepared_bundle",
        lambda gpx, *, recorded_on: Path("/bundles/found"),
    )

    argv = build_film_argv(
        parser,
        {"renderer": "scene", "gpx": "trip.gpx", "recorded-on": "2026-09-16"},
    )

    assert argv[-2:] == ["--bundle", "/bundles/found"]
