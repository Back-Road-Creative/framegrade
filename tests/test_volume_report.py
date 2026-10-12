"""The report-only volume path is gone: Waypost decides shorts volume from the
evidence export, so nothing in GMS recommends a number any more."""

from __future__ import annotations

import importlib

import pytest

from gomoveshift.cli.parser import _build_parser as build_parser


@pytest.mark.parametrize(
    "name", ["recommend_volume", "VolumeRecommendation", "DEFAULT_K", "DEFAULT_FLOOR_FRACTION"]
)
def test_the_recommend_half_of_volume_is_gone(name):
    volume = importlib.import_module("gomoveshift.tracking.volume")
    with pytest.raises(ImportError):
        exec(f"from gomoveshift.tracking.volume import {name}", {})  # noqa: S102
    assert not hasattr(volume, name)


def test_the_library_hygiene_half_of_volume_stays():
    from gomoveshift.tracking.volume import select_library_hygiene_candidates  # noqa: F401


@pytest.mark.parametrize("command", ["volume-report", "lf-length-report"])
def test_the_report_only_subcommands_are_not_registered(command, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    "module",
    [
        "gomoveshift.cli.handlers.volume_report",
        "gomoveshift.cli.handlers.lf_length_report",
        "gomoveshift.tracking.length_evidence",
    ],
)
def test_the_report_only_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
