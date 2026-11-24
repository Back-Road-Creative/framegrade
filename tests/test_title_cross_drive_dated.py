"""An exact title clash with a DIFFERENT drive's live video gets the drive date."""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.exceptions import PhaseBCExitContractError
from gomoveshift.pipeline.title_builder import ensure_unique_titles

_TITLE = "County Cork, Ireland"
_LIVE = "county cork, ireland in 4k"
_IDS = {_LIVE: "VID1"}


def _run(**kw):
    base = dict(
        published_titles={_LIVE},
        published_title_ids=_IDS,
        drive_project="ireland-2025-09-19-old-head-courtmacsherry",
        lf_dates={1: "2025-09-19"},
        video_projects={"VID1": "ireland-2025-09-17-baltimore-clonakilty"},
    )
    base.update(kw)
    return ensure_unique_titles({1: _TITLE}, **base)


def test_cross_project_clash_gets_the_dated_title():
    assert _run() == {1: "County Cork, Ireland (Sep 19, 2025)"}


def test_same_project_clash_still_raises():
    with pytest.raises(PhaseBCExitContractError):
        _run(video_projects={"VID1": "ireland-2025-09-19-old-head-courtmacsherry"})


def test_unknown_project_still_raises():
    with pytest.raises(PhaseBCExitContractError):
        _run(video_projects={})


def test_unknown_drive_project_still_raises():
    with pytest.raises(PhaseBCExitContractError):
        _run(drive_project=None)


def test_no_measured_date_still_raises():
    with pytest.raises(PhaseBCExitContractError):
        _run(lf_dates={})


def test_dated_title_that_also_collides_raises():
    dated = "county cork, ireland (sep 19, 2025) in 4k"
    with pytest.raises(PhaseBCExitContractError):
        _run(published_titles={_LIVE, dated})


def test_cross_drive_dated_never_dates_an_already_dated_title():
    from gomoveshift.pipeline.title_builder import _drive_dated

    assert _drive_dated("Ireland (Sep 16, 2025)", "2025-09-17") == "Ireland (Sep 17, 2025)"
    assert _drive_dated("Ireland", "2025-09-17") == "Ireland (Sep 17, 2025)"
    assert _drive_dated("Ireland", "") is None
    assert _drive_dated("Ireland", "garbage") is None


def test_an_already_dated_title_is_never_dated_twice_so_a_dated_live_title_refuses():
    dated = "County Cork, Ireland (Sep 19, 2025)"
    live = dated.lower() + " in 4k"
    with pytest.raises(PhaseBCExitContractError):
        ensure_unique_titles(
            {1: dated},
            published_titles={live},
            published_title_ids={live: "VID1"},
            drive_project="ireland-2025-09-19-old-head-courtmacsherry",
            lf_dates={1: "2025-09-19"},
            video_projects={"VID1": "ireland-2025-09-17-baltimore-clonakilty"},
        )
