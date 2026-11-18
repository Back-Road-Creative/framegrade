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
