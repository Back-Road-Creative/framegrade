"""`select_kept_photos` must report what it did through the log, not only in
the run receipt -- the receipt lands only when the film finishes, hours into
a render; the log is the operator's first and only look at selection while
it's still running."""

import logging

from PIL import Image

from gomoveshift.pipeline.route_story import still_selection as still_selection_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment
from gomoveshift.pipeline.route_story.select import (
    GroupResult,
    PhotoScore,
    SelectionOptions,
    SelectionResult,
)
from gomoveshift.pipeline.route_story.still_selection import select_kept_photos

LOGGER_NAME = still_selection_module.logger.name


def _photo(tmp_path, name: str) -> PhotoMoment:
    source = tmp_path / name
    Image.new("RGB", (16, 16), "#334455").save(source)
    anchor = MediaAnchor(name, "timestamp_interpolated", 5.0, 35.0, -80.0)
    return PhotoMoment(source, anchor, captured_at_exif_s=5.0)


def _score(name: str) -> PhotoScore:
    return PhotoScore(name=name, sharpness=1.0, exposure_badness=0.0, color_vibrancy=1.0)


def _fake_result(*, groups, kept):
    return SelectionResult(groups=tuple(groups), kept=tuple(kept))


def test_totals_line_reports_in_groups_kept_and_dropped_counts(tmp_path, monkeypatch, caplog):
    photos = [_photo(tmp_path, n) for n in ("a.jpg", "b.jpg", "c.jpg")]
    collapsed = GroupResult(
        members=("a.jpg", "b.jpg"),
        winner="a.jpg",
        kept=("a.jpg",),
        losers=("b.jpg",),
        scores={"a.jpg": _score("a.jpg"), "b.jpg": _score("b.jpg")},
        reason="sharpest",
    )
    solo = GroupResult(
        members=("c.jpg",),
        winner="c.jpg",
        kept=("c.jpg",),
        losers=(),
        scores={"c.jpg": _score("c.jpg")},
        reason="only member",
    )
    result = _fake_result(groups=[collapsed, solo], kept=["a.jpg", "c.jpg"])
    monkeypatch.setattr(
        still_selection_module, "select_best_photos", lambda candidates, options: result
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        select_kept_photos(photos, still_source="developed", options=SelectionOptions())

    totals = [r for r in caplog.records if "3 candidate" in r.getMessage()]
    assert len(totals) == 1
    message = totals[0].getMessage()
    assert "3 candidate" in message
    assert "2 group" in message
    assert "kept 2" in message
    assert "dropped 1" in message


def test_collapsed_group_logs_winner_losers_and_reason(tmp_path, monkeypatch, caplog):
    photos = [_photo(tmp_path, n) for n in ("a.jpg", "b.jpg")]
    collapsed = GroupResult(
        members=("a.jpg", "b.jpg"),
        winner="a.jpg",
        kept=("a.jpg",),
        losers=("b.jpg",),
        scores={"a.jpg": _score("a.jpg"), "b.jpg": _score("b.jpg")},
        reason="sharpest",
    )
    result = _fake_result(groups=[collapsed], kept=["a.jpg"])
    monkeypatch.setattr(
        still_selection_module, "select_best_photos", lambda candidates, options: result
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        select_kept_photos(photos, still_source="developed", options=SelectionOptions())

    group_lines = [
        r for r in caplog.records if "a.jpg" in r.getMessage() and "b.jpg" in r.getMessage()
    ]
    assert len(group_lines) == 1
    message = group_lines[0].getMessage()
    assert "a.jpg" in message
    assert "b.jpg" in message
    assert "sharpest" in message


def test_single_member_group_logs_no_per_group_line(tmp_path, monkeypatch, caplog):
    photos = [_photo(tmp_path, "solo.jpg")]
    solo = GroupResult(
        members=("solo.jpg",),
        winner="solo.jpg",
        kept=("solo.jpg",),
        losers=(),
        scores={"solo.jpg": _score("solo.jpg")},
        reason="only member",
    )
    result = _fake_result(groups=[solo], kept=["solo.jpg"])
    monkeypatch.setattr(
        still_selection_module, "select_best_photos", lambda candidates, options: result
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        select_kept_photos(photos, still_source="developed", options=SelectionOptions())

    # Only the totals line -- a 200-photo shoot with no look-alikes must not
    # emit 200 "only member" lines.
    assert len(caplog.records) == 1
    assert "1 candidate" in caplog.records[0].getMessage()


def test_disabled_selection_logs_one_line_and_no_group_spam(tmp_path, monkeypatch, caplog):
    photos = [_photo(tmp_path, n) for n in ("a.jpg", "b.jpg", "c.jpg")]
    # Mirrors what select_best_photos itself returns when disabled: every
    # candidate is its own one-member group.
    groups = [
        GroupResult(
            members=(p.source.name,),
            winner=p.source.name,
            kept=(p.source.name,),
            losers=(),
            scores={p.source.name: _score(p.source.name)},
            reason="selection disabled",
        )
        for p in photos
    ]
    result = _fake_result(groups=groups, kept=[p.source.name for p in photos])
    monkeypatch.setattr(
        still_selection_module, "select_best_photos", lambda candidates, options: result
    )
    options = SelectionOptions(enabled=False)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        select_kept_photos(photos, still_source="developed", options=options)

    assert len(caplog.records) == 1
    assert "disabled" in caplog.records[0].getMessage().lower()


def test_empty_candidate_list_logs_zero_totals_without_crashing(monkeypatch, caplog):
    result = _fake_result(groups=[], kept=[])
    monkeypatch.setattr(
        still_selection_module, "select_best_photos", lambda candidates, options: result
    )

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        kept, returned = select_kept_photos(
            [], still_source="developed", options=SelectionOptions()
        )

    assert kept == []
    assert returned is result
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "0 candidate" in message
    assert "kept 0" in message
    assert "dropped 0" in message
