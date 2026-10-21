"""A duplicate-title refusal names the live video it collides with.

U5 c (2026-10-15) refused "Washington County, VA" against a live video but
printed ``id=unknown``: the pre-encode dedup in ``_apply_demoted_labels`` got
no id map, and the on-disk ledger carries titles only. The refusal now falls
back to the local live-catalog snapshot (the one the publishing gate reads)
-- never the network. Ids and titles are synthetic.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers.pub_status import LIVE_CATALOG_SNAPSHOT
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.exceptions import PhaseBCExitContractError
from gomoveshift.pipeline.title_builder import ensure_unique_titles

_LIVE_ID = "SYNTHdup001"
_LIVE_TITLE = "Foothills Parkway, TN in 4K"


def _write_snapshot(videos):
    lm.DEFAULT_SEO_STORAGE.mkdir(parents=True, exist_ok=True)
    (lm.DEFAULT_SEO_STORAGE / LIVE_CATALOG_SNAPSHOT).write_text(
        json.dumps({"videos": videos}), encoding="utf-8"
    )


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the dedup called YouTube")

    monkeypatch.setattr(lm, "refresh_channel_title_ledger", boom)


def test_a_ledger_only_collision_names_the_id_from_the_local_snapshot():
    _write_snapshot({_LIVE_ID: {"title": _LIVE_TITLE, "privacy": "public"}})
    with pytest.raises(PhaseBCExitContractError, match=f"id={_LIVE_ID}"):
        ensure_unique_titles({1: "Foothills Parkway, TN"}, published_titles={_LIVE_TITLE.lower()})


def test_the_snapshot_match_ignores_case_and_padding():
    _write_snapshot({_LIVE_ID: {"title": f"  {_LIVE_TITLE.upper()} "}})
    with pytest.raises(PhaseBCExitContractError, match=f"id={_LIVE_ID}"):
        ensure_unique_titles({1: "Foothills Parkway, TN"}, published_titles={_LIVE_TITLE})


def test_a_caller_supplied_id_wins_over_the_snapshot():
    _write_snapshot({"SYNTHother1": {"title": _LIVE_TITLE}})
    with pytest.raises(PhaseBCExitContractError, match="id=SYNTHcaller"):
        ensure_unique_titles(
            {1: "Foothills Parkway, TN"},
            published_titles={_LIVE_TITLE},
            published_title_ids={_LIVE_TITLE.lower(): "SYNTHcaller"},
        )


def test_no_snapshot_says_where_the_id_would_come_from():
    with pytest.raises(PhaseBCExitContractError) as exc:
        ensure_unique_titles({1: "Foothills Parkway, TN"}, published_titles={_LIVE_TITLE})
    msg = str(exc.value)
    assert "id=unknown" in msg
    assert LIVE_CATALOG_SNAPSHOT in msg


def test_an_unreadable_snapshot_still_refuses():
    lm.DEFAULT_SEO_STORAGE.mkdir(parents=True, exist_ok=True)
    (lm.DEFAULT_SEO_STORAGE / LIVE_CATALOG_SNAPSHOT).write_text("{not json", encoding="utf-8")
    with pytest.raises(PhaseBCExitContractError, match="id=unknown"):
        ensure_unique_titles({1: "Foothills Parkway, TN"}, published_titles={_LIVE_TITLE})


def test_no_collision_never_reads_the_snapshot(monkeypatch):
    monkeypatch.setattr(lm, "live_catalog_title_ids", lambda: pytest.fail("read on the happy path"))
    assert ensure_unique_titles({1: "A"}, published_titles={_LIVE_TITLE}) == {1: "A"}


def test_the_demoted_label_preview_passes_the_channel_ids():
    """The U5 c refusal came from this site, which passed no id map."""
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _apply_demoted_labels

    seen = {}

    def unique_titles(titles, **kwargs):
        seen.update(kwargs)
        return dict(titles)

    kit = SimpleNamespace(unique_titles=unique_titles, published_titles=lambda: {"x"})
    state = SimpleNamespace(per_lf_ost_text={}, per_lf_titles={})
    _apply_demoted_labels({1: "L"}, {1: "old"}, {}, state, kit, title_ids={"x": _LIVE_ID})
    assert seen["published_title_ids"] == {"x": _LIVE_ID}
