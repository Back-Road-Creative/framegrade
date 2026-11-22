# ruff: noqa: F811  (fixtures are imported from the adapter test module)
"""A reviewed Short's description ends with one link to its parent drive.

Synthetic data only; the video ids are invented.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_publishing_youtube_adapter import (  # noqa: F401 -- fixtures
    _publish,
    _setup,
    mock_service,
    queue_conn,
)

from gomoveshift.pipeline.drive_phases import youtube_uploader as youtube_uploader_module
from gomoveshift.publishing.youtube_adapter import build_short_final_text_review

ROOT = Path(__file__).resolve().parent.parent
PIN = "fe920eb23f1678648591892de27270d9df357e68"  # pragma: allowlist secret
VID = "aB3_dE-fG1h"
TITLE = "County Kerry drive"


@pytest.fixture(autouse=True)
def _bind_open(monkeypatch):
    monkeypatch.setattr(youtube_uploader_module, "FINAL_TEXT_BIND_ON", True, raising=False)


def _record(mock_service, queue_conn, tmp_path, *, parent=VID, parent_at=None):
    adapter, record, did, vp = _setup(mock_service, queue_conn, tmp_path)
    yt = replace(record.payloads.youtube, title=TITLE, description="x", tags=())
    record = replace(
        record,
        payloads=replace(record.payloads, youtube=yt),
        sampled_regions=("County Kerry",),
        parent_lf_video_id=parent,
        parent_lf_publish_at=parent_at or record.parent_lf_publish_at,
    )
    return adapter, record, did, vp


def _description(review):
    from seo_manager.analytics.final_text_surfaces import compose_description

    return compose_description(review["surfaces"])


def test_a_parent_id_ends_the_description_with_the_link(mock_service, queue_conn, tmp_path):
    _, record, _, _ = _record(mock_service, queue_conn, tmp_path)
    review = build_short_final_text_review(record)
    assert review["verdict"] == "agree", review["reasons"]
    assert _description(review) == f"Drive through County Kerry\nFull drive: https://youtu.be/{VID}"


def test_the_link_is_the_last_description_surface_once_and_cites_evidence(
    mock_service, queue_conn, tmp_path
):
    _, record, _, _ = _record(mock_service, queue_conn, tmp_path)
    review = build_short_final_text_review(record)
    described = [s for s in review["surfaces"] if s["surface"].startswith("description")]
    parents = [s for s in described if s.get("template_id") == "description.short.parent"]
    assert len(parents) == 1 and described[-1] is parents[0]
    assert parents[0]["surface"].startswith("description.")
    cited = [e for e in parents[0]["evidence"] if e["kind"] == "parent_video"]
    assert [e["value"] for e in cited] == [VID]
    assert VID not in str(parents[0].get("slots", ""))


@pytest.mark.parametrize("parent", ["", "yt_abc123"])
def test_no_usable_parent_id_adds_no_link(mock_service, queue_conn, tmp_path, parent):
    _, record, _, _ = _record(mock_service, queue_conn, tmp_path, parent=parent)
    review = build_short_final_text_review(record)
    assert review["verdict"] == "agree", review["reasons"]
    assert _description(review) == "Drive through County Kerry"


def test_a_short_scheduled_before_its_parent_adds_no_link(mock_service, queue_conn, tmp_path):
    _, base, _, _ = _record(mock_service, queue_conn, tmp_path)
    at = base.schedule.scheduled_at_utc
    for parent_at, expect_link in (
        (at + timedelta(seconds=1), False),
        (at, True),
        (at - timedelta(seconds=1), True),
    ):
        record = replace(base, parent_lf_publish_at=parent_at)
        review = build_short_final_text_review(record)
        assert review["verdict"] == "agree", review["reasons"]
        assert ("https://youtu.be/" in _description(review)) is expect_link


def test_a_parent_time_without_a_timezone_adds_no_link(mock_service, queue_conn, tmp_path):
    # A naive time cannot be ordered against the Short's UTC slot; no link, no crash.
    _, base, _, _ = _record(mock_service, queue_conn, tmp_path)
    naive = base.schedule.scheduled_at_utc.replace(tzinfo=None) - timedelta(days=1)
    review = build_short_final_text_review(replace(base, parent_lf_publish_at=naive))
    assert review["verdict"] == "agree", review["reasons"]
    assert "https://youtu.be/" not in _description(review)


def test_publish_sends_the_linked_description(mock_service, queue_conn, tmp_path):
    adapter, record, did, vp = _record(mock_service, queue_conn, tmp_path)
    _publish(adapter, record, queue_conn, did, vp)
    body = mock_service.videos.return_value.insert.call_args.kwargs["body"]
    assert body["snippet"]["description"].endswith(f"Full drive: https://youtu.be/{VID}")


def test_the_waypost_pin_sites_all_name_the_new_sha():
    for rel in ("pyproject.toml", "ci/siblings.txt"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        shas = re.findall(r"waypost@([0-9a-f]{40})", text)
        assert shas == [PIN], rel
