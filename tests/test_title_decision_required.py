"""A title with no Waypost decision never reaches an upload call."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from test_publishing_plan_builder import _build, _post
from test_playlist_idempotency import (
    _build_mock_service,
    _make_cfg,
    _make_uploader,
    fake_creds_dict,  # noqa: F401
)
from _state_helpers import make_state

from gomoveshift.pipeline import title_decision as td


@pytest.fixture(autouse=True)
def _publication_gate_open(monkeypatch):
    # The publication gate holds every insert until the final text is bound;
    # these upload tests predate the gate, so they open it in-test only.
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.youtube_uploader.FINAL_TEXT_BIND_ON", True
    )


TITLE = "Alpha Town to Beta Town, ST in 4K"


def _decision(title=TITLE):
    return td.decide_title("longform", title, None, suffix=" in 4K")


@pytest.fixture
def video(tmp_path: Path) -> Path:
    p = tmp_path / "lf_01.mp4"
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


class TestRequireTitleDecision:
    def test_missing_decision_is_refused_by_name(self):
        with pytest.raises(td.TitleDecisionMissingError, match="no Waypost title decision"):
            td.require_title_decision(TITLE, {"title": TITLE})

    @pytest.mark.parametrize("metadata", [None, {}, {"title_decision": None}, "text"])
    def test_absent_shapes_are_refused(self, metadata):
        with pytest.raises(td.TitleDecisionMissingError):
            td.require_title_decision(TITLE, metadata)

    def test_matching_decision_passes(self):
        td.require_title_decision(TITLE, {"title_decision": _decision()})

    def test_mismatch_still_refused(self):
        with pytest.raises(td.TitleDecisionMismatchError):
            td.require_title_decision("Other", {"title_decision": _decision()})


class TestLongformUpload:
    def _up(self, creds):
        svc = _build_mock_service()
        up, yu = _make_uploader(svc, creds)
        return svc, up, yu

    def test_upload_without_decision_refuses_before_insert(self, fake_creds_dict, video):  # noqa: F811
        svc, up, yu = self._up(fake_creds_dict)
        with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
            with pytest.raises(td.TitleDecisionMissingError):
                up.upload_if_needed(
                    state=make_state(),
                    lf_idx=1,
                    video_path=video,
                    title=TITLE,
                    description="d",
                    tags=[],
                    metadata={"location": "Somewhere"},
                    config=_make_cfg(),
                )
        svc.videos.return_value.insert.assert_not_called()

    def test_playlist_helper_without_decision_refuses(self, fake_creds_dict, video):  # noqa: F811
        svc, up, yu = self._up(fake_creds_dict)
        with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
            with pytest.raises(td.TitleDecisionMissingError):
                up.upload_video_with_playlist(
                    video,
                    title=TITLE,
                    description="d",
                    tags=[],
                    metadata={"location": "Somewhere"},
                    config=_make_cfg(),
                )
        svc.videos.return_value.insert.assert_not_called()

    def test_decided_title_uploads(self, fake_creds_dict, video):  # noqa: F811
        svc, up, yu = self._up(fake_creds_dict)
        with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
            record = up.upload_if_needed(
                state=make_state(),
                lf_idx=1,
                video_path=video,
                title=TITLE,
                description="d",
                tags=[],
                metadata={"title_decision": _decision()},
                config=_make_cfg(),
            )
        assert record["video_id"] == "VIDEO_ID_1"

    def test_already_uploaded_resume_makes_no_call_and_needs_no_decision(
        self,
        fake_creds_dict,  # noqa: F811
        video,
    ):
        svc, up, yu = self._up(fake_creds_dict)
        state = make_state(
            youtube_uploads={
                1: {"video_id": "V", "playlist_id": "P", "title": "t", "publish_at": None}
            }
        )
        record = up.upload_if_needed(
            state=state,
            lf_idx=1,
            video_path=video,
            title=TITLE,
            description="d",
            tags=[],
            metadata={"location": "Somewhere"},
            config=_make_cfg(),
        )
        assert record["video_id"] == "V"
        svc.videos.return_value.insert.assert_not_called()


class TestShortsPlan:
    def test_plan_refuses_a_short_without_a_decision(self):
        with pytest.raises(td.TitleDecisionMissingError):
            _build(posts=[_post(title_decision=None)])

    def test_plan_builds_with_a_matching_decision(self):
        title = "County Kerry Drive #1"
        decision = td.decide_title("short", title, None, suffix="")
        record = _build(posts=[_post(title_decision=decision)]).shorts[0]
        assert record.payloads.youtube.title == title


class TestHistoricalArtifacts:
    """Rule: already-uploaded items make no call and are not refused; any NEW
    title write of an item with no decision is refused by name."""

    def test_orphan_playlist_recovery_over_an_uploaded_lf_is_not_refused(
        self,
        fake_creds_dict,  # noqa: F811
        video,
    ):
        svc = _build_mock_service()
        up, yu = _make_uploader(svc, fake_creds_dict)
        state = make_state(youtube_uploads={1: {"video_id": "V", "title": "t", "publish_at": None}})
        with patch.object(yu, "MediaFileUpload", return_value=MagicMock()):
            record = up.upload_if_needed(
                state=state,
                lf_idx=1,
                video_path=video,
                title=TITLE,
                description="d",
                tags=[],
                metadata={"location": "Somewhere"},
                config=_make_cfg(),
            )
        assert record["video_id"] == "V"
        svc.videos.return_value.insert.assert_not_called()

    def test_pre_decision_short_sidecar_loads_but_cannot_build_a_plan(self):
        from gomoveshift.pipeline.social_generator import SocialPost

        data = SocialPost(
            short_index=1,
            caption="c",
            youtube_title="Synthetic Drive #1",
            youtube_tags="t",
            youtube_category="24",
            tiktok_title="tt",
            pinterest_board="b",
            first_comment="fc",
        ).to_dict()
        data.pop("title_decision")
        post = SocialPost(**data)
        assert post.title_decision is None
        with pytest.raises(td.TitleDecisionMissingError):
            _build(posts=[_post(title_decision=post.title_decision)])
