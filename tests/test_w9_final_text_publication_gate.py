"""Waypost v4 re-pin, review plumbing with the byte-bind OFF, and the
publication gate that holds every YouTube insert until the final text is bound.

Stubs only the YouTube client (records calls), the files and the clock; the
review, the record and the pre-insert check are the real code on the pinned
seo_manager. Synthetic data, invented place names; no ``.confirmed`` file is
created or read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from seo_manager.analytics.final_text_surfaces import compose_description, compose_tags

from gomoveshift.config import Config
from gomoveshift.pipeline import final_text_record as ftr
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import phase_d_metadata as pdm
from gomoveshift.pipeline.drive_phases import youtube_uploader as yu
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.final_text_record import (
    FinalTextReviewError,
    PublicationGateError,
    build_final_text_review,
    check_final_text,
    final_text_context,
)
from gomoveshift.pipeline.title_decision import decide_title

from _state_helpers import make_state
from test_final_text_record import SUFFIX, TITLE, _metadata, _plan_lf, _state
from test_publishing_youtube_adapter import _publish, _setup, mock_service, queue_conn  # noqa: F401
from test_playlist_idempotency import (  # noqa: F401 — fixture
    _build_mock_service,
    _make_cfg,
    _make_uploader,
    fake_creds_dict,
)
from w9_drive_text import attach_drive_description

GATE = "publication_gate: final_text_bind_off"


@pytest.fixture(autouse=True)
def _no_confirmed_files(tmp_path):
    yield
    assert list(tmp_path.rglob("*.confirmed")) == []


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "lf_01.mp4"
    p.write_bytes(b"\x00\x00\x00\x20ftypiso5")
    return p


@pytest.fixture
def open_gate(monkeypatch):
    """Opened only here, in a test; no env var or flag does it."""
    monkeypatch.setattr(yu, "FINAL_TEXT_BIND_ON", True, raising=False)


def _inserts(svc) -> int:
    return svc.videos.return_value.insert.call_count


def _send(
    creds,
    video,
    state,
    meta,
    *,
    title=TITLE,
    description="d",
    tags=None,
    plan_lf=None,
):
    """``upload_if_needed`` with a recording stub client; returns (svc, record|exc)."""
    svc = _build_mock_service()
    up, mod = _make_uploader(svc, creds)
    with patch.object(mod, "MediaFileUpload", return_value=MagicMock()):
        try:
            return svc, up.upload_if_needed(
                state=state,
                lf_idx=1,
                video_path=video,
                title=title,
                description=description,
                tags=list(tags or []),
                metadata=meta,
                config=_make_cfg(),
                plan_lf=plan_lf or _plan_lf(),
            )
        except Exception as exc:  # noqa: BLE001 — returned for the assertion
            return svc, exc


# ---------------------------------------------------- publication gate


class TestPublicationGate:
    """The publication-gate hold itself, simulated by turning the shipped-on constant off."""

    @pytest.fixture(autouse=True)
    def _gate_off(self, monkeypatch):
        monkeypatch.setattr(yu, "FINAL_TEXT_BIND_ON", False)

    def test_constant_ships_on_and_no_environment_opens_it(self):
        src = Path(yu.__file__).read_text()
        assert re.search(r"^FINAL_TEXT_BIND_ON = True$", src, re.M)
        assert "os.environ" not in src and "getenv" not in src

    def test_upload_video_refuses_before_insert(self, fake_creds_dict, video):  # noqa: F811
        svc = _build_mock_service()
        up, mod = _make_uploader(svc, fake_creds_dict)
        with patch.object(mod, "MediaFileUpload", return_value=MagicMock()):
            with pytest.raises(PublicationGateError, match=GATE):
                up.upload_video(video, title="t", description="d", tags=[])
        assert _inserts(svc) == 0

    def test_upload_video_with_playlist_refuses_before_insert(
        self,
        fake_creds_dict,  # noqa: F811
        video,
    ):
        state = _state()
        meta = _metadata(state)
        svc = _build_mock_service()
        up, mod = _make_uploader(svc, fake_creds_dict)
        with patch.object(mod, "MediaFileUpload", return_value=MagicMock()):
            with pytest.raises(PublicationGateError, match=GATE):
                up.upload_video_with_playlist(
                    video,
                    title=TITLE,
                    description=compose_description(meta["final_text_review"]["surfaces"]),
                    tags=compose_tags(meta["final_text_review"]["surfaces"]),
                    metadata=meta,
                    config=_make_cfg(),
                    final_text_context=final_text_context(state, 1, _plan_lf(), meta),
                )
        assert _inserts(svc) == 0

    def test_upload_if_needed_refuses_even_for_an_agreeing_review(
        self,
        fake_creds_dict,  # noqa: F811
        video,
    ):
        state = _state()
        meta = _metadata(state)
        assert meta["final_text_review"]["verdict"] == "agree"
        surfaces = meta["final_text_review"]["surfaces"]
        svc, out = _send(
            fake_creds_dict,
            video,
            state,
            meta,
            description=compose_description(surfaces),
            tags=compose_tags(surfaces),
        )
        assert isinstance(out, PublicationGateError) and GATE in str(out), out
        assert _inserts(svc) == 0

    def test_gps_less_lf_is_held_too(self, fake_creds_dict, video):  # noqa: F811
        state = _state(per_lf_location_evidence=None, per_lf_route_record=None)
        svc, out = _send(fake_creds_dict, video, state, _metadata(state))
        assert isinstance(out, PublicationGateError), out
        assert _inserts(svc) == 0

    def test_shorts_adapter_is_held(self, mock_service, queue_conn, tmp_path):  # noqa: F811
        adapter, record, delivery_id, video_path = _setup(mock_service, queue_conn, tmp_path)
        with pytest.raises(PublicationGateError, match=GATE):
            _publish(adapter, record, queue_conn, delivery_id, video_path)
        assert _inserts(mock_service) == 0

    def test_superlong_coordinator_is_held(self, fake_creds_dict, tmp_path, caplog):  # noqa: F811
        path = tmp_path / "sl_01.mp4"
        path.write_bytes(b"x")
        rec = {
            "path": str(path),
            "part_index": 1,
            "title": "Superlong Part",
            "description": "d",
            "location": "Harrow, Westmark",
        }
        state = make_state(
            region="usa",
            superlong_parts={"2026-03-01": {"part_count": 1, "records": [rec]}},
        )
        config = Config()
        config.seo.storage_path = str(tmp_path)
        svc = _build_mock_service()
        up, mod = _make_uploader(svc, fake_creds_dict)
        # Superlong parts carry no Waypost title decision today, which refuses
        # them earlier still; stub that one check so the part reaches the gate.
        with (
            patch.object(mod, "MediaFileUpload", return_value=MagicMock()),
            patch("gomoveshift.pipeline.title_decision.require_title_decision"),
            caplog.at_level("ERROR"),
        ):
            esc._upload_superlong_parts_to_youtube(
                yt=up,
                state=state,
                output_base=tmp_path,
                config=config,
                save_state=lambda s: None,
            )
        assert _inserts(svc) == 0
        assert GATE in caplog.text


# --------------------------------------------------- kind, v3, layout (b)


class TestRecord:
    def test_every_reviewed_surface_has_a_kind(self):
        review = _metadata(_state())["final_text_review"]
        assert review["verdict"] == "agree", review
        assert review["surfaces"] and all(s.get("kind") for s in review["surfaces"])
        by_surface = {s["surface"]: s["kind"] for s in review["surfaces"]}
        assert by_surface["ost"] == by_surface["thumbnail"] == "place_line"

    def test_record_version_3_is_refused(self, monkeypatch):
        monkeypatch.setattr(ftr, "RECORD_VERSION", "final-text-record-3")
        review = _metadata(_state())["final_text_review"]
        assert review["verdict"] == "refused"
        assert "record_version:final-text-record-3" in review["reasons"]

    def test_a_stored_v3_review_is_refused_before_insert(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        meta["final_text_review"] = {
            **meta["final_text_review"],
            "record_version": "final-text-record-3",
        }
        svc, out = _send(fake_creds_dict, video, state, meta)
        assert isinstance(out, FinalTextReviewError), out
        assert _inserts(svc) == 0

    def test_review_is_sent_with_the_registered_drive_layout(self):
        from seo_manager.analytics.drive_templates import LAYOUT

        assert _metadata(_state())["final_text_review"]["description_layout"] == LAYOUT

    def test_drive_review_without_a_layout_is_refused(
        self,
        monkeypatch,
        fake_creds_dict,  # noqa: F811
        video,
        open_gate,
    ):
        state = _state()
        monkeypatch.setattr(ftr, "DESCRIPTION_LAYOUT", None)
        meta = _metadata(state)  # nothing in it needs a layout, so it still agrees
        monkeypatch.undo()
        monkeypatch.setattr(yu, "FINAL_TEXT_BIND_ON", True, raising=False)
        assert meta["final_text_review"]["description_layout"] is None
        svc, out = _send(fake_creds_dict, video, state, meta)
        assert isinstance(out, FinalTextReviewError), out
        assert "layout_missing" in out.reasons
        assert _inserts(svc) == 0


# ------------------------------------------------------ happy path (LF)


def _phase_d_metadata(tmp_path, state):
    from gomoveshift.pipeline.drive_phases.phase_d import _build_metadata_per_lf

    loc = state.per_lf_location_evidence[1]
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: attach_drive_description(
        state,
        {
            "title": TITLE,
            "description": "Reviewed description",
            "tags": ["harrow", "pellmere"],
            "title_decision": kw.get("preselected_title_decision")
            or decide_title("longform", TITLE, loc, suffix=SUFFIX),
        },
    )
    lf = _plan_lf()
    lf.total_duration = 600.0
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        return lf, _build_metadata_per_lf(
            meta_gen=gen,
            drive_plan=SimpleNamespace(regular_lfs=[lf]),
            regular_lf_paths=["/v/lf.mp4"],
            state=state,
            location="Harrow",
            publish_day_full="Wednesday",
            project_base="p",
            output_base=tmp_path,
        )


def _upload_lf(creds, video, state, lf, meta, **over):
    svc = _build_mock_service()
    up, mod = _make_uploader(svc, creds)
    args = {
        "title": meta["title"],
        "description": meta["description"],
        "tags": meta["tags"],
        **over,
    }
    with patch.object(mod, "MediaFileUpload", return_value=MagicMock()):
        out = esc._upload_lf_to_youtube(
            yt=up,
            state=state,
            lf=lf,
            lf_path=video,
            publish_at=None,
            metadata=meta,
            config=_make_cfg(),
            **args,
        )
    return svc, out


class TestHappyPath:
    def test_lf_through_phase_d_metadata_inserts_once_with_the_reviewed_text(
        self,
        fake_creds_dict,  # noqa: F811
        video,
        tmp_path,
        open_gate,
        monkeypatch,
    ):
        # This fixture's Phase-D description names only Harrow while 300 s of the
        # footage is in Pellmere; the zero-tolerance place gate (test_place_gate.py)
        # would correctly refuse it, and this test is about the reviewed-text insert.
        monkeypatch.setattr(
            "gomoveshift.pipeline.final_text_record._place_gate", lambda *a, **k: None
        )
        state = _state()
        lf, per_lf = _phase_d_metadata(tmp_path, state)
        meta = per_lf[1]
        assert meta["final_text_review"]["verdict"] == "agree"
        svc, out = _upload_lf(fake_creds_dict, video, state, lf, meta)
        assert out is not None and out["video_id"]
        assert _inserts(svc) == 1
        snippet = svc.videos.return_value.insert.call_args.kwargs["body"]["snippet"]
        assert snippet["title"] == meta["title"] == TITLE
        entries = meta["final_text_description"]
        assert snippet["description"] == compose_description(entries)
        assert snippet["tags"] == compose_tags(entries)


# --------------------------------------------------------- LF refusals


class TestRefusals:
    def _refused(self, svc, out):
        assert isinstance(out, FinalTextReviewError), out
        assert not isinstance(out, PublicationGateError), out
        assert _inserts(svc) == 0

    def test_unsupported_location(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state(per_lf_ost_text={1: "Longmere to Pellmere, Westmark"})
        title = "Longmere to Pellmere, Westmark in 4K"
        meta = _metadata(state, title=title)
        assert meta["final_text_review"]["verdict"] != "agree"
        self._refused(*_send(fake_creds_dict, video, state, meta, title=title))

    def test_missing_description_surfaces(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        meta.pop("final_text_description")
        meta["final_text_review"] = build_final_text_review(state, _plan_lf(), meta)
        assert meta["final_text_review"]["verdict"] == "refused"
        self._refused(*_send(fake_creds_dict, video, state, meta))

    def test_one_byte_title_edit(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        svc, out = _send(fake_creds_dict, video, state, meta, title=TITLE + "!")
        assert isinstance(out, Exception) and not isinstance(out, PublicationGateError), out
        assert _inserts(svc) == 0
        # past the title-decision check, the review itself pins the title
        with pytest.raises(FinalTextReviewError) as caught:
            check_final_text(
                meta["final_text_review"],
                final_text_context(state, 1, _plan_lf(), meta),
                TITLE + "!",
            )
        assert "stale:title" in caught.value.reasons

    def test_one_byte_description_edit(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        sent = compose_description(meta["final_text_review"]["surfaces"])
        self._refused(*_send(fake_creds_dict, video, state, meta, description=sent + "x"))

    def test_one_byte_tag_edit(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        self._refused(*_send(fake_creds_dict, video, state, _metadata(state), tags=["x"]))

    def test_stale_review_version(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        meta["final_text_review"] = {
            **meta["final_text_review"],
            "review_version": "final-text-review-3",
        }
        self._refused(*_send(fake_creds_dict, video, state, meta))

    def test_resumed_state_mismatch(self, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        meta = json.loads(json.dumps(_metadata(state)))
        resumed = PipelineState.from_dict(json.loads(json.dumps(state.to_dict(), default=str)))
        resumed.episode_files = [Path("/footage/changed.mp4")]
        self._refused(*_send(fake_creds_dict, video, resumed, meta))

    def test_layout_missing_in_the_stored_review(self):
        state = _state()
        meta = _metadata(state)
        review = {**meta["final_text_review"], "description_layout": None}
        with pytest.raises(FinalTextReviewError) as caught:
            check_final_text(review, final_text_context(state, 1, _plan_lf(), meta), TITLE)
        assert caught.value.reasons == ["layout_missing"]

    def test_lf_context_naming_no_layout_still_enforces_the_drive_layout(self):
        # Only a Short's context names ``description_layout=None``; an LF
        # context omits the key and must still be held to gms-drive-1.
        state = _state()
        meta = _metadata(state)
        context = final_text_context(state, 1, _plan_lf(), meta)
        assert "description_layout" not in context
        review = {**meta["final_text_review"], "description_layout": "other-layout"}
        with pytest.raises(FinalTextReviewError) as caught:
            check_final_text(review, context, TITLE)
        assert caught.value.reasons == ["layout_missing"]


# ------------------------------------- post-review edits, bytes plumbed


class TestPostReviewEdits:
    def test_review_is_built_after_the_chapter_embed(self, tmp_path):
        state = _state(per_lf_gpx_path={1: str(tmp_path / "lf1.gpx")})
        (tmp_path / "lf1.gpx").write_text("<gpx></gpx>")
        lf = _plan_lf()
        lf.total_duration = 600.0
        chapters = [
            {"timestamp": "0:00", "title": "Start"},
            {"timestamp": "2:00", "title": "Middle"},
            {"timestamp": "4:00", "title": "End"},
        ]
        loc = state.per_lf_location_evidence[1]
        meta = attach_drive_description(
            state,
            {
                "description": "Hook line",
                "title": TITLE,
                "title_decision": decide_title("longform", TITLE, loc, suffix=SUFFIX),
            },
        )
        seen = []
        real = pdm.build_final_text_review

        def spy(st, plan_lf, metadata):
            seen.append(metadata["description"])
            return real(st, plan_lf, metadata)

        with (
            patch.object(pdm, "build_final_text_review", spy),
            patch.object(pdm, "_lf_chapters", return_value=(chapters, "0:00 Start")),
            patch.object(pdm, "_below_chapter_floor", return_value=False),
            patch(
                "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
                return_value=SimpleNamespace(passed=True, message=""),
            ),
        ):
            pdm._generate_chapters_all_lfs(
                config=MagicMock(),
                drive_plan=SimpleNamespace(regular_lfs=[lf]),
                state=state,
                metadata_per_lf={1: meta},
                location="Harrow",
                project_base="p",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
        assert seen and "0:00 Start" in seen[-1]
        assert meta["description"] == seen[-1]
        assert meta["final_text_review"]["verdict"] == "agree"

    def test_exact_description_and_tag_bytes_reach_the_check(
        self,
        monkeypatch,
        fake_creds_dict,  # noqa: F811
        video,
        open_gate,
    ):
        state = _state()
        meta = _metadata(state)
        surfaces = meta["final_text_review"]["surfaces"]
        sent, sent_tags = compose_description(surfaces), compose_tags(surfaces)
        calls = []
        real = ftr.check_final_text

        def spy(review, context, title, description=None, tags=None):
            calls.append((description, tags))
            return real(review, context, title, description=description, tags=tags)

        monkeypatch.setattr(ftr, "check_final_text", spy)
        svc, out = _send(fake_creds_dict, video, state, meta, description=sent, tags=sent_tags)
        assert not isinstance(out, Exception), out
        assert calls == [(sent, sent_tags)]  # exact bytes, blank lines included

    @pytest.mark.parametrize("edit", ["one_byte", "blank_line"])
    def test_description_mismatch_is_refused(self, edit, fake_creds_dict, video, open_gate):  # noqa: F811
        state = _state()
        meta = _metadata(state)
        sent = compose_description(meta["final_text_review"]["surfaces"])
        sent = sent + "x" if edit == "one_byte" else sent.replace("\n", "\n\n", 1)
        svc, out = _send(fake_creds_dict, video, state, meta, description=sent)
        assert isinstance(out, FinalTextReviewError), out
        assert _inserts(svc) == 0

    def test_switch_on_compares_the_exact_description(self, monkeypatch):
        state = _state()
        meta = _metadata(state)
        review = meta["final_text_review"]
        context = final_text_context(state, 1, _plan_lf(), meta)
        monkeypatch.setattr(ftr, "BIND_DESCRIPTION_TAGS", True)
        sent = compose_description(review["surfaces"])
        check_final_text(review, context, TITLE, description=sent, tags=[])
        for bad in (sent + "x", sent.replace("\n", "\n\n", 1)):
            with pytest.raises(FinalTextReviewError) as caught:
                check_final_text(review, context, TITLE, description=bad, tags=[])
            assert "stale:description" in caught.value.reasons
