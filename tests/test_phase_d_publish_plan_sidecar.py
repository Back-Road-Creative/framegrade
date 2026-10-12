"""Phase D publish-plan sidecar: ADDITIVE to ``_emit_consolidated_shorts_csv``
-- writes ``<project>_publish_plan.json`` iff a ``publishing_route_*`` is
``"first_party"``, and never alters the CSV's rows/gates itself (skips its
Drive-media gate). ``_emit_consolidated_shorts_csv`` separately suppresses
that same platform's Metricool toggle (via ``platforms_for_publishing_routes``)
so the sidecar and the CSV never both post the same short.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.drive_phases.phase_d import (
    PhaseD,
    _emit_consolidated_shorts_csv,
    _emit_publish_plan_sidecar,
)
from gomoveshift.publishing.models import MediaFingerprint

sys.path.insert(0, str(Path(__file__).parent))

from test_phase_d import (  # noqa: E402
    _install_fake_youtube_uploader,
    _phase_d_patches,
    _setup_mocks,
    get_config,
)
from test_phase_d import phase_d_state as phase_d_state  # noqa: E402,F401 -- fixture re-export

FP = MediaFingerprint(
    path="/x/short_01.mp4",
    size_bytes=1,
    duration_seconds=9.0,
    width=1080,
    height=1920,
    codec="h264",
    sha256="a" * 64,
)
FINGERPRINT = "gomoveshift.publishing.media_fingerprint.fingerprint_media"


def _slot(idx=10):
    return SimpleNamespace(short_index=idx, date="2026-03-15", time="08:00:00")


def _base_kwargs(media_dir, **route):
    # Every route is spelled "off" explicitly (DriveConfig's own default is
    # also off) so a test that flips exactly one platform can still assert
    # the other three stayed unselected.
    all_off = {
        "publishing_route_youtube": "off",
        "publishing_route_instagram": "off",
        "publishing_route_facebook": "off",
        "publishing_route_tiktok": "off",
    }
    all_off.update(route)
    return dict(
        project_base="p",
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
        social_posts_per_lf={1: [SimpleNamespace(short_index=10)]},
        drive_schedule=SimpleNamespace(shorts_slots=[_slot()]),
        youtube_uploads={1: {"video_id": "vid-1", "publish_at": "2026-03-14T08:00:00-05:00"}},
        per_lf_recording_date={1: "2026-03-14"},
        shorts_paths_by_lf={1: [str(media_dir / "short_01.mp4")]},
        publish_pass=True,
        config=SimpleNamespace(drive=DriveConfig(**all_off)),
    )


class TestSidecar:
    def test_default_routes_never_fingerprints_or_writes(self, tmp_path):
        with patch(FINGERPRINT) as fp:
            result = _emit_publish_plan_sidecar(output_base=tmp_path, **_base_kwargs(tmp_path))
        assert result is None
        fp.assert_not_called()
        assert not (tmp_path / "p_publish_plan.json").exists()

    def test_tiktok_audience_defaults_to_private_without_operator_opt_in(self, tmp_path):
        # The wiring that makes the config field load-bearing. Without it the
        # first audited TikTok run publishes every short to everyone.
        kwargs = _base_kwargs(tmp_path, publishing_route_tiktok="first_party")
        with patch(FINGERPRINT, return_value=FP):
            plan_path = _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        payload = json.loads(plan_path.read_text())
        assert payload["shorts"][0]["payloads"]["tiktok"]["privacy_level"] == "SELF_ONLY"

    def test_tiktok_audience_follows_the_configured_privacy_level(self, tmp_path):
        kwargs = _base_kwargs(
            tmp_path,
            publishing_route_tiktok="first_party",
            publishing_tiktok_privacy_level="PUBLIC_TO_EVERYONE",
        )
        with patch(FINGERPRINT, return_value=FP):
            plan_path = _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        payload = json.loads(plan_path.read_text())
        assert payload["shorts"][0]["payloads"]["tiktok"]["privacy_level"] == "PUBLIC_TO_EVERYONE"

    def test_first_party_route_writes_only_that_platform(self, tmp_path):
        kwargs = _base_kwargs(tmp_path, publishing_route_youtube="first_party")
        with patch(FINGERPRINT, return_value=FP) as fp:
            result = _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        assert result == tmp_path / "p_publish_plan.json"
        fp.assert_called_once()
        record = json.loads(result.read_text(encoding="utf-8"))["shorts"][0]
        assert record["routes"] == {"selected": ["youtube"]}
        assert record["media"]["sha256"] == "a" * 64

    def test_written_without_drive_links_csv_gate_does_not_apply(self, tmp_path):
        """Non-obvious: the CSV's Drive-media gate does NOT apply here — a
        first-party route needs the LOCAL file, not a Drive link."""
        kwargs = _base_kwargs(tmp_path, publishing_route_tiktok="first_party")
        csv_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("config", "shorts_paths_by_lf", "publish_pass")
        }
        csv_result = _emit_consolidated_shorts_csv(
            output_base=tmp_path,
            upload_manifests={},
            per_lf_trim_ratio={},
            skip_drive_upload=True,
            **csv_kwargs,
        )
        assert csv_result is None  # CSV deferred — no Drive links

        with patch(FINGERPRINT, return_value=FP):
            plan_result = _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        assert plan_result is not None
        assert plan_result.exists()

    def test_identical_inputs_are_byte_identical(self, tmp_path):
        def build(out_dir):
            out_dir.mkdir()
            with patch(FINGERPRINT, return_value=FP):
                return _emit_publish_plan_sidecar(
                    output_base=out_dir,
                    **_base_kwargs(out_dir, publishing_route_instagram="first_party"),
                )

        assert build(tmp_path / "a").read_bytes() == build(tmp_path / "b").read_bytes()

    def test_fingerprint_error_propagates_and_writes_nothing(self, tmp_path):
        kwargs = _base_kwargs(tmp_path, publishing_route_facebook="first_party")
        with patch(FINGERPRINT, side_effect=FileNotFoundError("gone")):
            with pytest.raises(FileNotFoundError):
                _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        assert not (tmp_path / "p_publish_plan.json").exists()


class TestCsvSuppressesFlippedRoutePlatform:
    """A flipped ``publishing_route_*`` must NOT double-post: the sidecar
    above starts posting that platform directly, so the Metricool CSV's
    toggle for the SAME platform must flip to ``"FALSE"`` -- everything
    else in the row stays identical to the default-route CSV.
    """

    def test_csv_youtube_toggle_flips_false_everything_else_unchanged(self, tmp_path):
        import csv

        manifest = SimpleNamespace(files={"short_10.mp4": "https://drive.google.com/x"})
        seo = SimpleNamespace(storage_path=str(tmp_path))
        common = dict(
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
            social_posts_per_lf={1: [SimpleNamespace(short_index=10)]},
            upload_manifests={1: manifest},
            drive_schedule=SimpleNamespace(shorts_slots=[_slot()]),
            youtube_uploads={1: {"video_id": "vid-1"}},
            per_lf_trim_ratio={1: 0.4},
            per_lf_recording_date={1: "2026-03-14"},
        )
        dir_default, dir_flipped = tmp_path / "default", tmp_path / "flipped"
        dir_default.mkdir()
        dir_flipped.mkdir()

        path_default = _emit_consolidated_shorts_csv(
            output_base=dir_default,
            project_base="p",
            config=SimpleNamespace(drive=DriveConfig(publishing_route_youtube="off"), seo=seo),
            **common,
        )
        path_flipped = _emit_consolidated_shorts_csv(
            output_base=dir_flipped,
            project_base="p",
            config=SimpleNamespace(
                drive=DriveConfig(publishing_route_youtube="first_party"), seo=seo
            ),
            **common,
        )
        with open(path_default, encoding="utf-8") as f:
            row_default = next(csv.DictReader(f))
        with open(path_flipped, encoding="utf-8") as f:
            row_flipped = next(csv.DictReader(f))

        assert row_default["YouTube"] == "TRUE"
        assert row_flipped["YouTube"] == "FALSE"
        for col in row_default:
            if col != "YouTube":
                assert row_default[col] == row_flipped[col], col


class TestSkipUploadRun:
    def test_first_party_route_without_uploads_defers_instead_of_raising(self, tmp_path, caplog):
        """A pass with no LF video id (``--skip-youtube-upload``) would make the
        builder raise PUBLISH-PLAN-NO-PARENT-ID and fail the run on its last step.
        Defer with a warning; nothing is fingerprinted or written."""
        kwargs = _base_kwargs(tmp_path, publishing_route_youtube="first_party")
        kwargs["youtube_uploads"] = {}
        with patch(FINGERPRINT) as fp, caplog.at_level("WARNING"):
            result = _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        assert result is None
        fp.assert_not_called()
        assert not (tmp_path / "p_publish_plan.json").exists()
        assert any("DEFERRED" in r.message for r in caplog.records)


class TestAuditPassNeverWritesThePlan:
    """Only a publish pass writes the plan. An audit pass enforces neither the
    verified-place gate nor the hand-title gate -- it only warns -- yet a re-run
    over an already-uploaded drive still carries every LF's video id. Writing
    the plan there hands ``enqueue-plan --apply`` shorts no gate refused: the
    wrong-place short closed on the publish pass, reopened by hand."""

    def test_audit_pass_with_recorded_uploads_writes_nothing(self, tmp_path, caplog):
        kwargs = _base_kwargs(tmp_path, publishing_route_youtube="first_party")
        kwargs["publish_pass"] = False
        with patch(FINGERPRINT) as fp, caplog.at_level("WARNING"):
            result = _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        assert result is None
        fp.assert_not_called()
        assert not (tmp_path / "p_publish_plan.json").exists()
        assert any("DEFERRED" in r.message and "audit" in r.message for r in caplog.records)

    def test_audit_pass_leaves_the_publish_pass_plan_untouched(self, tmp_path):
        plan_path = tmp_path / "p_publish_plan.json"
        plan_path.write_text('{"written": "by the publish pass"}\n', encoding="utf-8")
        kwargs = _base_kwargs(tmp_path, publishing_route_tiktok="first_party")
        kwargs["publish_pass"] = False
        with patch(FINGERPRINT, return_value=FP):
            _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)
        assert plan_path.read_text(encoding="utf-8") == '{"written": "by the publish pass"}\n'

    def test_every_caller_must_say_which_pass_it_is(self, tmp_path):
        kwargs = _base_kwargs(tmp_path, publishing_route_youtube="first_party")
        del kwargs["publish_pass"]
        with pytest.raises(TypeError, match="publish_pass"):
            _emit_publish_plan_sidecar(output_base=tmp_path, **kwargs)


EMIT = "gomoveshift.pipeline.drive_phases.phase_d._emit_publish_plan_sidecar"


class TestPhaseDNamesItsPass:
    """``PhaseD.run`` hands the emitter the run's publish intent -- the same
    ``bool(state.youtube_upload)`` the place and title gates enforce on."""

    @staticmethod
    def _run(state):
        with (
            patch(EMIT) as emit,
            _phase_d_patches()[0] as m0,
            _phase_d_patches()[1] as m1,
            _phase_d_patches()[2] as m2,
            _phase_d_patches()[3] as m3,
            _phase_d_patches()[4] as m4,
            _phase_d_patches()[5] as m5,
            _phase_d_patches()[6] as m6,
            _phase_d_patches()[7] as m7,
        ):
            _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7))
            PhaseD(get_config()).run(state)
        emit.assert_called_once()
        return emit.call_args.kwargs

    def test_publish_run_is_a_publish_pass(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)
        assert self._run(phase_d_state)["publish_pass"] is True

    def test_audit_rerun_over_an_uploaded_drive_is_not(self, phase_d_state, monkeypatch):
        _install_fake_youtube_uploader(monkeypatch)
        phase_d_state.youtube_upload = False
        phase_d_state.skip_upload = True
        # The publish pass already ran: every LF's video id is still in state.
        phase_d_state.youtube_uploads = {1: {"video_id": "v1"}, 2: {"video_id": "v2"}}
        kwargs = self._run(phase_d_state)
        assert kwargs["publish_pass"] is False
        assert kwargs["youtube_uploads"]  # the ids alone never made it a publish pass
