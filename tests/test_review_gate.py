"""Tests for ReviewGate -- human-review checkpoints at pipeline phase boundaries."""

import json
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.review_gate import ReviewFailedError, ReviewGate

from _state_helpers import make_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ffprobe_streams_response(streams: list[dict]) -> MagicMock:
    """Return a mock subprocess.run result containing an ffprobe stream listing."""
    payload = {"streams": streams}
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps(payload)
    return mock


def _ffprobe_format_response(bit_rate: str = "500000000") -> MagicMock:
    """Return a mock subprocess.run result for format/bitrate probe."""
    payload = {"format": {"bit_rate": bit_rate}}
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps(payload)
    return mock


def _make_video_stream(duration: float = 600.0) -> dict:
    return {"codec_type": "video", "duration": str(duration)}


def _make_audio_stream(duration: float = 600.0) -> dict:
    return {"codec_type": "audio", "duration": str(duration)}


def _make_tmcd_stream() -> dict:
    return {"codec_type": "data", "codec_tag_string": "tmcd"}


def _gate_b_state(processed_files: dict | None = None) -> dict:
    return {"processed_files": processed_files or {}}


def _good_ffprobe_side_effect(cmd, **kwargs):
    """Side-effect that returns healthy 2-stream, high-bitrate, low-drift data."""
    two_streams = [_make_video_stream(600.0), _make_audio_stream(600.0)]
    if "-show_streams" in cmd:
        return _ffprobe_streams_response(two_streams)
    if "-show_format" in cmd:
        return _ffprobe_format_response("500000000")
    return MagicMock(returncode=0, stdout="{}")


# ---------------------------------------------------------------------------
# TestStreamCheck
# ---------------------------------------------------------------------------


class TestStreamCheck:
    """_stream_count and gate_b stream validation."""

    def test_two_stream_passes(self, tmp_path):
        """Two streams (video + audio) should not raise."""
        gate = ReviewGate(interactive=False)
        streams = [_make_video_stream(), _make_audio_stream()]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _ffprobe_streams_response(streams)
            count = gate._stream_count(tmp_path / "ep.mp4")
        assert count == 2

    def test_three_stream_passes_gate_b(self, tmp_path):
        """Three streams (video + audio + tmcd) pass gate_b (>= 1 required)."""
        gate = ReviewGate(interactive=False)
        final_file = tmp_path / "ep_001_FINAL.mp4"
        final_file.touch()

        three_streams = [_make_video_stream(), _make_audio_stream(), _make_tmcd_stream()]
        format_resp = _ffprobe_format_response("500000000")

        def mock_run_side_effect(cmd, **kwargs):
            if "-show_streams" in cmd:
                return _ffprobe_streams_response(three_streams)
            if "-show_format" in cmd:
                return format_resp
            return MagicMock(returncode=0, stdout="{}")

        state = make_state(processed_files={0: str(final_file)})
        with patch("subprocess.run", side_effect=mock_run_side_effect):
            gate.gate_b(state=state, output_dir=tmp_path)

    def test_ffprobe_error_treated_as_zero_streams(self, tmp_path):
        """ffprobe failure returns 0 streams -- gate_b must raise ReviewFailedError."""
        gate = ReviewGate(interactive=False)
        final_file = tmp_path / "ep_001_FINAL.mp4"
        final_file.touch()

        error_resp = MagicMock(returncode=1, stdout="", stderr="error")

        def mock_run_side_effect(cmd, **kwargs):
            if "-show_streams" in cmd:
                return error_resp
            if "-show_format" in cmd:
                return _ffprobe_format_response("500000000")
            return MagicMock(returncode=0, stdout="{}")

        state = make_state(processed_files={0: str(final_file)})
        with patch("subprocess.run", side_effect=mock_run_side_effect):
            with pytest.raises(ReviewFailedError):
                gate.gate_b(state=state, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# TestBitrateCheck
# ---------------------------------------------------------------------------


class TestBitrateCheck:
    """Bitrate floor enforcement in gate_b."""

    def test_above_floor_passes(self, tmp_path):
        """Bitrate above 50 Mbps floor should pass."""
        gate = ReviewGate(interactive=False)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _ffprobe_format_response("500000000")  # 500 Mbps in bps
            kbps_val = gate._probe_bitrate_kbps(tmp_path / "ep.mp4")
        # 500000000 bps / 1000 = 500000 kbps >> 50000 kbps
        assert kbps_val > ReviewGate.MIN_BITRATE_KBPS

    def test_below_floor_fails_gate_b(self, tmp_path):
        """Bitrate below 50 Mbps floor raises ReviewFailedError in gate_b."""
        gate = ReviewGate(interactive=False)
        final_file = tmp_path / "ep_001_FINAL.mp4"
        final_file.touch()

        two_streams = [_make_video_stream(), _make_audio_stream()]
        low_bitrate_resp = _ffprobe_format_response("10000000")  # 10 Mbps in bps -> 10000 kbps

        def mock_run_side_effect(cmd, **kwargs):
            if "-show_streams" in cmd:
                return _ffprobe_streams_response(two_streams)
            if "-show_format" in cmd:
                return low_bitrate_resp
            return MagicMock(returncode=0, stdout="{}")

        state = make_state(processed_files={0: str(final_file)})
        with patch("subprocess.run", side_effect=mock_run_side_effect):
            with pytest.raises(ReviewFailedError, match="[Bb]itrate"):
                gate.gate_b(state=state, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# TestDriftCheck
# ---------------------------------------------------------------------------


class TestDriftCheck:
    """A/V duration drift enforcement."""

    def test_low_drift_passes(self, tmp_path):
        """Drift under 2% passes."""
        gate = ReviewGate(interactive=False)
        # video=600.0, audio=601.0 -> drift = 1/600 * 100 ~ 0.17%
        streams = [_make_video_stream(600.0), _make_audio_stream(601.0)]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = _ffprobe_streams_response(streams)
            drift = gate._av_drift_pct(tmp_path / "ep.mp4")
        assert drift < ReviewGate.MAX_AV_DRIFT_PCT

    def test_high_drift_fails_gate_b(self, tmp_path):
        """Drift >= 2% raises ReviewFailedError in gate_b."""
        gate = ReviewGate(interactive=False)
        final_file = tmp_path / "ep_001_FINAL.mp4"
        final_file.touch()

        # video=600, audio=500 -> drift = 100/600 * 100 ~ 16.7%
        drifted_streams = [_make_video_stream(600.0), _make_audio_stream(500.0)]
        format_resp = _ffprobe_format_response("500000000")

        def mock_run_side_effect(cmd, **kwargs):
            if "-show_streams" in cmd:
                return _ffprobe_streams_response(drifted_streams)
            if "-show_format" in cmd:
                return format_resp
            return MagicMock(returncode=0, stdout="{}")

        state = make_state(processed_files={0: str(final_file)})
        with patch("subprocess.run", side_effect=mock_run_side_effect):
            with pytest.raises(ReviewFailedError, match="[Dd]rift"):
                gate.gate_b(state=state, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# TestPlaceholderCheck
# ---------------------------------------------------------------------------


class TestPlaceholderCheck:
    """Placeholder pattern enforcement in metadata JSON files."""

    def test_clean_json_passes(self, tmp_path):
        """JSON with no placeholder patterns passes."""
        gate = ReviewGate(interactive=False)
        clean = {"title": "Kerry Coastline Drive -- Part 1", "description": "Scenic drive"}
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(clean))
        hits = gate._check_placeholders(json_path)
        assert hits == []

    def test_field_names_are_not_placeholders(self, tmp_path):
        """Key names like ``insert_count`` and ``template_id`` are schema, not
        text; only string values are checked."""
        gate = ReviewGate(interactive=False)
        data = {
            "title": "Beaufort County, SC in 4K",
            "insert_count": 0,
            "title_decision": {"candidates": [{"template_id": "region_only_lf"}]},
        }
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(data))
        assert gate._check_placeholders(json_path) == []

    def test_final_text_review_record_is_not_scanned(self, tmp_path):
        """The embedded final-text review carries policy ids such as
        ``drive-templates-4``; its own gate judges it."""
        gate = ReviewGate(interactive=False)
        data = {
            "title": "Beaufort County, SC in 4K",
            "final_text_review": {"text_policy": {"drive_version": "drive-templates-4"}},
        }
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(data))
        assert gate._check_placeholders(json_path) == []

    def test_placeholder_in_a_nested_value_is_caught(self, tmp_path):
        """A placeholder anywhere in a string value still fails."""
        gate = ReviewGate(interactive=False)
        data = {"chapters": [{"title": "INSERT_TOWN here"}], "tags": ["TEMPLATE tag"]}
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(data))
        assert gate._check_placeholders(json_path) == ["INSERT_", "TEMPLATE"]

    def test_final_text_entry_provenance_is_not_scanned(self, tmp_path):
        """Waypost description entries label their source (``kind: template``);
        only each entry's ``text`` ships."""
        gate = ReviewGate(interactive=False)
        data = {
            "title": "Beaufort County, SC in 4K",
            "final_text_description": [
                {"surface": "description.rule", "kind": "template", "text": "---\n"},
                {"surface": "description.intro", "kind": "param_template", "text": "Drive.\n"},
            ],
        }
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(data))
        assert gate._check_placeholders(json_path) == []

    def test_placeholder_in_final_text_entry_text_is_caught(self, tmp_path):
        """The entry text is what uploads, so a placeholder there still fails."""
        gate = ReviewGate(interactive=False)
        data = {"final_text_description": [{"kind": "template", "text": "INSERT_TOWN"}]}
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(data))
        assert gate._check_placeholders(json_path) == ["INSERT_"]

    def test_guess_the_location_fails_gate_cd(self, tmp_path):
        """'guess the location' in JSON raises ReviewFailedError in gate_cd."""
        gate = ReviewGate(interactive=False)
        bad = {"title": "Can you guess the location?", "description": "Drive video"}
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(bad))

        state = make_state(
            regular_lf_paths=[],
            episode_dates=["2026-03-01"],
        )
        with pytest.raises(ReviewFailedError, match="[Pp]laceholder"):
            gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-03-01")

    def test_todo_string_fails_gate_cd(self, tmp_path):
        """'TODO' in JSON raises ReviewFailedError in gate_cd."""
        gate = ReviewGate(interactive=False)
        bad = {"title": "Drive Video TODO: update title", "description": "Drive"}
        json_path = tmp_path / "ep_001_youtube.json"
        json_path.write_text(json.dumps(bad))

        state = make_state(
            regular_lf_paths=[],
            episode_dates=["2026-03-01"],
        )
        with pytest.raises(ReviewFailedError, match="[Pp]laceholder"):
            gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-03-01")


# ---------------------------------------------------------------------------
# TestCSVColumnCheck
# ---------------------------------------------------------------------------


class TestCSVColumnCheck:
    """Metricool CSV column count check (warn-only, not hard-fail)."""

    def test_correct_cols_passes(self, tmp_path):
        """CSV with expected column count returns that count."""
        gate = ReviewGate(interactive=False)
        csv_path = tmp_path / "ep_001_metricool.csv"
        header = ",".join([f"col{i}" for i in range(31)])
        csv_path.write_text(header + "\n")
        assert gate._count_csv_columns(csv_path) == 31

    def test_wrong_cols_warns_but_does_not_fail_gate_cd(self, tmp_path):
        """CSV with wrong column count warns but does NOT raise (warn-only per spec)."""
        gate = ReviewGate(interactive=False)
        csv_path = tmp_path / "ep_001_metricool.csv"
        header = ",".join([f"col{i}" for i in range(20)])
        csv_path.write_text(header + "\n")

        state = make_state(
            regular_lf_paths=[],
            episode_dates=["2026-03-01"],
        )
        # Should NOT raise -- CSV column mismatch is a warn, not a hard-fail
        gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-03-01")


# ---------------------------------------------------------------------------
# TestGatePreview
# ---------------------------------------------------------------------------


class TestGatePreview:
    """gate_preview behaviour."""

    def test_no_preview_files_fails(self, tmp_path):
        """gate_preview with no preview clips raises ReviewFailedError."""
        gate = ReviewGate(interactive=False)
        # Create the preview directory but leave it empty
        preview_dir = tmp_path / "_stages" / "preview"
        preview_dir.mkdir(parents=True)
        with pytest.raises(ReviewFailedError, match="[Pp]review"):
            gate.gate_preview(output_dir=tmp_path)

    def test_preview_files_exist_passes(self, tmp_path):
        """gate_preview with at least one clip passes (no prompt in non-interactive)."""
        gate = ReviewGate(interactive=False)
        preview_dir = tmp_path / "_stages" / "preview"
        preview_dir.mkdir(parents=True)
        clip = preview_dir / "ep_001_preview.mp4"
        clip.touch()

        with patch("subprocess.run", side_effect=_good_ffprobe_side_effect):
            # Should not raise
            gate.gate_preview(output_dir=tmp_path)


# ---------------------------------------------------------------------------
# TestHumanPrompt
# ---------------------------------------------------------------------------


class TestHumanPrompt:
    """_prompt behaviour with various user inputs."""

    def test_y_input_continues(self):
        """Input 'y' passes the prompt without raising."""
        gate = ReviewGate(interactive=True)
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            gate._prompt("Continue?")  # should not raise

    def test_n_input_raises(self):
        """Input 'n' raises ReviewFailedError."""
        gate = ReviewGate(interactive=True)
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
        ):
            with pytest.raises(ReviewFailedError):
                gate._prompt("Continue?")

    def test_empty_input_raises(self):
        """Empty input (Enter) raises ReviewFailedError (default = No)."""
        gate = ReviewGate(interactive=True)
        with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value=""):
            with pytest.raises(ReviewFailedError):
                gate._prompt("Continue?")


# ---------------------------------------------------------------------------
# TestNoReviewMode
# ---------------------------------------------------------------------------


class TestNoReviewMode:
    """interactive=False skips prompts but not automated checks."""

    def test_skips_prompt_on_clean_data(self, tmp_path):
        """Non-interactive mode auto-passes human prompt on clean gate_a data."""
        gate = ReviewGate(interactive=False)
        state = make_state(
            episode_files=["/tmp/ep_001.mp4"],
            episode_dates=["2026-03-01"],
            file_durations=[3600.0],
            gps_per_episode=[{"lat": 51.77, "lon": -9.98}],
        )
        # Should not raise and not prompt input
        with patch("builtins.input") as mock_input:
            gate.gate_a(state=state, output_dir=tmp_path, location="County Kerry, Ireland")
            mock_input.assert_not_called()

    def test_still_fails_automated_check_in_no_review_mode(self, tmp_path):
        """interactive=False still raises ReviewFailedError on automated failures."""
        gate = ReviewGate(interactive=False)
        # Empty episode_files -> automated fail
        state = make_state(
            episode_files=[],
            episode_dates=[],
            file_durations=[],
            gps_per_episode=[],
        )
        with pytest.raises(ReviewFailedError):
            gate.gate_a(state=state, output_dir=tmp_path, location="County Kerry, Ireland")


# ---------------------------------------------------------------------------
# TestDriveConfigIntegration
# ---------------------------------------------------------------------------


class TestDriveConfigIntegration:
    """DriveConfig.no_review field and orchestrator wiring."""

    def test_drive_config_no_review_default_false(self):
        """DriveConfig.no_review defaults to False."""
        from gomoveshift.config import DriveConfig

        cfg = DriveConfig()
        assert cfg.no_review is False

    def test_drive_config_no_review_true(self):
        """DriveConfig.no_review can be set to True."""
        from gomoveshift.config import DriveConfig

        cfg = DriveConfig(no_review=True)
        assert cfg.no_review is True

    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_review_failed_error_stops_pipeline(self, mock_a, tmp_path):
        """ReviewFailedError during gate_a stops the pipeline with failed result."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        mock_a.side_effect = ReviewFailedError("No episode files found")

        source = tmp_path / "source"
        source.mkdir()

        cfg = Config()
        cfg.drive.no_review = True  # skip prompts but automated checks still run
        orch = DrivePipelineOrchestrator(cfg)
        result = orch.run(source_dir=source, output_base=tmp_path / "out")

        assert result.success is False
        assert any("ReviewFailedError" in e for e in result.errors)

    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator"
        "._phase_a_ingest_and_analyze"
    )
    def test_no_review_creates_non_interactive_gate(self, mock_a, mock_bc, mock_d, tmp_path):
        """When config.drive.no_review=True, ReviewGate is created with interactive=False."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        source = tmp_path / "source"
        source.mkdir()

        cfg = Config()
        cfg.drive.no_review = True

        with patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate") as mock_review_cls:
            mock_gate_instance = MagicMock()
            mock_review_cls.return_value = mock_gate_instance

            orch = DrivePipelineOrchestrator(cfg)
            orch.run(source_dir=source, output_base=tmp_path / "out")

            mock_review_cls.assert_called_once_with(interactive=False)


# ---------------------------------------------------------------------------
# TestGateBDayArtifactsShapes — regression for the DayArtifacts dataclass
# with master_path=None after the master purge and the finals-only
# reconstruction on resume.
# ---------------------------------------------------------------------------


class TestGateBDayArtifactsShapes:
    """``gate_b`` iterates ``state["day_artifacts"]``. Each entry can be:

    1. ``DayArtifacts`` dataclass with ``master_path`` populated (fresh run).
    2. ``DayArtifacts`` dataclass with ``master_path=None`` and real LF
       paths (post-purge resume — see ``_reconstruct_day_artifacts_from_disk``).
    3. Plain dict (legacy state shape).

    The prior code did ``getattr(art, "master_path", None) or art.get(...)``
    which crashed in case (2): getattr returns None (falsy) so the ``or``
    branch fires, and dataclasses don't have ``.get`` →
    ``AttributeError: 'DayArtifacts' object has no attribute 'get'``.
    """

    def _valid_lf_file(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (2 * 1024 * 1024))
        return path

    def test_dataclass_with_none_master_uses_lf_paths(self, tmp_path):
        """Case 2: master_path=None (purged) but regular_lf_paths populated.
        Must fall through to LF paths without AttributeError."""
        from gomoveshift.pipeline.day_pipeline import DayArtifacts

        lf_path = self._valid_lf_file(tmp_path / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4")
        art = DayArtifacts(
            date="2026-02-20",
            master_path=None,  # purged post-bookkeeping
            regular_lf_paths=[lf_path],
            short_paths=[],
        )
        state = make_state(day_artifacts=[art])

        gate = ReviewGate(interactive=False)
        with patch("gomoveshift.pipeline.review_gate.subprocess.run") as mock_run:
            mock_run.side_effect = _good_ffprobe_side_effect
            # Must not raise AttributeError; the LF path is discovered
            # and validated.
            gate.gate_b(state=state, output_dir=tmp_path)

    def test_dataclass_with_master_path_populated(self, tmp_path):
        """Case 1: fresh run with master on disk."""
        from gomoveshift.pipeline.day_pipeline import DayArtifacts

        master = self._valid_lf_file(tmp_path / "day_2026-02-20" / "day_2026-02-20_master.mp4")
        art = DayArtifacts(
            date="2026-02-20",
            master_path=master,
            regular_lf_paths=[],
            short_paths=[],
        )
        state = make_state(day_artifacts=[art])

        gate = ReviewGate(interactive=False)
        with patch("gomoveshift.pipeline.review_gate.subprocess.run") as mock_run:
            mock_run.side_effect = _good_ffprobe_side_effect
            gate.gate_b(state=state, output_dir=tmp_path)

    def test_dict_art_still_works(self, tmp_path):
        """Case 3: legacy dict form — backward compat must be preserved."""
        master = self._valid_lf_file(tmp_path / "day_2026-02-20" / "day_2026-02-20_master.mp4")
        art = {
            "date": "2026-02-20",
            "master_path": str(master),
            "regular_lf_paths": [],
            "short_paths": [],
        }
        state = make_state(day_artifacts=[art])

        gate = ReviewGate(interactive=False)
        with patch("gomoveshift.pipeline.review_gate.subprocess.run") as mock_run:
            mock_run.side_effect = _good_ffprobe_side_effect
            gate.gate_b(state=state, output_dir=tmp_path)


class TestGatePreRenderDocstringHonesty:
    """gate_pre_render historically claimed it 'cannot be bypassed by
    --no-review'. That was false — only the automated OST checks are
    unconditional; the human confirmation IS skipped under --no-review. The
    docstring/claim must match reality."""

    def test_docstring_no_longer_claims_unbypassable(self):
        doc = ReviewGate.gate_pre_render.__doc__ or ""
        assert "cannot be bypassed" not in doc.lower()

    def test_no_review_bypasses_human_prompt(self, tmp_path, monkeypatch):
        """interactive=False (--no-review) skips the human confirmation — proving
        the gate IS bypassable, so the old claim was false. Automated checks
        still run (clean OST here → no raise)."""
        import gomoveshift.pipeline.longform_metadata as lm

        monkeypatch.setattr(lm, "load_published_titles", lambda: set())
        gate = ReviewGate(interactive=False)
        with patch("builtins.input") as mock_input:
            gate.gate_pre_render(lf_index=1, ost_text="Newfound Gap Road, NC", output_dir=tmp_path)
            mock_input.assert_not_called()


class TestGatePreviewPromptDropsAudio:
    """After the audio rework the shipped audio is deterministically near-silent
    (hard-gated), so the preview's 'audio OK?' question is moot and removed."""

    def test_prompt_omits_audio_question(self, tmp_path, monkeypatch):
        gate = ReviewGate(interactive=True)
        preview_dir = tmp_path / "_stages" / "preview"
        preview_dir.mkdir(parents=True)
        (preview_dir / "ep_001_preview.mp4").touch()
        captured = {}

        def _fake_prompt(msg):
            captured["msg"] = msg

        monkeypatch.setattr(gate, "_prompt", _fake_prompt)
        with patch("subprocess.run", side_effect=_good_ffprobe_side_effect):
            gate.gate_preview(output_dir=tmp_path)
        assert "audio" not in captured["msg"].lower()


class TestPreRenderTitleUniquenessGate:
    """gate_pre_render hard-fails when the FINAL title exactly duplicates an
    already-published channel title, and only warns on a substring echo."""

    def _patch_published(self, monkeypatch, titles):
        import gomoveshift.pipeline.longform_metadata as lm

        monkeypatch.setattr(lm, "load_published_titles", lambda: set(titles))

    def test_exact_channel_dupe_hard_fails(self, tmp_path, monkeypatch):
        # The ledger holds ASSEMBLED final titles ("… in 4K"); the
        # gate compares the OST label's assembled form against them.
        self._patch_published(monkeypatch, {"Blue Ridge Parkway, NC in 4K"})
        gate = ReviewGate(interactive=False)
        with pytest.raises(ReviewFailedError, match="exactly duplicates"):
            gate.gate_pre_render(
                lf_index=1,
                ost_text="Blue Ridge Parkway, NC",
                output_dir=tmp_path,
            )

    def test_exact_match_is_case_insensitive(self, tmp_path, monkeypatch):
        self._patch_published(monkeypatch, {"foothills parkway, tn in 4k"})
        gate = ReviewGate(interactive=False)
        with pytest.raises(ReviewFailedError, match="exactly duplicates"):
            gate.gate_pre_render(
                lf_index=1,
                ost_text="Foothills Parkway, TN",
                output_dir=tmp_path,
            )

    def test_substring_only_warns_not_fails(self, tmp_path, monkeypatch, capsys):
        # Published title merely *contains* the OST text → advisory WARN, no fail.
        self._patch_published(monkeypatch, {"Blue Ridge Parkway, NC to Asheville Scenic Loop"})
        gate = ReviewGate(interactive=False)
        gate.gate_pre_render(
            lf_index=1,
            ost_text="Blue Ridge Parkway, NC",
            output_dir=tmp_path,
        )
        assert "[WARN]" in capsys.readouterr().out

    def test_gate_checks_manifest_proposed_title_not_ost(self, tmp_path, monkeypatch):
        # The manifest's proposed_title is the FINAL title; the gate must key
        # off it even when the OST location text differs and is collision-free.
        self._patch_published(monkeypatch, {"Foothills Parkway, TN"})
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"proposed_title": "Foothills Parkway, TN"}))
        gate = ReviewGate(interactive=False)
        with pytest.raises(ReviewFailedError, match="exactly duplicates"):
            gate.gate_pre_render(
                lf_index=1,
                ost_text="Cades Cove Loop, TN",
                manifest_path=manifest,
                output_dir=tmp_path,
            )

    def test_assembled_form_dupe_hard_fails(self, tmp_path, monkeypatch):
        # Nothing ever writes the manifest's proposed_title, so the
        # hard check silently compared the raw OST label. The label ships as
        # its ASSEMBLED final title ("A to B via US-221, NC" → "A to B, NC in
        # 4K") — the gate must compare that assembled form to the channel.
        self._patch_published(monkeypatch, {"Asheville to Boone, NC in 4K"})
        gate = ReviewGate(interactive=False)
        with pytest.raises(ReviewFailedError, match="exactly duplicates"):
            gate.gate_pre_render(
                lf_index=1,
                ost_text="Asheville to Boone via US-221, NC",
                output_dir=tmp_path,
            )

    def test_no_published_titles_passes_clean(self, tmp_path, monkeypatch):
        self._patch_published(monkeypatch, set())
        gate = ReviewGate(interactive=False)
        # Must not raise.
        gate.gate_pre_render(
            lf_index=1,
            ost_text="Newfound Gap Road, NC",
            output_dir=tmp_path,
        )


class TestPreRenderLedgerReadability:
    """gate_pre_render must distinguish a MISSING published-titles ledger
    (fresh channel — skip) from an UNREADABLE one (dedup unverifiable — hard
    fail, never a silent ship on unverified dedup)."""

    def _make_unreadable_ledger(self):
        # Directory where the ledger file is expected: exists() True, open()
        # raises IsADirectoryError (OSError subclass). Portable across root.
        import gomoveshift.pipeline.longform_metadata as lm

        events = lm.DEFAULT_SEO_STORAGE / "gomoveshift" / "events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        events.mkdir()

    def test_unreadable_ledger_hard_fails_and_warns(self, tmp_path, caplog):
        import logging

        self._make_unreadable_ledger()
        gate = ReviewGate(interactive=False)
        with caplog.at_level(logging.WARNING):
            with pytest.raises(ReviewFailedError, match="unreadable"):
                gate.gate_pre_render(
                    lf_index=1,
                    ost_text="Newfound Gap Road, NC",
                    output_dir=tmp_path,
                )
        assert any("unreadable" in r.getMessage().lower() for r in caplog.records)

    def test_missing_ledger_skips_no_hard_fail(self, tmp_path):
        # No events.jsonl in the isolated storage -> MISSING -> skip, not fail.
        gate = ReviewGate(interactive=False)
        gate.gate_pre_render(
            lf_index=1,
            ost_text="Newfound Gap Road, NC",
            output_dir=tmp_path,
        )
