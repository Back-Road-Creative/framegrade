"""Regression tests for the ``upload`` CLI handler's pipeline_state.json mutation.

Guards the upload handler's state-file mutation — upload.py loads
``_stages/pipeline_state.json`` as a raw dict, mutates ``stages_completed``,
and writes it back. If any other writer (main-pipeline ``state_manager``
or drive orchestrator ``_drive_state``) introduces additional fields,
those fields must survive the upload handler's read-mutate-write cycle
intact.

Background
----------
The upload handler operates on projects produced by the main pipeline
(``pipeline/orchestrator.py``), which persists state via
``state_manager.save_state`` as a flat dict with keys like
``source_hash``, ``current_video``, ``video_properties``, ``timings``,
``params`` — none of which are declared on the drive-pipeline
``PipelineState`` dataclass.

Swapping ``json.load`` / ``json.dump`` for
``PipelineState.from_dict`` / ``state.to_dict`` is tempting. A naive swap would
silently drop every main-pipeline field because ``to_dict`` only emits
declared dataclass fields. These tests pin the key-preservation
invariant so future refactors catch that regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


UPLOADER_MOD = "gomoveshift.pipeline.uploader"


def _build_main_pipeline_state(
    source_video: str = "/projects/foo/source.mp4",
    source_hash: str = "deadbeef",
    stages: list[str] | None = None,
) -> dict:
    """Construct a main-pipeline-shape state dict as state_manager writes it."""
    return {
        "project_name": "foo",
        "source_video": source_video,
        "source_hash": source_hash,
        "stages_completed": list(
            stages if stages is not None else ["preflight", "shorts_generation"]
        ),
        "current_video": "/projects/foo/_stages/step_09.mp4",
        "video_properties": {
            "width": 3840,
            "height": 2160,
            "fps": 29.97,
            "duration": 4321.5,
            "bitrate_kbps": 45000,
            "codec": "h264",
            "audio_codec": "aac",
            "file_size_bytes": 27_500_000_000,
        },
        "timings": {"preflight": 1.2, "shorts_generation": 340.7},
        "params": {"publish_day": "wed", "output_dir": "/projects/foo"},
        "keep_segments": [[0.0, 120.5], [240.0, 4200.0]],
    }


@pytest.fixture
def project_dir_with_state(tmp_path: Path):
    """Project layout with a realistic main-pipeline state file and shorts."""
    project = tmp_path / "foo"
    project.mkdir()

    shorts = project / "shorts"
    shorts.mkdir()
    (shorts / "short_01.mp4").write_text("fake-short")
    # The upload gate requires the run's own manifest with a matching count.
    (shorts / "social_posts.json").write_text(json.dumps([{"short_index": 1, "caption": "x"}]))

    stages = project / "_stages"
    stages.mkdir()
    state = _build_main_pipeline_state()
    state_file = stages / "pipeline_state.json"
    state_file.write_text(json.dumps(state, indent=2))
    return project, state_file, state


def _run_cmd_upload_with_mocked_drive(
    project_dir: Path,
    *,
    skip_csv: bool = True,
) -> int:
    """Invoke cmd_upload with Drive I/O stubbed out."""
    mock_manifest = MagicMock()
    mock_manifest.files = [MagicMock()]

    mock_uploader_instance = MagicMock()
    mock_uploader_instance.upload_shorts.return_value = mock_manifest
    mock_uploader_cls = MagicMock(return_value=mock_uploader_instance)

    mock_args = MagicMock()
    mock_args.project_dir = project_dir
    mock_args.credentials = None
    mock_args.token = None
    mock_args.init_oauth = False
    mock_args.skip_csv = skip_csv

    import gomoveshift.pipeline.uploader as _uploader_mod  # noqa: F401
    from gomoveshift.pipeline.pipeline_validator import PipelineValidator

    with (
        patch.object(_uploader_mod, "PipelineUploader", mock_uploader_cls),
        # The upload gate runs the shorts quality validator; these fixtures
        # use a fake text short, so stub the geometry/integrity gate to PASS —
        # the point of THIS suite is state-key preservation, not shorts QA.
        patch.object(
            PipelineValidator,
            "validate_shorts",
            lambda self, d: MagicMock(passed=True, summary=lambda: "ok"),
        ),
        patch.object(
            _uploader_mod,
            "DEFAULT_CREDENTIALS_ENC",
            MagicMock(exists=MagicMock(return_value=True)),
        ),
        patch.object(
            _uploader_mod,
            "DEFAULT_TOKEN_ENC",
            MagicMock(exists=MagicMock(return_value=True)),
        ),
    ):
        from gomoveshift.cli.handlers.upload import cmd_upload

        return cmd_upload(mock_args, {})


class TestAppendStageCompletedHelper:
    """Unit tests for ``state_manager.append_stage_completed``."""

    def test_preserves_unknown_top_level_keys(self, tmp_path):
        from gomoveshift.pipeline.state_manager import append_stage_completed

        state_file = tmp_path / "pipeline_state.json"
        original = _build_main_pipeline_state(stages=[])
        original["youtube_uploads_sha"] = "abc123"
        original["some_future_field"] = {"nested": [1, 2]}
        state_file.write_text(json.dumps(original, indent=2))

        assert append_stage_completed(state_file, "drive_upload") is True

        post = json.loads(state_file.read_text())
        assert post["stages_completed"] == ["drive_upload"]
        for key, old_val in original.items():
            if key == "stages_completed":
                continue
            assert post[key] == old_val

    def test_idempotent(self, tmp_path):
        from gomoveshift.pipeline.state_manager import append_stage_completed

        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text(json.dumps({"stages_completed": ["drive_upload"]}, indent=2))

        assert append_stage_completed(state_file, "drive_upload") is True
        post = json.loads(state_file.read_text())
        assert post["stages_completed"].count("drive_upload") == 1

    def test_missing_file_returns_false(self, tmp_path):
        from gomoveshift.pipeline.state_manager import append_stage_completed

        assert append_stage_completed(tmp_path / "does_not_exist.json", "x") is False

    def test_corrupt_file_logs_and_returns_false(self, tmp_path, caplog):
        import logging as _logging

        from gomoveshift.pipeline.state_manager import append_stage_completed

        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text("NOT VALID JSON")

        with caplog.at_level(_logging.WARNING):
            result = append_stage_completed(state_file, "drive_upload")
        assert result is False
        assert any("corrupt" in r.message.lower() for r in caplog.records)
        # Corrupt file is left untouched, not overwritten with {}.
        assert state_file.read_text() == "NOT VALID JSON"

    def test_creates_stages_completed_when_missing(self, tmp_path):
        from gomoveshift.pipeline.state_manager import append_stage_completed

        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text(json.dumps({"source_hash": "abc"}, indent=2))

        assert append_stage_completed(state_file, "drive_upload") is True
        post = json.loads(state_file.read_text())
        assert post["stages_completed"] == ["drive_upload"]
        assert post["source_hash"] == "abc"


class TestUploadHandlerPreservesStateKeys:
    """Upload handler must not drop unknown or main-pipeline fields on save."""

    def test_all_main_pipeline_fields_survive_drive_upload_mutation(self, project_dir_with_state):
        """After the handler marks drive_upload complete, every pre-existing
        top-level key must still be present with its original value."""
        project, state_file, original = project_dir_with_state

        rc = _run_cmd_upload_with_mocked_drive(project, skip_csv=True)
        assert rc == 0

        post = json.loads(state_file.read_text())

        # Every pre-existing key must be present.
        missing = set(original) - set(post)
        assert not missing, (
            f"upload handler dropped keys: {missing}. "
            "Root cause is usually a state-shape refactor that reads via "
            "PipelineState.from_dict (drive-pipeline dataclass) and writes "
            "via state.to_dict, which only serializes declared fields."
        )

        # Every unmutated key must retain its exact value. ``stages_completed``
        # is expected to grow by one entry; everything else is invariant.
        for key, old_val in original.items():
            if key == "stages_completed":
                continue
            assert post[key] == old_val, (
                f"upload handler mutated unrelated key {key!r}: {old_val!r} -> {post[key]!r}"
            )

        # Mutation is visible and idempotent-safe.
        assert "drive_upload" in post["stages_completed"]

    def test_hypothetical_future_field_survives_round_trip(self, project_dir_with_state):
        """If a future orchestrator writes a new top-level field (e.g. the
        ``youtube_uploads_sha`` added to drive_orchestrator), the
        upload handler must preserve it.

        This pins the invariant that motivates the raw-dict approach: the
        handler's read-mutate-write must not drop keys it doesn't know
        about.
        """
        project, state_file, original = project_dir_with_state

        raw = json.loads(state_file.read_text())
        raw["youtube_uploads_sha"] = "abc123def456"  # pragma: allowlist secret
        raw["future_orchestrator_field"] = {"nested": ["value", 1, True]}
        state_file.write_text(json.dumps(raw, indent=2))

        rc = _run_cmd_upload_with_mocked_drive(project, skip_csv=True)
        assert rc == 0

        post = json.loads(state_file.read_text())
        assert post.get("youtube_uploads_sha") == "abc123def456", (  # pragma: allowlist secret
            "youtube_uploads_sha dropped by upload handler on save."
        )
        assert post.get("future_orchestrator_field") == {"nested": ["value", 1, True]}, (
            "Unknown nested field dropped by upload handler on save."
        )

    def test_drive_upload_marker_is_idempotent(self, project_dir_with_state):
        """Re-running the handler must not append drive_upload twice."""
        project, state_file, _ = project_dir_with_state

        rc1 = _run_cmd_upload_with_mocked_drive(project, skip_csv=True)
        rc2 = _run_cmd_upload_with_mocked_drive(project, skip_csv=True)
        assert rc1 == 0 and rc2 == 0

        post = json.loads(state_file.read_text())
        assert post["stages_completed"].count("drive_upload") == 1

    def test_naive_pipelinestate_round_trip_would_drop_main_pipeline_keys(self, tmp_path):
        """Documents why replacing
        raw JSON with PipelineState.from_dict / state.to_dict is unsafe for
        this call site. The main-pipeline state file is a flat dict with
        keys that don't exist on the drive-pipeline ``PipelineState``
        dataclass, so a naive swap silently drops them.

        This test is the canonical counterexample: if someone attempts the
        naive refactor anyway, this test stays green (demonstrating the
        data-loss path) while the preservation tests above flip red.
        """
        from gomoveshift.pipeline.drive_state_model import PipelineState

        raw = _build_main_pipeline_state()

        # Simulate the naive swap literally.
        state = PipelineState.from_dict(raw)
        round_tripped = state.to_dict()

        # Every main-pipeline-only key is lost through the dataclass filter.
        # (youtube_uploads_sha is *not* in this list — it became a declared
        # PipelineState field.)
        for dropped in (
            "project_name",
            "source_video",
            "source_hash",
            "stages_completed",
            "current_video",
            "video_properties",
            "timings",
            "params",
            "keep_segments",
        ):
            assert dropped not in round_tripped, (
                f"PipelineState.from_dict/to_dict unexpectedly preserved "
                f"{dropped!r}; the counterexample no longer applies. "
                "Revisit state_manager.append_stage_completed to confirm it "
                "still protects against key loss."
            )

    def test_csv_regeneration_path_also_preserves_fields(self, project_dir_with_state):
        """The second state-mutation site (csv_generation append) must
        preserve unknown fields as strictly as the first."""
        project, state_file, original = project_dir_with_state

        # Provide the pieces required for CSV regeneration to run.
        raw = json.loads(state_file.read_text())
        raw["youtube_uploads_sha"] = "csv-path-sha"
        state_file.write_text(json.dumps(raw, indent=2))

        shorts = project / "shorts"
        # One post per on-disk short so the count gate matches.
        (shorts / "social_posts.json").write_text(json.dumps([{"short_index": 1, "caption": "x"}]))

        # Mock CSVGenerator so we don't exercise real CSV logic.
        with patch("gomoveshift.pipeline.csv_generator.CSVGenerator") as mock_csv_cls:
            mock_csv_cls.return_value.generate.return_value = project / "out.csv"
            rc = _run_cmd_upload_with_mocked_drive(project, skip_csv=False)
        assert rc == 0

        post = json.loads(state_file.read_text())
        assert post.get("youtube_uploads_sha") == "csv-path-sha", (
            "csv_generation mutation path dropped unknown field."
        )
        assert post.get("source_hash") == original["source_hash"]
        assert post.get("video_properties") == original["video_properties"]
