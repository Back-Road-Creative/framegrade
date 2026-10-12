"""Tests for voice_map dedup and drive_orchestrator re-export cleanup.

Verifies:
1. VOICE_MAP constant in photos.py works correctly
2. Imports from source modules resolve after re-export removal
"""

import argparse

from gomoveshift.pipeline.photos.voice_styles import VoiceStyle


def _voice_choice_sets(parser: argparse.ArgumentParser) -> list[set[str]]:
    """Collect the ``--voice`` ``choices`` of every subparser that declares it."""
    found: list[set[str]] = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                found.extend(_voice_choice_sets(sub))
        elif "--voice" in action.option_strings and action.choices:
            found.append(set(action.choices))
    return found


class TestVoiceMapConstant:
    """VOICE_MAP is a single module-level constant, not duplicated."""

    def test_voice_map_matches_cli_voice_choices_exactly(self):
        """VOICE_MAP keys == the CLI's --voice choices, so the two cannot drift.

        Set equality (not a count) is the exhaustiveness guard: a new voice
        style needs no edit here, but a style added to only one side fails.
        """
        from gomoveshift.cli.handlers.photos import _get_voice_map
        from gomoveshift.cli.parser import _build_parser

        choice_sets = _voice_choice_sets(_build_parser())
        assert choice_sets, "no --voice argument with choices found in the CLI parser"
        for choices in choice_sets:
            assert choices == choice_sets[0], (
                f"--voice choices disagree between subcommands: {choices} != {choice_sets[0]}"
            )
        assert set(_get_voice_map()) == choice_sets[0]

    def test_voice_map_values_are_voice_style_enums(self):
        from gomoveshift.cli.handlers.photos import _get_voice_map

        voice_map = _get_voice_map()
        for key, val in voice_map.items():
            assert isinstance(val, VoiceStyle), f"{key} maps to {type(val)}, not VoiceStyle"

    def test_voice_map_first_person(self):
        from gomoveshift.cli.handlers.photos import _get_voice_map

        assert _get_voice_map()["first_person"] is VoiceStyle.FIRST_PERSON

    def test_voice_map_minimal_zen(self):
        from gomoveshift.cli.handlers.photos import _get_voice_map

        assert _get_voice_map()["minimal_zen"] is VoiceStyle.MINIMAL_ZEN

    def test_voice_map_engaging(self):
        from gomoveshift.cli.handlers.photos import _get_voice_map

        assert _get_voice_map()["engaging"] is VoiceStyle.ENGAGING_CONVERSATIONAL


class TestDriveOrchestratorReexportCleanup:
    """After cleanup, symbols import from their source modules."""

    def test_differentiate_duplicate_ost_from_drive_helpers(self):
        """differentiate_duplicate_ost is importable from _drive_helpers."""
        from gomoveshift.pipeline._drive_helpers import differentiate_duplicate_ost

        assert callable(differentiate_duplicate_ost)

    def test_bearing_to_cardinal_from_drive_helpers(self):
        """_bearing_to_cardinal is importable from _drive_helpers."""
        from gomoveshift.pipeline._drive_helpers import _bearing_to_cardinal

        assert callable(_bearing_to_cardinal)

    def test_drive_orchestrator_class_still_importable(self):
        """DrivePipelineOrchestrator is still importable (it's defined there)."""
        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        assert DrivePipelineOrchestrator is not None

    def test_drive_orchestrator_only_resolve_reexports(self):
        """drive_orchestrator.py re-exports only symbols needed by resolve() + test patches."""
        from pathlib import Path

        orch_path = Path(__file__).resolve().parent.parent / (
            "gomoveshift/pipeline/drive_orchestrator.py"
        )
        content = orch_path.read_text()
        noqa_lines = [line.strip() for line in content.splitlines() if "# noqa: F401" in line]
        # Only the 6 lines needed by resolve() and test patches should remain
        allowed_symbols = {
            "CSVGenerator",
            "LongFormMetadataGenerator",
            "PipelineScheduler",
            "PipelineUploader",
            "PipelineValidator",
            "ReleaseSchedule",
            "SocialGenerator",
            "UploadManifest",
        }
        for line in noqa_lines:
            # Extract imported names from the line
            found = False
            for sym in allowed_symbols:
                if sym in line:
                    found = True
                    break
            assert found, f"Unexpected re-export: {line}"
        # Verify the old bulk re-exports are gone (color, ingest, etc.)
        assert not any("color" in line for line in noqa_lines)
        assert not any("ingest" in line for line in noqa_lines)
        assert not any("DriveAssembler" in line for line in noqa_lines)
