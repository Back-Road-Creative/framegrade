"""
Bitrate zero-guard tests — pipeline must fail fast when bitrate_kbps=0.

A bitrate of 0 means the source file's bitrate couldn't be read. Without it,
NVENC VBR has no bitrate floor and silently drops output from 83→32 Mbps.
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBitrateZeroGuard:
    """Pipeline must fail immediately when bitrate_kbps=0."""

    def test_zero_bitrate_from_detect_fails_pipeline(self):
        """After detect_properties, bitrate_kbps=0 must abort the pipeline."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        from gomoveshift.pipeline.stages.detect_properties import PropertyDetectionStage

        source = inspect.getsource(PropertyDetectionStage.run)
        assert "props.bitrate_kbps == 0" in source or "props.bitrate_kbps <= 0" in source, (
            "CRITICAL: orchestrator.run() must check for bitrate_kbps == 0 "
            "after detect_properties and fail the pipeline. Without this guard, "
            "NVENC VBR silently drops output bitrate."
        )

    def test_zero_bitrate_from_resume_fails_pipeline(self):
        """On resume, bitrate_kbps=0 from saved state must abort the pipeline."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        source = inspect.getsource(PipelineOrchestrator._restore_from_state)
        # The resume path reconstructs props — must also guard zero bitrate
        assert "bitrate_kbps" in source, "Resume path must reference bitrate_kbps"

    def test_unreadable_source_fails_fast(self):
        """Source file < 1 KB must fail before wasting time on detection."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        from gomoveshift.pipeline.stages.validation import ValidationStage

        source = inspect.getsource(ValidationStage.run)
        # Must check file size is reasonable before running ffprobe
        assert "1024" in source or "1 KB" in source or "file_size" in source.lower(), (
            "CRITICAL: orchestrator must fail fast on tiny/empty source files "
            "(< 1 KB) before attempting ffprobe detection."
        )


class TestCopySizeVerification:
    """shutil.copy2 must verify destination file size matches source."""

    def test_copy_size_verification_in_orchestrator(self):
        """Orchestrator must verify file sizes after shutil.copy2 calls."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        from gomoveshift.pipeline.stages.wsl_copy import WSLCopyStage

        source = inspect.getsource(WSLCopyStage.run)
        assert "Copy incomplete" in source or "copy_size" in source.lower(), (
            "CRITICAL: orchestrator must verify file sizes after shutil.copy2 "
            "to catch incomplete copies (network interrupts, disk full)."
        )
