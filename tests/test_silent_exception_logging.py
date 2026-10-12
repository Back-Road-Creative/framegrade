"""Regressions for previously-removed dead code paths."""


class TestClaudeOCRRemoved:
    """_run_claude_ocr should no longer exist on OCREngine."""

    def test_method_removed(self):
        from gomoveshift.analysis.ocr import OCREngine

        assert not hasattr(OCREngine, "_run_claude_ocr"), (
            "_run_claude_ocr should have been removed — it was permanently disabled dead code"
        )
