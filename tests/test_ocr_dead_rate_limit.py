"""The ``max_claude_calls_per_hour`` knob limited nothing, so it is gone.

``OCREngine.__init__`` read the value into ``self.max_claude_calls`` and no code
path ever compared anything against it — the Claude OCR fallback is disabled in
``process_frame`` (source is hardcoded to ``easyocr``), and the
``claude_calls`` counter that would have fed such a check was removed. An operator tuning
the knob changed nothing, which is worse than no knob at all.

Retiring an operator-facing config key is only safe because the pydantic models
here ignore unknown keys, so an existing ``config.yaml`` that still carries it
keeps loading. That is pinned below.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from gomoveshift.analysis.ocr import OCREngine
from gomoveshift.config import Config, OCRConfig

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


class TestKnobIsGone:
    def test_engine_has_no_claude_call_budget(self):
        engine = OCREngine(Config())

        assert not hasattr(engine, "max_claude_calls")

    def test_ocr_config_no_longer_declares_the_key(self):
        assert "max_claude_calls_per_hour" not in OCRConfig.model_fields

    def test_sample_config_no_longer_ships_the_key(self):
        data = yaml.safe_load(REPO_CONFIG.read_text()) or {}

        assert "max_claude_calls_per_hour" not in (data.get("ocr") or {})

    def test_no_shipped_config_carries_the_key(self):
        """No yaml the repo ships (a sample ``config.yaml`` at the root, the
        deployment overrides under ``docker/``, the release template) names the
        retired key under ``ocr``."""
        root = REPO_CONFIG.parent
        shipped = [*root.glob("*.yaml"), *(root / "docker").glob("*.yaml")]
        for path in shipped:
            data = yaml.safe_load(path.read_text()) or {}
            ocr = data.get("ocr") if isinstance(data, dict) else None
            assert "max_claude_calls_per_hour" not in (ocr or {}), path


class TestRetiredKeyStaysLoadable:
    """An operator's existing config.yaml must not start failing to load."""

    def test_config_with_retired_key_still_loads(self, tmp_path: Path):
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump({"ocr": {"max_claude_calls_per_hour": 300, "languages": ["en"]}})
        )

        config = Config.load(path)

        assert config.ocr.languages == ["en"]


class TestLiveOCRKnobsSurvive:
    def test_languages_still_reaches_the_easyocr_reader(self):
        assert "languages" in OCRConfig.model_fields
        assert Config().ocr.languages == ["en"]
