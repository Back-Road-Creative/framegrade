"""Tests for per-caption generation provenance.

Covers: GenerationProvenance invariants/serialization, wiring into
CaptionGenerator.generate_batch, and backward-compatible YAML loading.
"""

import json

import pytest

from gomoveshift.pipeline.photos import caption_generator as cg_module
from gomoveshift.pipeline.photos.caption_generator import (
    CaptionBatch,
    CaptionGenerator,
    PhotoCaption,
)
from gomoveshift.pipeline.photos.generation_provenance import GenerationProvenance
from gomoveshift.pipeline.photos.selector import PhotoSelection
from gomoveshift.pipeline.photos.voice_styles import VoiceStyle


def _mock_subprocess_result(stdout_text, returncode=0):
    """Fake Popen handle matching the CLI subprocess output (mirrors the helper in
    test_photo_caption_generator.py; duplicated to avoid a relative import
    across test modules)."""
    from unittest.mock import MagicMock

    proc = MagicMock()
    proc.returncode = returncode
    proc.stderr = ""
    proc.communicate.side_effect = lambda *a, **k: (stdout_text, proc.stderr)
    return proc


def _make_selection():
    return PhotoSelection(
        journey_slug="example-journey",
        journey_title="Example Journey",
        day_number=3,
        day_date="2025-01-15",
        location="Fablemara",
        filename="photo-042.jpg",
        photo_path="/fake/path/photo-042.jpg",
        locations_visited=["Fablebridge", "Fablemara", "Mockstone"],
        day_summary="Exploring the rugged coast of Fablemara",
        statistics={"distance_km": 120, "duration_hours": 5.2},
    )


class TestGenerationProvenanceDataclass:
    def test_not_recorded_sentinel(self):
        prov = GenerationProvenance.not_recorded()
        assert prov.recorded is False
        assert prov.model is None
        assert prov.fallback_used is False

    def test_from_dict_none_is_not_recorded(self):
        prov = GenerationProvenance.from_dict(None)
        assert prov.recorded is False

    def test_from_dict_empty_is_not_recorded(self):
        prov = GenerationProvenance.from_dict({})
        assert prov.recorded is False

    def test_recorded_round_trips(self):
        prov = GenerationProvenance(
            recorded=True,
            model="sonnet",
            recipe_version="abc123",
            evidence=("journey_title", "location"),
            fallback_used=False,
            fallback_reason=None,
            attempts=1,
            duration_s=1.5,
            validation_passed=("main_caption",),
            validation_failed=(),
        )
        data = prov.to_dict()
        restored = GenerationProvenance.from_dict(data)
        assert restored == prov

    def test_fallback_used_requires_model(self):
        with pytest.raises(ValueError):
            GenerationProvenance(
                recorded=True,
                model=None,
                recipe_version="abc123",
                fallback_used=True,
                fallback_reason="validation failed",
                attempts=1,
            )

    def test_fallback_used_requires_reason(self):
        with pytest.raises(ValueError):
            GenerationProvenance(
                recorded=True,
                model="sonnet",
                recipe_version="abc123",
                fallback_used=True,
                fallback_reason=None,
                attempts=1,
            )

    def test_recorded_requires_at_least_one_attempt(self):
        with pytest.raises(ValueError):
            GenerationProvenance(
                recorded=True,
                model="sonnet",
                recipe_version="abc123",
                attempts=0,
            )


class TestRecipeVersionDerivation:
    def test_recipe_version_is_a_hash_of_the_prompt_template(self):
        assert cg_module.RECIPE_VERSION
        assert len(cg_module.RECIPE_VERSION) >= 8

    def test_recipe_version_changes_when_prompt_template_changes(self, monkeypatch):
        original = cg_module._CRITICAL_RULES
        monkeypatch.setattr(cg_module, "_CRITICAL_RULES", original + "\nExtra new rule.\n")
        recomputed = cg_module._compute_recipe_version()
        assert recomputed != cg_module.RECIPE_VERSION


class TestGenerateBatchProvenance:
    @pytest.fixture(autouse=True)
    def _patch_popen(self, monkeypatch):
        self._monkeypatch = monkeypatch

    def test_successful_generation_records_no_fallback(self, tmp_path, monkeypatch):
        sel = _make_selection()
        clean = [
            {
                "main_caption": "The rugged coast of Fablemara, Fablebridge to Mockstone.",
                "tiktok_caption": "Fablemara coast road",
                "hashtags": ["ireland"],
                "first_comment_hook": "The coast road never ends.",
            }
        ]
        import subprocess

        monkeypatch.setattr(
            subprocess, "Popen", lambda *a, **k: _mock_subprocess_result(json.dumps(clean))
        )

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch([sel], VoiceStyle.FIRST_PERSON, week_start="2026-03-18")

        caption = batch.captions[0]
        prov = GenerationProvenance.from_dict(caption.provenance)
        assert prov.recorded is True
        assert prov.fallback_used is False
        assert prov.model == gen.model
        assert prov.attempts == 1
        assert prov.evidence  # non-empty — metadata keys actually fed to the prompt
        assert "journey_title" in prov.evidence

    def test_fallback_generation_records_reason(self, tmp_path, monkeypatch):
        sel = _make_selection()
        stats_heavy = [
            {
                "main_caption": "198 km looping through Inishowen. Ten hours on the road.",
                "tiktok_caption": "198 km in ten hours",
                "hashtags": ["ireland"],
                "first_comment_hook": "106 photos across the peninsula.",
            }
        ]
        bad_retry = {
            "main_caption": "120 km and five hours through Fablemara coast.",
            "tiktok_caption": "120 km in five hours",
            "first_comment_hook": "54 shots of Atlantic roads.",
        }
        responses = [
            _mock_subprocess_result(json.dumps(stats_heavy)),
            _mock_subprocess_result(json.dumps(bad_retry)),
            _mock_subprocess_result(json.dumps(bad_retry)),
        ]

        import subprocess

        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: responses.pop(0))

        gen = CaptionGenerator(caption_dir=tmp_path)
        batch = gen.generate_batch([sel], VoiceStyle.FIRST_PERSON, week_start="2026-03-18")

        caption = batch.captions[0]
        prov = GenerationProvenance.from_dict(caption.provenance)
        assert prov.recorded is True
        assert prov.fallback_used is True
        assert prov.fallback_reason
        assert prov.model == gen.model
        assert prov.attempts == 3

        # And the saved YAML actually says so.
        loaded = CaptionBatch.load(tmp_path / "2026-03-18_captions.yaml")
        loaded_prov = GenerationProvenance.from_dict(loaded.captions[0].provenance)
        assert loaded_prov.fallback_used is True


class TestBackwardCompatibleLoad:
    def test_old_yaml_with_no_provenance_reads_as_not_recorded(self, tmp_path):
        batch = CaptionBatch(
            week_start="2026-02-23",
            voice_style="voice_first_person",
            captions=[
                PhotoCaption(
                    photo_index=0,
                    main_caption="Test caption",
                    tiktok_caption="Test",
                    hashtags=["ireland"],
                    journey_slug="example-journey",
                    filename="photo-042.jpg",
                    location="Fablemara",
                ),
            ],
            generated_at="2026-02-16T10:00:00",
            model="sonnet",
        )
        path = tmp_path / "old_captions.yaml"
        batch.save(path)

        # Simulate a pre-provenance file by stripping the field from the YAML.
        import yaml

        raw = yaml.safe_load(path.read_text())
        raw["captions"][0].pop("provenance", None)
        path.write_text(yaml.safe_dump(raw, default_flow_style=False, sort_keys=False))

        loaded = CaptionBatch.load(path)
        prov = GenerationProvenance.from_dict(loaded.captions[0].provenance)
        assert prov.recorded is False
        # Not-recorded is NOT the same claim as "no fallback used".
        assert prov.recorded is False and prov.fallback_used is False


class TestNotRecordedCarriesNoFindings:
    """recorded=False is "we do not know", and it may not smuggle a claim."""

    def test_a_not_recorded_provenance_may_not_carry_a_model(self):
        with pytest.raises(ValueError, match="cannot carry findings"):
            GenerationProvenance(recorded=False, model="sonnet", recipe_version=None)

    def test_a_not_recorded_provenance_may_not_claim_a_fallback(self):
        with pytest.raises(ValueError, match="cannot carry findings"):
            GenerationProvenance(
                recorded=False, model=None, recipe_version=None, fallback_used=True
            )

    def test_a_not_recorded_provenance_may_not_carry_attempts(self):
        with pytest.raises(ValueError, match="cannot carry findings"):
            GenerationProvenance(recorded=False, model=None, recipe_version=None, attempts=3)

    def test_the_sentinel_itself_still_builds(self):
        prov = GenerationProvenance.not_recorded()
        assert prov.recorded is False
        assert prov.model is None
        assert prov.fallback_used is False
