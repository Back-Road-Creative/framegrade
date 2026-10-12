"""Tests for encoding constants unification.

OutputConfig encoding fields (crf, preset, profile, pix_fmt) must be frozen
to the canonical values in encoding_constants.py. YAML overrides for these
quality-critical fields must be silently rejected.
"""

import textwrap

from gomoveshift.config import Config, OutputConfig
from gomoveshift.pipeline.encoding_constants import (
    LONGFORM_CRF,
    LONGFORM_PIX_FMT,
    LONGFORM_PRESET,
    LONGFORM_PROFILE,
)


class TestOutputConfigDefaults:
    """OutputConfig defaults must match encoding_constants.py canonical values."""

    def test_default_crf_matches_longform_crf(self):
        cfg = OutputConfig()
        assert cfg.crf == LONGFORM_CRF, (
            f"OutputConfig.crf default ({cfg.crf}) != LONGFORM_CRF ({LONGFORM_CRF})"
        )

    def test_default_preset_matches_longform_preset(self):
        cfg = OutputConfig()
        assert cfg.preset == LONGFORM_PRESET

    def test_default_profile_matches_longform_profile(self):
        cfg = OutputConfig()
        assert cfg.profile == LONGFORM_PROFILE

    def test_default_pix_fmt_matches_longform_pix_fmt(self):
        cfg = OutputConfig()
        assert cfg.pix_fmt == LONGFORM_PIX_FMT


class TestYAMLOverrideFrozen:
    """YAML config must NOT override frozen encoding fields."""

    def test_yaml_crf_override_ignored(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            output:
              crf: 23
        """)
        )
        cfg = Config.load(yaml_file)
        assert cfg.output.crf == LONGFORM_CRF, (
            f"YAML override should be ignored: got {cfg.output.crf}, expected {LONGFORM_CRF}"
        )

    def test_yaml_preset_override_ignored(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            output:
              preset: "ultrafast"
        """)
        )
        cfg = Config.load(yaml_file)
        assert cfg.output.preset == LONGFORM_PRESET

    def test_yaml_profile_override_ignored(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            output:
              profile: "main"
        """)
        )
        cfg = Config.load(yaml_file)
        assert cfg.output.profile == LONGFORM_PROFILE

    def test_yaml_pix_fmt_override_ignored(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            output:
              pix_fmt: "yuv420p"
        """)
        )
        cfg = Config.load(yaml_file)
        assert cfg.output.pix_fmt == LONGFORM_PIX_FMT

    def test_yaml_non_frozen_fields_still_work(self, tmp_path):
        """audio_codec, embed_chapters should still be overridable."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            output:
              audio_codec: "opus"
              embed_chapters: false
        """)
        )
        cfg = Config.load(yaml_file)
        assert cfg.output.audio_codec == "opus"
        assert cfg.output.embed_chapters is False

    def test_yaml_mixed_frozen_and_overridable(self, tmp_path):
        """Frozen fields locked, non-frozen fields applied."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
            output:
              crf: 30
              audio_codec: "flac"
              preset: "ultrafast"
        """)
        )
        cfg = Config.load(yaml_file)
        # Frozen
        assert cfg.output.crf == LONGFORM_CRF
        assert cfg.output.preset == LONGFORM_PRESET
        # Overridable
        assert cfg.output.audio_codec == "flac"
