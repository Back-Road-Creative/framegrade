"""Tests for Phase D bare-except cleanup.

``_build_social_posts_per_lf`` narrows the legacy bare
``except Exception: pass`` so AttributeError (MagicMock test convenience)
stays silent while real IO/serialization failures log at WARNING.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from gomoveshift.pipeline.drive_phases.phase_d import _build_social_posts_per_lf


def _make_lf(index: int):
    return SimpleNamespace(index=index)


def _make_social_gen_cls(post: MagicMock):
    """Return a class-like callable whose instance.generate_all yields `[post]`."""
    instance = MagicMock()
    instance.generate_all.return_value = [post]
    return MagicMock(return_value=instance)


class TestPhaseDSocialPostsWriteWarning:
    """Real serialization/IO failures must surface at WARNING."""

    def test_attribute_error_is_silent(self, tmp_path, caplog):
        """MagicMock without to_dict raises AttributeError — stays silent."""
        post = MagicMock(spec=[])  # no to_dict attribute
        drive_plan = SimpleNamespace(regular_lfs=[_make_lf(0)])

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_phases.phase_d"):
            _build_social_posts_per_lf(
                social_gen_cls=_make_social_gen_cls(post),
                drive_plan=drive_plan,
                shorts_per_lf={0: ["/x/s.mp4"]},
                per_lf_locations_override={0: "Test"},
                project_base="proj",
                output_base=tmp_path,
            )

        assert not any("social_posts.json" in r.message for r in caplog.records), (
            "AttributeError should be silent; bare-except was the legacy escape hatch"
        )

    def test_type_error_logs_warning(self, tmp_path, caplog):
        """TypeError (json.dump on a non-serializable object) logs WARNING."""
        post = MagicMock()
        post.to_dict.side_effect = TypeError("not serializable")
        drive_plan = SimpleNamespace(regular_lfs=[_make_lf(7)])

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_phases.phase_d"):
            _build_social_posts_per_lf(
                social_gen_cls=_make_social_gen_cls(post),
                drive_plan=drive_plan,
                shorts_per_lf={7: ["/x/s.mp4"]},
                per_lf_locations_override={7: "Test"},
                project_base="proj",
                output_base=tmp_path,
            )

        assert any(
            "failed to write social_posts.json" in r.message and "LF 7" in r.message
            for r in caplog.records
        ), f"Expected WARN for TypeError, got: {[r.message for r in caplog.records]}"
