"""Legacy single-video ``SocialGenerationStage`` (source of the earlier
15-shorts spam) must fail loud on ``TitleCapacityExceeded``, never pad.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from gomoveshift.config import Config
from gomoveshift.pipeline.stages.social import SocialGenerationStage


def _ctx(tmp_path: Path, num_shorts: int) -> SimpleNamespace:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"\x00")
    return SimpleNamespace(
        config=Config(),
        paths=SimpleNamespace(project_dir=tmp_path),
        shorts_data=None,
        current_video=video,
        recording_date=None,
        metadata=None,
        project_name="episode_001",
        num_shorts=num_shorts,
        location="Test Drive, County Test",
        social_posts=None,
    )


class TestSocialStageTitleCapacity:
    def test_over_capacity_batch_fails_loud(self, tmp_path):
        # 4 shorts, no titles.yaml, no road/county (cap 3) → fails loud.
        result = SocialGenerationStage().run(_ctx(tmp_path, num_shorts=4))
        assert result.success is False
        assert "TitleCapacityExceeded" in result.error and "titles.yaml" in result.error

    def test_titles_yaml_covering_the_batch_succeeds(self, tmp_path):
        # A titles.yaml covering every index bypasses composition.
        social_dir = tmp_path / "social"
        social_dir.mkdir(parents=True, exist_ok=True)
        titles = yaml.safe_dump({i: {"title": f"Short {i}"} for i in range(1, 5)})
        (social_dir / "titles.yaml").write_text(titles)
        result = SocialGenerationStage().run(_ctx(tmp_path, num_shorts=4))
        assert result.success is True, result.error
