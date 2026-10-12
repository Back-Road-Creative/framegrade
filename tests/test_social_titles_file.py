"""Tests for hand-written titles.yaml overriding the template pool.

TDD: written BEFORE the implementation. See
docs/pipeline/135-shorts-titles-and-interactive-family.md "Hand titles".
"""

from types import SimpleNamespace

import pytest
import yaml

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _build_social_posts_per_lf
from gomoveshift.pipeline.social_generator import (
    HandTitle,
    SocialGenerator,
    load_hand_titles,
)


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    """Default: nothing published, so ledger collisions never fire unasked."""
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _write_titles_file(tmp_path, mapping):
    path = tmp_path / "titles.yaml"
    path.write_text(yaml.safe_dump(mapping))
    return path


def _generator(**kwargs):
    return SocialGenerator(
        location="County Kerry, Ireland",
        project_name="test_project",
        series_part=1,
        **kwargs,
    )


class TestHandTitleCovered:
    def test_covered_index_uses_hand_title_never_pool(self, tmp_path, monkeypatch):
        titles_file = _write_titles_file(
            tmp_path,
            {1: {"title": "Guess this coastal road", "caption": "A quiet coastal drive."}},
        )
        gen = _generator(titles_file=titles_file)

        def _boom(*a, **k):
            raise AssertionError("_pick_title must not be called for a covered index")

        monkeypatch.setattr(gen, "_pick_title", _boom)

        post = gen.generate_post(1)
        assert post.youtube_title == "Guess this coastal road"
        # The hand caption becomes the YouTube caption verbatim; the
        # other platforms (incl. the legacy ``.caption`` alias, which is the
        # Instagram text) compose it as their opening sentence.
        assert post.captions["youtube"] == "A quiet coastal drive."
        assert post.caption.startswith("A quiet coastal drive.")

    def test_interactive_flag_round_trips_onto_post(self, tmp_path):
        titles_file = _write_titles_file(
            tmp_path,
            {1: {"title": "Guess this coastal road", "interactive": True}},
        )
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)
        assert post.interactive is True
        # to_dict() carries interactive so downstream sidecars see it.
        assert post.to_dict()["interactive"] is True

    def test_interactive_defaults_false(self, tmp_path):
        titles_file = _write_titles_file(tmp_path, {1: {"title": "A hand title"}})
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(1)
        assert post.interactive is False

    def test_generate_all_keys_on_global_post_offset_index(self, tmp_path, monkeypatch):
        # index_offset=10 means the SECOND short in this batch (local i=2)
        # gets global short_index=12 — the file is keyed on that global index.
        titles_file = _write_titles_file(tmp_path, {12: {"title": "Hand title for global 12"}})
        gen = _generator(titles_file=titles_file)
        posts = gen.generate_all(num_shorts=2, index_offset=10)
        assert posts[0].short_index == 11
        assert posts[1].short_index == 12
        assert posts[1].youtube_title == "Hand title for global 12"
        assert posts[0].youtube_title != "Hand title for global 12"


class TestPartialFileFallsThroughToPool:
    def test_uncovered_index_composes_without_a_hand_title(self, tmp_path):
        titles_file = _write_titles_file(tmp_path, {1: {"title": "Only short 1"}})
        gen = _generator(titles_file=titles_file)
        post = gen.generate_post(2)
        assert post.youtube_title != "Only short 1"
        assert post.youtube_title  # pool still produced something


class TestLedgerCollision:
    def test_hand_title_collision_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lm, "load_published_titles", lambda: {"Already Published Title"})
        titles_file = _write_titles_file(tmp_path, {1: {"title": "Already Published Title"}})
        gen = _generator(titles_file=titles_file)
        with pytest.raises(ValueError) as exc_info:
            gen.generate_post(1)
        msg = str(exc_info.value)
        assert "1" in msg
        assert "Already Published Title" in msg


class TestMalformedFile:
    def test_not_a_mapping_raises(self, tmp_path):
        path = tmp_path / "titles.yaml"
        path.write_text(yaml.safe_dump(["not", "a", "mapping"]))
        with pytest.raises(ValueError, match="mapping"):
            load_hand_titles(path)

    def test_non_int_key_raises(self, tmp_path):
        path = tmp_path / "titles.yaml"
        path.write_text(yaml.safe_dump({"first": {"title": "x"}}))
        with pytest.raises(ValueError, match="non-integer"):
            load_hand_titles(path)

    def test_missing_title_raises(self, tmp_path):
        path = tmp_path / "titles.yaml"
        path.write_text(yaml.safe_dump({1: {"caption": "no title here"}}))
        with pytest.raises(ValueError, match="title"):
            load_hand_titles(path)

    def test_non_str_title_raises(self, tmp_path):
        path = tmp_path / "titles.yaml"
        path.write_text(yaml.safe_dump({1: {"title": 42}}))
        with pytest.raises(ValueError, match="title"):
            load_hand_titles(path)

    def test_empty_title_raises(self, tmp_path):
        path = tmp_path / "titles.yaml"
        path.write_text(yaml.safe_dump({1: {"title": "   "}}))
        with pytest.raises(ValueError, match="title"):
            load_hand_titles(path)

    def test_absent_file_returns_empty_mapping(self, tmp_path):
        assert load_hand_titles(tmp_path / "nope.yaml") == {}


class TestDrivePublishPathHandTitleOverride:
    """``_build_social_posts_per_lf`` threads titles_file through to every
    per-LF SocialGenerator as an OPTIONAL override: a hand title still wins on
    a publish run, an uncovered short composes its own."""

    def _lf(self, idx=1):
        return SimpleNamespace(index=idx, total_duration=1800.0)

    def test_publish_run_hand_title_still_wins(self, tmp_path):
        from _state_helpers import verified_shorts_per_lf

        titles_dir = tmp_path / "social"
        titles_dir.mkdir()
        (titles_dir / "titles.yaml").write_text(
            yaml.safe_dump({1: {"title": "Guess this coastal road"}})
        )
        out = _build_social_posts_per_lf(
            social_gen_cls=SocialGenerator,
            drive_plan=SimpleNamespace(regular_lfs=[self._lf()]),
            shorts_per_lf=verified_shorts_per_lf(tmp_path / "stages", {1: 2}),
            per_lf_locations_override={1: "Cork, Ireland"},
            project_base="p",
            output_base=tmp_path,
            titles_file=titles_dir / "titles.yaml",
            require_verified_places=True,
        )
        assert out[1][0].youtube_title == "Guess this coastal road"
        assert "stretch:" in out[1][1].youtube_title  # uncovered short composes
