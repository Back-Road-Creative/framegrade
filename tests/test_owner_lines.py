"""Owner lines: one sentence in the owner's voice on Shorts and long-form text."""

import pytest
import yaml

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.owner_lines import (
    OwnerLines,
    load_owner_lines,
    owner_lines_path,
    with_owner_paragraph,
    with_owner_sentence,
)
from gomoveshift.pipeline.social_generator import SocialGenerator


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _file(tmp_path, data):
    path = tmp_path / "owner_lines.yaml"
    path.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data)
    return path


def test_missing_file_means_no_lines(tmp_path):
    assert load_owner_lines(tmp_path / "owner_lines.yaml") == OwnerLines()
    assert load_owner_lines(None) == OwnerLines()


def test_lines_load_by_index_with_whitespace_folded(tmp_path):
    lines = load_owner_lines(
        _file(tmp_path, {"shorts": {3: "Best chips  in the\ncounty"}, "longform": {"1": "Hi"}})
    )
    assert lines.shorts == {3: "Best chips in the county"}
    assert lines.longform == {1: "Hi"}


@pytest.mark.parametrize(
    "data, needle",
    [
        ({"shorts": {"x": "Hi"}}, "not an int"),
        ({"shorts": {1: ""}}, "non-empty"),
        ({"shorts": {1: "a" * 101}}, "over 100"),
        ({"shorts": {1: "See https://example.com"}}, "link or hashtag"),
        ({"longform": {1: "Loved it #roadtrip"}}, "link or hashtag"),
        ({"reels": {1: "Hi"}}, "unknown section"),
        ("- just a list\n", "must be a mapping"),
    ],
)
def test_malformed_file_raises_naming_the_problem(tmp_path, data, needle):
    with pytest.raises(ValueError, match=needle):
        load_owner_lines(_file(tmp_path, data))


def test_path_sits_beside_titles_yaml(tmp_path):
    assert owner_lines_path(tmp_path / "social" / "titles.yaml") == (
        tmp_path / "social" / "owner_lines.yaml"
    )


def test_sentence_and_paragraph_helpers():
    assert with_owner_sentence("Driving through Kerry.", "Loved this road") == (
        "Loved this road. Driving through Kerry."
    )
    assert with_owner_sentence("Same.", None) == "Same."
    assert with_owner_paragraph("Desc line.", "My favourite stretch!") == (
        "My favourite stretch!\n\nDesc line."
    )
    assert with_owner_paragraph("Desc line.", "") == "Desc line."


def _generator(tmp_path):
    return SocialGenerator(
        location="County Kerry, Ireland",
        project_name="test_project",
        series_part=1,
        titles_file=tmp_path / "titles.yaml",
    )


def test_short_line_opens_facebook_and_instagram_only(tmp_path):
    _file(tmp_path, {"shorts": {1: "Stopped here for chips"}})
    with_line = _generator(tmp_path).generate_post(1)
    (tmp_path / "owner_lines.yaml").unlink()
    without = _generator(tmp_path).generate_post(1)

    for platform in ("facebook", "instagram"):
        assert with_line.captions[platform].startswith("Stopped here for chips. ")
        rest = with_line.captions[platform].removeprefix("Stopped here for chips. ")
        assert rest == without.captions[platform]
        assert "County Kerry" in with_line.captions[platform]
    assert with_line.captions["youtube"] == without.captions["youtube"]
    assert with_line.captions["tiktok"] == without.captions["tiktok"]


def test_other_shorts_are_untouched(tmp_path):
    _file(tmp_path, {"shorts": {2: "Only short two"}})
    post = _generator(tmp_path).generate_post(1)
    assert "Only short two" not in " ".join(post.captions.values())
