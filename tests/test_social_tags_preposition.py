"""A bare-preposition operator label must never surface as a YouTube tag.

``_build_tags`` used to split the raw label on ',' without stripping a leading
operator title word, so 'To Goleen, County Cork' shipped the tag "To Goleen".
``_region_tags`` had the same gap on its one-component string-parse fallback.
"""

from __future__ import annotations

from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.pipeline.validation.content_gates import _GRAMMAR_LEADING_PREP_RE


def _tags(location: str, **kwargs) -> list[str]:
    gen = SocialGenerator(location=location, project_name="tags-prep", **kwargs)
    post = gen.generate_post(short_index=1)
    return [t.strip() for t in post.youtube_tags.split(",") if t.strip()]


class TestBuildTagsStripsLeadingPreposition:
    def test_county_label_never_emits_a_prepositioned_tag(self) -> None:
        tags = _tags("To Goleen, County Cork")
        assert not any(_GRAMMAR_LEADING_PREP_RE.match(t) for t in tags)
        assert any("Goleen" in t for t in tags)

    def test_single_component_label_never_emits_a_prepositioned_tag(self) -> None:
        tags = _tags("To Goleen", resolved_location=None)
        assert not any(_GRAMMAR_LEADING_PREP_RE.match(t) for t in tags)
        assert any("Goleen" in t for t in tags)

    def test_plain_label_unaffected(self) -> None:
        tags = _tags("Goleen, County Cork")
        assert not any(_GRAMMAR_LEADING_PREP_RE.match(t) for t in tags)
        assert any("Goleen" in t for t in tags)
