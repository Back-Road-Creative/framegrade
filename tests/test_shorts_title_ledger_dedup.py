"""A short's title must not collide with a title already published on
the channel. Reuses ``longform_metadata.read_published_titles`` (the same
``events.jsonl`` ledger the long-form dedup gate reads — it records every
channel video's title, shorts included) rather than duplicating a reader.

A collision is resolved with a DETERMINISTIC alternative composed from the
same grounded fact slots (``TITLE_FRAME_TEMPLATES``) — never a random
pick — so the same inputs always produce the same (different) title across
runs/resumes.

``DEFAULT_SEO_STORAGE`` is redirected to a fresh per-test tempdir by the
autouse ``isolate_seo_storage_from_workspace`` conftest fixture.
"""

import json

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline.social_generator import (
    TITLE_FRAME_TEMPLATES,
    SocialGenerator,
    _frame_fields,
)

_LOCATION = "County Kerry, Ireland"
_PROJECT = "dedup-test"


def _write_ledger(titles):
    events = lm.DEFAULT_SEO_STORAGE / "gomoveshift" / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"data": {"title": t}} for t in titles]
    events.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _gen():
    return SocialGenerator(location=_LOCATION, project_name=_PROJECT, series_part=1)


def _default_pick():
    """The title _gen() would produce for short_index=1, absent any ledger
    collision — computed the same way the generator does, so this test
    doesn't hardcode a frame-rotation-dependent literal string."""
    return _gen().generate_post(short_index=1).youtube_title


class TestShortsTitleDedupAgainstLedger:
    def test_colliding_title_is_replaced(self):
        colliding = _default_pick()
        _write_ledger([colliding])
        post = _gen().generate_post(short_index=1)
        assert post.youtube_title != colliding

    def test_replacement_is_deterministic_not_random(self):
        _write_ledger([_default_pick()])
        first = _gen().generate_post(short_index=1).youtube_title
        second = _gen().generate_post(short_index=1).youtube_title
        assert first == second

    def test_replacement_still_comes_from_the_frame_set(self):
        colliding = _default_pick()
        _write_ledger([colliding])
        gen = _gen()
        title = gen.generate_post(short_index=1).youtube_title
        fmap = {
            "location": gen.location_short,
            "terrain_desc": gen.terrain_desc.title(),
        }
        applicable = [
            t for t in TITLE_FRAME_TEMPLATES if _frame_fields(t) <= {"location", "terrain_desc"}
        ]
        rendered_pool = {t.format_map(_SafeMap(fmap)) for t in applicable}
        assert title in rendered_pool


class _SafeMap(dict):
    def __missing__(self, key):
        return ""
