"""Shorts text never opens like text already spooled for another project.

The anti-spam gate's ``cross-batch-repeat`` rule refuses a short whose text
opens with the same 6 words as one spooled in the last 60 days. Two drives
at the same place used to collide by construction: the YouTube description
was the fixed "POV driving through {location} Pt.N" (the part number is word
7, past the key), and the caption openers rotate per project, so a second
project at the same place reuses some. The generator now walks its pools
past every opener the spool already holds, and rotates the description lead.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.publishing.anti_spam_gate import text_opener_key

CAPTION_PLATFORMS = ["facebook", "instagram", "tiktok"]


def _gen(project, spooled_texts=None, **kw):
    return SocialGenerator(
        location="County Cork, Ireland",
        project_name=project,
        series_part=1,
        spooled_texts={} if spooled_texts is None else spooled_texts,
        **kw,
    )


def _texts(gen, n=7):
    """Platform -> the gate-comparable texts of ``n`` shorts."""
    out = {p: [] for p in [*CAPTION_PLATFORMS, "youtube"]}
    for i in range(1, n + 1):
        captions = gen._build_platform_captions(None, i)
        for p in CAPTION_PLATFORMS:
            out[p].append(captions[p])
        out["youtube"].append(gen._build_youtube_description(i))
    return out


def test_youtube_description_openers_unique_within_a_batch():
    descs = _texts(_gen("proj_alpha"), n=12)["youtube"]
    assert len({text_opener_key(d) for d in descs}) == 12, descs


def test_youtube_description_differs_between_projects_at_one_place():
    a = _texts(_gen("proj_alpha"))["youtube"]
    b = _texts(_gen("proj_bravo"))["youtube"]
    assert {text_opener_key(d) for d in a} != {text_opener_key(d) for d in b}


def test_interactive_description_openers_also_rotate():
    gen = _gen("proj_alpha")
    descs = [gen._build_youtube_description(i, interactive=True) for i in range(1, 8)]
    assert len({text_opener_key(d) for d in descs}) == 7, descs


@pytest.mark.parametrize("platform", [*CAPTION_PLATFORMS, "youtube"])
def test_second_project_at_one_place_avoids_the_first_ones_spooled_openers(platform):
    spooled = _texts(_gen("proj_alpha"))
    second = _texts(_gen("proj_bravo", spooled_texts=spooled))[platform]
    taken = {text_opener_key(t) for t in spooled[platform]}
    clashes = [t for t in second if text_opener_key(t) in taken]
    assert clashes == []
    assert len({text_opener_key(t) for t in second}) == len(second), second


def test_exhausted_pool_falls_back_to_the_rotation_instead_of_raising():
    spooled = _texts(_gen("proj_alpha"), n=14)
    for p in CAPTION_PLATFORMS:
        spooled[p] += _texts(_gen("proj_charlie"), n=14)[p]
    gen = _gen("proj_bravo", spooled_texts=spooled)
    assert len(gen._build_platform_captions(None, 1)["facebook"]) > 0


def test_default_reads_the_spool_without_this_project(monkeypatch, tmp_path):
    from gomoveshift.pipeline import longform_metadata
    from gomoveshift.publishing import anti_spam_gate
    from gomoveshift.publishing.queue import open_queue_db

    open_queue_db(tmp_path).close()
    monkeypatch.setattr(longform_metadata, "DEFAULT_SEO_STORAGE", tmp_path)
    taken = "Would you drive through County Cork? Filmed September 2025."
    calls = []

    def fake_recent(conn, **kw):
        calls.append(kw)
        return {"facebook": [taken]}

    monkeypatch.setattr(anti_spam_gate, "recent_platform_texts", fake_recent)
    gen = SocialGenerator(location="County Cork, Ireland", project_name="proj_bravo")
    caps = [
        gen._build_platform_captions(None, i)["facebook"] for i in range(1, 14)
    ]  # 14 openers, 1 taken
    assert calls == [{"exclude_projects": {"proj_bravo"}}]
    assert all(text_opener_key(c) != text_opener_key(taken) for c in caps), caps


def test_default_with_no_queue_db_creates_nothing(monkeypatch, tmp_path):
    from gomoveshift.pipeline import longform_metadata

    monkeypatch.setattr(longform_metadata, "DEFAULT_SEO_STORAGE", tmp_path / "seo")
    gen = SocialGenerator(location="County Cork, Ireland", project_name="proj_bravo")
    assert gen._build_platform_captions(None, 1)["facebook"]
    assert not (tmp_path / "seo").exists()


@pytest.mark.parametrize("project", [f"proj_{i}" for i in range(40)])
def test_every_description_lead_passes_the_plan_grammar_gate(project):
    # A lead opening with a bare preposition ("Through the windshield ...") was
    # refused by PUBLISH-PLAN-GRAMMAR, failing the whole plan (Kenmare, 11-07).
    from gomoveshift.pipeline.validation.content_gates import find_grammar_issues

    gen = _gen(project)
    descriptions = [gen._build_youtube_description(i) for i in range(1, 8)]
    assert all(not find_grammar_issues(d) for d in descriptions), descriptions


def test_no_description_lead_opens_with_a_bare_preposition():
    from gomoveshift.pipeline._series_utils import is_title_prefixed
    from gomoveshift.pipeline.social_generator import YOUTUBE_DESCRIPTION_LEADS

    assert [lead for lead in YOUTUBE_DESCRIPTION_LEADS if is_title_prefixed(lead)] == []
