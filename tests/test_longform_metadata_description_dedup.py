"""TDD tests for description deduplication.

Each fact must appear in exactly one place:
- recording_date: only in the 📅 Filmed line (not in the hook).
- location: hook + 📍 Location line (NOT in the body when GPS-less).
- The body line ``"A {duration} drive through {location}."`` adds no new
  facts when chapter_towns / named_routes are empty — it must be omitted.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator


class _StubGenerator(LongFormMetadataGenerator):
    """LongFormMetadataGenerator with SEO Manager disabled — exercises
    ``_generate_basic`` deterministically.
    """

    def _init_seo_manager(self):  # pragma: no cover
        self._manager = None
        self._optimizer = None
        self._seo_degraded = True

    def _init_ledger(self):  # pragma: no cover
        self._ledger = None
        self._ledger_path = None


@pytest.fixture
def gen(tmp_path):
    return _StubGenerator(seo_storage_path=tmp_path)


LOCATION = "Glenfable Peninsula toward Gortmore, County Kerry"


def test_hook_omits_recording_date_when_filmed_line_present(gen):
    """The hook (first line) must NOT carry the recording date — the 📅 Filmed
    line is the single source of truth for that fact.
    """
    result = gen.generate(
        location=LOCATION,
        video_duration=588.0,  # 9m 48s
        project_name="t",
        trim_ratio=0.3,
        recording_date="2025-09-14",
        source_duration=1800.0,  # 30m source
    )
    first_line = result["description"].split("\n", 1)[0]
    assert first_line == (f"Highlights from 30m 0s of driving through {LOCATION}."), (
        f"Hook leaked date or other text; got: {first_line!r}"
    )
    assert " on 2025-09-14" not in first_line


def test_filmed_line_present_unchanged(gen):
    """The 📅 Filmed: <date> line is preserved in the body."""
    result = gen.generate(
        location=LOCATION,
        video_duration=588.0,
        project_name="t",
        trim_ratio=0.3,
        recording_date="2025-09-14",
        source_duration=1800.0,
    )
    # Long-form routes through the shared "Mon YYYY" formatter.
    assert "📅 Filmed: Sep 2025" in result["description"]


def test_body_empty_when_no_chapter_data(gen):
    """With neither chapter_towns nor named_routes, the body line is
    omitted entirely. No ``A 9m 48s drive through ...`` sentence in the
    description body.
    """
    result = gen.generate(
        location=LOCATION,
        video_duration=588.0,  # 9m 48s
        project_name="t",
        trim_ratio=0.3,
        recording_date="2025-09-14",
        source_duration=1800.0,
    )
    desc = result["description"]
    assert "A 9m 48s drive through" not in desc
    # Also no "A <anything> drive through {location}" restatement.
    assert f"drive through {LOCATION}." not in desc.split("\n", 1)[1]


def test_body_present_when_chapter_towns(gen):
    """When chapter_towns is provided, the body line still appears with
    the ``passing through`` phrase that adds new facts not in the
    structured block.
    """
    desc = gen._build_gms_description(
        location=LOCATION,
        seo_description=f"POV drive through {LOCATION}.",
        duration_str="9m 48s",
        publish_day="Monday",
        recording_date="2025-09-14",
        chapter_towns=["Glenfable", "Fable Head"],
        trim_ratio=0.3,
        source_duration=1800.0,
        video_duration=588.0,
    )
    assert "passing through Glenfable and Fable Head" in desc


def test_location_appears_only_in_hook_and_location_line(gen):
    """The exact location string occurs exactly TWICE in the description:
    once in the hook, once in the 📍 Location line. Was THREE times before
    the dedup (hook + body restatement + 📍).
    """
    result = gen.generate(
        location=LOCATION,
        video_duration=588.0,
        project_name="t",
        trim_ratio=0.3,
        recording_date="2025-09-14",
        source_duration=1800.0,
    )
    desc = result["description"]
    assert desc.count(LOCATION) == 2, (
        f"location appeared {desc.count(LOCATION)}× — expected 2 (hook + "
        f"📍 Location). Description:\n{desc}"
    )
