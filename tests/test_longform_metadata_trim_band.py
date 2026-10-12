"""TDD tests for trim_ratio threading + Highlights band hook + always-on disclosure.

* Title verb (`Highlights from` vs `Driving`) is selected by `_select_verb(trim_ratio)`.
* Description hook is banded by trim_ratio.
* Disclosure block (`🎬 Final cut`, `📅 Filmed`, `📍 Location`) is unconditional.
"""

from __future__ import annotations

import re

import pytest

from gomoveshift.pipeline.longform_metadata import (
    HIGHLIGHTS_TRIM_THRESHOLD,
    LongFormMetadataGenerator,
)

DISCLOSURE_RE = re.compile(r"🎬 Final cut: .+? \(edited from .+? of footage\)")


class _StubGenerator(LongFormMetadataGenerator):
    """LongFormMetadataGenerator with SEO Manager disabled — exercises
    ``_generate_basic`` (the path tests can drive deterministically).
    """

    def _init_seo_manager(self):  # pragma: no cover - simple override
        self._manager = None
        self._optimizer = None
        self._seo_degraded = True

    def _init_ledger(self):  # pragma: no cover - skip ledger IO in tests
        self._ledger = None
        self._ledger_path = None


@pytest.fixture
def gen(tmp_path):
    return _StubGenerator(seo_storage_path=tmp_path)


LOCATION = "Dingle Peninsula, County Kerry, Ireland"


class TestTitleBand:
    # The title is verb-less + front-loaded regardless of trim ratio; the
    # band now drives only the description (see TestDescriptionHook below).
    def test_low_trim_title_is_verb_less(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=0.3,
            recording_date="2025-09-14",
            source_duration=6000.0,
        )
        assert "in 4K" in result["title"]
        assert "Scenic Drive" not in result["title"]

    def test_high_trim_title_is_verb_less(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=0.7,
            recording_date="2025-09-14",
            source_duration=2570.0,
        )
        assert not result["title"].startswith("Driving ")

    def test_none_trim_title_is_verb_less(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date="2025-09-14",
        )
        assert not result["title"].startswith("Driving ")


class TestDescriptionHook:
    def test_low_trim_ratio_uses_highlights_hook(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=0.3,
            recording_date="2025-09-14",
            source_duration=6000.0,
        )
        first_line = result["description"].split("\n", 1)[0]
        assert first_line.startswith("Highlights from ")
        # The source-duration phrasing — "of driving through {location}".
        assert "driving through" in first_line.lower()

    def test_high_trim_ratio_uses_pov_hook(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=0.7,
            recording_date="2025-09-14",
            source_duration=2570.0,
        )
        first_line = result["description"].split("\n", 1)[0]
        assert first_line.startswith("POV drive through ")

    def test_none_trim_ratio_uses_pov_hook(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date="2025-09-14",
        )
        first_line = result["description"].split("\n", 1)[0]
        assert first_line.startswith("POV drive through ")


class TestDisclosureLine:
    """Every generation contains the disclosure block, regardless of band."""

    @pytest.mark.parametrize("ratio", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_disclosure_unconditional(self, gen, ratio):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=ratio,
            recording_date="2025-09-14",
            source_duration=3600.0,
        )
        desc = result["description"]
        assert DISCLOSURE_RE.search(desc), (
            f"disclosure line missing for trim_ratio={ratio}; got desc:\n{desc}"
        )
        # 📅 Filmed line preserved (not duplicated).
        assert desc.count("📅 Filmed:") == 1
        # 📍 Location line present.
        assert "📍 Location:" in desc

    def test_disclosure_includes_both_durations(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,  # 30m final
            project_name="t",
            trim_ratio=0.3,
            recording_date="2025-09-14",
            source_duration=6000.0,  # 1h 40m source
        )
        desc = result["description"]
        m = DISCLOSURE_RE.search(desc)
        assert m is not None
        # Both source + final pretty-formatted appear in the disclosure.
        assert "30m" in desc  # final
        # Source rendering uses minutes/hours; "1h 40m" is the human form.
        assert "1h 40m" in desc

    def test_disclosure_present_when_trim_ratio_none(self, gen):
        # The disclosure depends only on durations, not on the band — so it
        # still appears for legacy callers that don't pass trim_ratio.
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            recording_date="2025-09-14",
            source_duration=2400.0,
        )
        assert DISCLOSURE_RE.search(result["description"]) is not None


class TestHonestDisclosureFullDriveMode:
    """In full_drive mode, when the formatted source and final durations are
    identical, there was no effective edit. Emitting "🎬 Final cut: X (edited from X of footage)"
    is (1) a false claim and (2) produces a 3× redundancy on the duration
    string in the description, which the redundancy gate hard-fails.

    Contract: when source_pretty == final_pretty, use the legacy honest
    "🎬 Duration: X" form. The "Final cut ... edited from" disclosure
    only appears when an edit actually happened."""

    def test_full_drive_emits_legacy_duration_form(self, gen):
        """source_duration ≈ video_duration → no "edited from" lie."""
        result = gen.generate(
            location=LOCATION,
            video_duration=1799.4,
            project_name="t",
            trim_ratio=1.0,
            recording_date="2025-09-14",
            source_duration=1799.4,
        )
        desc = result["description"]
        assert "🎬 Duration:" in desc, (
            f"full_drive (source≈final) must use the legacy honest "
            f"'🎬 Duration:' form; got:\n{desc}"
        )
        assert "edited from" not in desc, (
            f"no edit happened in full_drive — 'edited from' must not appear; got:\n{desc}"
        )

    def test_full_drive_description_passes_redundancy_gate(self, gen):
        """Regression guard for the full-drive redundancy failure:
        in full_drive mode the same duration string previously appeared 3×
        (POV hook + Final cut + edited-from). With the honest disclosure
        the count drops to 2× (hook + Duration) — within the gate's
        allowed range."""
        result = gen.generate(
            location=LOCATION,
            video_duration=1799.4,
            project_name="t",
            trim_ratio=1.0,
            recording_date="2025-09-14",
            source_duration=1799.4,
        )
        desc = result["description"]
        # The "29m 59s" (or "30m 0s") duration string must appear at most
        # twice in the description — once in the POV hook, once in the
        # Duration line. Three times would re-trigger the gate.
        formatted = gen._format_duration(1799.4)
        assert desc.count(formatted) <= 2, (
            f"duration string {formatted!r} appears "
            f"{desc.count(formatted)}× in description (max 2×); "
            f"the 3× case triggers the description redundancy gate. Got:\n{desc}"
        )

    def test_real_edit_keeps_final_cut_disclosure(self, gen):
        """When source > final (a real edit), the "Final cut ... edited from"
        disclosure still appears — the honesty contract only suppresses
        the false claim, not the truthful one."""
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=0.3,
            recording_date="2025-09-14",
            source_duration=6000.0,
        )
        desc = result["description"]
        assert DISCLOSURE_RE.search(desc) is not None, (
            f"real edits must still surface the 'edited from' disclosure; got:\n{desc}"
        )

    def test_full_drive_passes_phase_d_redundancy_check(self, gen):
        """Integration: drive the same content_gates check the orchestrator
        uses, so a future change to either the generator or the gate keeps
        the contract aligned."""
        from gomoveshift.pipeline.validation.content_gates import (
            check_description_redundancy,
        )

        result = gen.generate(
            location=LOCATION,
            video_duration=1799.4,
            project_name="t",
            trim_ratio=1.0,
            recording_date="2025-09-14",
            source_duration=1799.4,
        )
        verdict = check_description_redundancy(result)
        assert verdict.passed, (
            f"redundancy gate must pass on full_drive descriptions; got "
            f"violations: {verdict.message!r}"
        )


class TestThresholdBoundary:
    """Ratio == threshold no longer affects the title (verb-less)."""

    def test_at_threshold_title_is_verb_less(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=HIGHLIGHTS_TRIM_THRESHOLD,
            recording_date="2025-09-14",
            source_duration=3600.0,
        )
        assert not result["title"].startswith("Driving ")


class TestTitleOverrideVerb:
    """An explicit title_override (the canonical per_lf_ost_text, verb-less) is
    routed through _build_title and receives the route-only SEO shape —
    verb-less, trim-independent, idempotent over a leading verb, no genre word."""

    OVERRIDE = "Tralee to Dingle via N86, Kerry"
    EXPECTED = "Tralee to Dingle, Kerry in 4K"

    def test_override_low_trim_front_loaded(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            title_override=self.OVERRIDE,
            trim_ratio=0.3,
            recording_date="2025-09-14",
            source_duration=6000.0,
        )
        assert result["title"] == self.EXPECTED

    def test_override_high_trim_front_loaded(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            title_override=self.OVERRIDE,
            trim_ratio=0.7,
            recording_date="2025-09-14",
            source_duration=2570.0,
        )
        assert result["title"] == self.EXPECTED

    def test_override_at_threshold_front_loaded(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            title_override=self.OVERRIDE,
            trim_ratio=HIGHLIGHTS_TRIM_THRESHOLD,
            recording_date="2025-09-14",
            source_duration=3600.0,
        )
        assert result["title"] == self.EXPECTED

    def test_override_none_trim_front_loaded(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            title_override=self.OVERRIDE,
            recording_date="2025-09-14",
        )
        assert result["title"] == self.EXPECTED

    def test_override_already_verbed_is_idempotent(self, gen):
        # A leading verb is stripped first, so the shape never double-prefixes.
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            title_override=f"Driving {self.OVERRIDE}",
            trim_ratio=0.7,
            recording_date="2025-09-14",
            source_duration=2570.0,
        )
        assert result["title"] == self.EXPECTED
        assert "Driving" not in result["title"]

    def test_no_override_path_front_loaded(self, gen):
        result = gen.generate(
            location=LOCATION,
            video_duration=1800.0,
            project_name="t",
            trim_ratio=0.3,
            recording_date="2025-09-14",
            source_duration=6000.0,
        )
        assert "in 4K" in result["title"]
        assert "Scenic Drive" not in result["title"]
