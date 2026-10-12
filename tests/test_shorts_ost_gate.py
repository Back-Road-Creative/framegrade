"""Shorts-OST gate on the drive path.

Before this gate the day-first drive short loop burned ``short_ost_text``
into all 14 shorts/LF raw — it never ran ``shorten_ost_for_shorts`` nor
``validate_ost(text, "shorts")``, so a ``"None, None"`` / coordinate /
HOA label could burn irreversibly into the 9:16 mobile frame. The OST audit
sidecar also enumerated LF text only, never the shorts.

``_gate_short_ost_text`` shortens-then-validates with a parent-LF fallback,
and ``format_shorts_ost_audit`` records every short's final burned text.
"""

import json

import pytest

from gomoveshift.pipeline.day_pipeline import _gate_short_ost_text
from gomoveshift.pipeline.ffmpeg_filters import build_ost_filter
from gomoveshift.pipeline.ost_audit_sidecar import (
    compute_ost_audit_hash,
    confirmation_is_stale,
    format_ost_audit,
    format_shorts_ost_audit,
    write_shorts_ost_audit,
)
from gomoveshift.pipeline.ost_validator import validate_ost


class TestGateShortOstText:
    def test_valid_candidate_is_shortened_and_passes(self):
        # "via …" + " Scenic Drive" are stripped for the mobile frame.
        out = _gate_short_ost_text(
            "Killarney to Dingle via N86 Scenic Drive", None, rank=1, date="2026-05-24"
        )
        assert out == "Killarney to Dingle"
        ok, _ = validate_ost(out, "shorts")
        assert ok is True

    def test_garbage_candidate_falls_back_to_parent(self):
        out = _gate_short_ost_text("None, None", "County Kerry, Ireland", rank=2, date="2026-05-24")
        assert out == "County Kerry, Ireland"

    def test_coordinate_candidate_rejected(self):
        out = _gate_short_ost_text(
            "37.123, -122.456", "Marin County, CA", rank=3, date="2026-05-24"
        )
        assert out == "Marin County, CA"

    def test_garbage_sources_fall_back_to_drive_location(self):
        # Both candidate and parent are garbage, but the drive-wide location is
        # a real place — the short still gets a title (operator decision: always
        # show a title; hard-fail only when even the drive location is unusable).
        out = _gate_short_ost_text(
            "None, None", "null", rank=4, date="2026-05-24", drive_location="County Clare"
        )
        assert out == "County Clare"

    def test_no_usable_source_raises(self):
        # All sources garbage / empty and no drive location → a short must carry
        # a title, so the day hard-fails rather than silently shipping CTAs-only.
        with pytest.raises(RuntimeError, match="SHORTS-OST-UNRESOLVED"):
            _gate_short_ost_text("None, None", "null", rank=4, date="2026-05-24")

    def test_empty_candidate_uses_parent(self):
        out = _gate_short_ost_text("", "Beaufort County, SC", rank=5, date="2026-05-24")
        assert out == "Beaufort County, SC"

    def test_overlong_label_truncated_under_shorts_budget(self):
        long = "Somewhereville to Anotherlongplacename in the Far Distant County"
        out = _gate_short_ost_text(long, None, rank=6, date="2026-05-24")
        assert len(out) <= 40
        ok, _ = validate_ost(out, "shorts")
        assert ok is True

    def test_frame_overflowing_real_label_is_hard_fit_not_dropped(self):
        # A real label that only overflows the 2-line mobile frame must be
        # hard-fit (non-empty, valid) — never dropped to an empty overlay.
        crowded = "Aaaaaa Bbbbbb Cccccc Dddddd Eeeeee Ffffff Gggggg Hhhhhh"
        out = _gate_short_ost_text(crowded, None, rank=7, date="2026-05-24")
        assert out.strip()  # never empty
        ok, _ = validate_ost(out, "shorts")
        assert ok is True


class TestBuildOstFilterRequiresTitle:
    def test_empty_location_raises(self):
        # Burn-site regression guard: a short can never silently render with no
        # title (the degenerate empty-text drawtext path).
        with pytest.raises(ValueError, match="empty location"):
            build_ost_filter("", "sub", 38.0)

    def test_whitespace_location_raises(self):
        with pytest.raises(ValueError, match="empty location"):
            build_ost_filter("   ", "sub", 38.0)

    def test_real_location_renders(self):
        out = build_ost_filter("County Clare", "sub", 38.0)
        assert "County Clare" in out or "drawtext" in out


class TestShortsOstAudit:
    def test_format_enumerates_every_short(self):
        entries = [
            {
                "rank": 1,
                "parent_lf_index": 0,
                "ost_text": "Killarney to Dingle",
                "start_sec": 10.0,
                "end_sec": 40.0,
            },
            {
                "rank": 2,
                "parent_lf_index": 0,
                "ost_text": "County Kerry, Ireland",
                "start_sec": 80.0,
                "end_sec": 110.0,
            },
        ]
        payload = format_shorts_ost_audit("2026-05-24", entries)
        assert payload["date"] == "2026-05-24"
        assert len(payload["shorts"]) == 2
        assert payload["shorts"][0]["ost_text"] == "Killarney to Dingle"
        assert payload["shorts"][1]["rank"] == 2

    def test_write_round_trips(self, tmp_path):
        entries = [
            {
                "rank": 1,
                "parent_lf_index": 0,
                "ost_text": "Killarney to Dingle",
                "start_sec": 10.0,
                "end_sec": 40.0,
            },
        ]
        path = write_shorts_ost_audit(tmp_path, "2026-05-24", entries)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["shorts"][0]["ost_text"] == "Killarney to Dingle"


class TestOstAuditDriveWideCascade:
    """The audit must mirror the burn cascade — including drive-wide
    ``location`` — so review-text == burn-text, and ``thumbnail_text`` must
    be the REAL resolved burn text, not a divergent proxy.
    """

    def test_drive_wide_location_used_when_no_per_lf_text(self):
        # No per_lf_ost_text / per_lf_locations → the burn falls back to
        # the drive-wide --location, and so must the audit.
        state = {"per_lf_ost_text": {}, "per_lf_locations": {}}
        payload = format_ost_audit("2026-05-24", state, [1], location="Dingle, Kerry")
        lf = payload["lfs"][0]
        assert lf["ost_text"] == "Dingle, Kerry"
        # thumbnail_text is the REAL resolved burn text, not a separate proxy.
        assert lf["thumbnail_text"] == "Dingle, Kerry"

    def test_per_lf_text_still_wins_over_drive_wide(self):
        state = {"per_lf_ost_text": {1: "Killarney to Dingle"}, "per_lf_locations": {}}
        payload = format_ost_audit("2026-05-24", state, [1], location="Dingle, Kerry")
        assert payload["lfs"][0]["ost_text"] == "Killarney to Dingle"

    def test_per_lf_locations_wins_over_drive_wide(self):
        state = {"per_lf_ost_text": {}, "per_lf_locations": {1: "County Kerry"}}
        payload = format_ost_audit("2026-05-24", state, [1], location="Dingle, Kerry")
        assert payload["lfs"][0]["ost_text"] == "County Kerry"

    def test_omitting_location_keeps_empty_fallback(self):
        state = {"per_lf_ost_text": {}, "per_lf_locations": {}}
        payload = format_ost_audit("2026-05-24", state, [1])
        assert payload["lfs"][0]["ost_text"] == ""


class TestConfirmationStaleness:
    """A stale ``.confirmed`` / ``--ost-audit-confirmed`` must be
    INSUFFICIENT when the resolved label changed materially since the prior
    confirmed run. First-ever confirmations (no prior hash) still pass.
    """

    def test_first_run_no_prior_hash_is_not_stale(self):
        assert confirmation_is_stale(None, "abc123") is False

    def test_matching_hash_is_not_stale(self):
        assert confirmation_is_stale("abc123", "abc123") is False

    def test_material_change_is_stale(self):
        assert confirmation_is_stale("abc123", "def456") is True

    def test_label_edit_changes_hash(self):
        # A materially different label produces a different hash, so a
        # prior confirmation bound to the old hash is detected as stale.
        before = compute_ost_audit_hash(
            format_ost_audit("2026-05-24", {"per_lf_ost_text": {1: "A to B"}}, [1])
        )
        after = compute_ost_audit_hash(
            format_ost_audit("2026-05-24", {"per_lf_ost_text": {1: "C to D"}}, [1])
        )
        assert before != after
        assert confirmation_is_stale(before, after) is True
