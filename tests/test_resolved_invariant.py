"""Cross-surface location invariant.

Pins: ``check_location_invariant`` (HARD-fail only when the ACTUAL shipped
title/OST text names a county in ZERO sampled regions — the far-road class;
HARD-fail when a material county is omitted), ``assert_location_invariant_hard_fail`` (the
hard-fail escalation), and
that the drive path (``_run_final_validation_gate``) RAISES — not warns —
when an LF's ``location_invariant`` gate fails (``phase_d_gates`` otherwise
demotes every non-audio gate to ``logger.warning``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import assert_location_invariant_hard_fail
from gomoveshift.pipeline.drive_phases.phase_d_gates import _run_final_validation_gate
from gomoveshift.pipeline.validation.content_gates import check_location_invariant


def _lf(idx, *, total_duration=1800.0):
    return SimpleNamespace(index=idx, total_duration=total_duration)


# ── check_location_invariant ────────────────────────────────────────────


class TestCheckLocationInvariant:
    def test_dominant_in_regions_with_strong_plurality_passes(self):
        meta = {
            "location": "Beaufort, South Carolina",
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": ["Beaufort County, South Carolina", "Jasper County, South Carolina"],
                "vote": {
                    "Beaufort County, South Carolina": 0.85,
                    "Jasper County, South Carolina": 0.15,
                },
            },
        }
        res = check_location_invariant(meta)
        assert res.gate == "location_invariant"
        assert res.passed is True

    def test_title_county_desynced_from_regions_hard_fails(self):
        # The far-road class the gate MUST catch: the SHIPPED title names
        # Colleton County, but the GPS trace only ever sampled Charleston —
        # a county the footage never visited. The resolved model's OWN
        # ``dominant_county`` is (by construction) a member of ``regions``, so
        # reading it here is vacuous; the gate must reconcile the real title
        # text against the sampled regions instead.
        meta = {
            "title": "Drive in Colleton County",
            "resolved_location": {
                "dominant_county": "Charleston County",
                "dominant_state": "South Carolina",
                "regions": ["Charleston County, South Carolina"],
                "vote": {"Charleston County, South Carolina": 1.0},
            },
        }
        res = check_location_invariant(meta)
        assert res.passed is False
        assert "HARD FAIL" in res.message
        assert "ZERO sampled regions" in res.message

    def test_title_county_in_regions_passes(self):
        # The shipped title names a county the trace actually sampled.
        meta = {
            "title": "Charleston County, South Carolina Scenic Drive",
            "resolved_location": {
                "dominant_county": "Charleston County",
                "dominant_state": "South Carolina",
                "regions": [
                    "Charleston County, South Carolina",
                    "Berkeley County, South Carolina",
                ],
                "vote": {
                    "Charleston County, South Carolina": 0.85,
                    "Berkeley County, South Carolina": 0.15,  # below the material floor
                },
            },
        }
        assert check_location_invariant(meta).passed is True

    def test_town_form_title_is_noop(self):
        # A town-form title (no ``County`` token) can't be reconciled against a
        # county-region set across state-abbreviation mismatches, so the hard
        # check no-ops — mirrors the shorts gate's town-form handling; never a
        # false abort of a legitimate town-anchored title.
        meta = {
            "title": "Fort Fremont, SC Scenic Drive",
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": ["Beaufort County, South Carolina"],
                "vote": {"Beaufort County, South Carolina": 1.0},
            },
        }
        assert check_location_invariant(meta).passed is True

    def test_weak_plurality_label_omitting_material_counties_hard_fails(self):
        # Genuinely split across counties; the town-form label names none of them.
        meta = {
            "location": "Somewhere, South Carolina",
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": [
                    "Beaufort County, South Carolina",
                    "Jasper County, South Carolina",
                    "Colleton County, South Carolina",
                ],
                "vote": {
                    "Beaufort County, South Carolina": 0.40,
                    "Jasper County, South Carolina": 0.35,
                    "Colleton County, South Carolina": 0.25,
                },
            },
        }
        res = check_location_invariant(meta)
        # Coverage rule: three material counties, none named.
        assert res.passed is False
        assert "Jasper County, South Carolina" in res.message

    def test_town_form_desync_hard_fails_with_place_sequence(self):
        # With town-level truth present, a town-form label naming a town
        # the route never passed is no longer a silent no-op.
        meta = {
            "title": "Walterboro, SC Scenic Drive",
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": ["Beaufort County, South Carolina"],
                "vote": {"Beaufort County, South Carolina": 1.0},
                "ordered_place_sequence": ["Fort Fremont", "Saint Helena Island"],
            },
        }
        res = check_location_invariant(meta)
        assert res.passed is False
        assert "town-form" in res.message

    def test_town_form_matching_destination_passes(self):
        # Destination-form title validates against the resolved destination POI.
        meta = {
            "title": "Donnelley Wildlife Management Area, SC",
            "resolved_location": {
                "dominant_county": "Colleton County",
                "dominant_state": "South Carolina",
                "regions": ["Colleton County, South Carolina"],
                "vote": {"Colleton County, South Carolina": 1.0},
                "ordered_place_sequence": ["Green Pond"],
                "destination": "Donnelley Wildlife Management Area",
            },
        }
        assert check_location_invariant(meta).passed is True

    def test_endpoint_town_absent_from_place_sequence_hard_fails(self):
        # "A to B" endpoint towns must appear in the route's place
        # sequence / regions — the wrong-town-in-right-county class.
        meta = {
            "title": "Ridgeland to Beaufort via US-21, SC",
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": ["Beaufort County, South Carolina"],
                "vote": {"Beaufort County, South Carolina": 1.0},
                "ordered_place_sequence": ["Beaufort", "Port Royal"],
            },
        }
        res = check_location_invariant(meta)
        assert res.passed is False
        assert "Ridgeland" in res.message
        assert "wrong-town-in-right-county" in res.message

    def test_endpoint_towns_in_place_sequence_pass(self):
        meta = {
            "title": "Beaufort to Port Royal via US-21, SC",
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": ["Beaufort County, South Carolina"],
                "vote": {"Beaufort County, South Carolina": 1.0},
                "ordered_place_sequence": ["Beaufort", "Port Royal"],
            },
        }
        assert check_location_invariant(meta).passed is True

    def test_cross_artifact_county_desync_hard_fails(self):
        # A stray-county tag or description hashtag fails the gate
        # even when the title itself is consistent.
        base = {
            "dominant_county": "Beaufort County",
            "dominant_state": "South Carolina",
            "regions": ["Beaufort County, South Carolina"],
            "vote": {"Beaufort County, South Carolina": 1.0},
        }
        meta = {
            "title": "Beaufort County, South Carolina Scenic Drive",
            "tags": ["scenic drive", "colleton county"],
            "resolved_location": dict(base),
        }
        res = check_location_invariant(meta)
        assert res.passed is False
        assert "colleton county" in res.message
        meta2 = {
            "title": "Beaufort County, South Carolina Scenic Drive",
            "description": "Hook.\n\n📍 Location: Beaufort County, SC\n\n#ColletonCounty",
            "resolved_location": dict(base),
        }
        res2 = check_location_invariant(meta2)
        assert res2.passed is False
        assert "#ColletonCounty" in res2.message

    def test_consistent_cross_artifact_surfaces_pass(self):
        meta = {
            "title": "Beaufort County, South Carolina Scenic Drive",
            "description": (
                "Hook.\n\nBeaufort County, South Carolina 4K drive — tour.\n\n"
                "📍 Location: Beaufort County, SC\n\n#BeaufortCounty #SouthCarolina"
            ),
            "tags": ["beaufort county", "scenic drive"],
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": ["Beaufort County, South Carolina"],
                "vote": {"Beaufort County, South Carolina": 1.0},
            },
        }
        assert check_location_invariant(meta).passed is True

    def test_seo_line_wrong_county_is_flagged(self):
        # The above-the-fold SEO search line is produced as "… 4K drive …"
        # (the scenic purge dropped "scenic"). The gate's line marker must
        # match that, or a wrong county in the SEO line escapes the
        # cross-artifact sweep. Here ONLY the SEO line carries a bad county —
        # the title is town-form + correct and the hashtag is correct — so the
        # failure can come from nowhere else.
        meta = {
            "title": "POV Driving Beaufort, SC",
            "description": (
                "Hook.\n\nColleton County, South Carolina 4K drive — tour.\n\n#BeaufortCounty"
            ),
            "resolved_location": {
                "dominant_county": "Beaufort County",
                "dominant_state": "South Carolina",
                "regions": ["Beaufort County, South Carolina"],
                "vote": {"Beaufort County, South Carolina": 1.0},
            },
        }
        res = check_location_invariant(meta)
        assert res.passed is False
        assert "SEO line" in res.message
        assert "colleton" in res.message.lower()

    def test_no_resolved_location_is_noop(self):
        assert check_location_invariant({"location": "x"}).passed is True

    def test_empty_regions_is_noop(self):
        # Also covers quorum-degraded dominant=None (hard check skipped).
        meta = {"resolved_location": {"dominant_county": None, "regions": [], "vote": {}}}
        assert check_location_invariant(meta).passed is True


# ── assert_location_invariant_hard_fail ─────────────────────────────────


class TestAssertLocationInvariantHardFail:
    def test_no_failures_is_silent_noop(self):
        assert_location_invariant_hard_fail([])

    def test_single_failure_raises(self):
        with pytest.raises(RuntimeError) as exc:
            assert_location_invariant_hard_fail(
                [
                    "  LF 2 (project_lf02_FINAL.mp4): title county 'Oakland County, NC' in ZERO regions"
                ]
            )
        msg = str(exc.value)
        assert "HARD FAIL" in msg
        assert "LF 2" in msg
        assert "far-road" in msg


# ── drive path RAISES (the hard-fail requirement) ───────────────────────────────


class TestDrivePathRaisesOnLocationInvariant:
    def _mk_config(self, *, skip_audio=False):
        return SimpleNamespace(
            drive=SimpleNamespace(skip_audio=skip_audio),
            output=SimpleNamespace(audio_max_lufs=-18.0, audio_target_tp=-1.5),
        )

    def test_failing_location_invariant_raises_on_drive_path(self, tmp_path):
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        location_gate = SimpleNamespace(
            gate="location_invariant",
            passed=False,
            message="location invariant HARD FAIL: title county 'Oakland County, NC' in ZERO regions",
        )
        result = SimpleNamespace(passed=False, gates=[location_gate], summary=lambda: "loc fail")
        validator = SimpleNamespace(validate=lambda **kw: result)
        with pytest.raises(RuntimeError, match="Location invariant HARD FAIL"):
            _run_final_validation_gate(
                validator_cls=lambda **kw: validator,
                config=self._mk_config(skip_audio=True),
                drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
                regular_lf_paths=[big],
                metadata_per_lf={1: {"resolved_location": {}}},
            )

    def test_passing_location_invariant_does_not_raise(self, tmp_path):
        big = tmp_path / "lf.mp4"
        big.write_bytes(b"x" * (1024 * 1024 + 1))
        location_gate = SimpleNamespace(gate="location_invariant", passed=True, message="ok")
        result = SimpleNamespace(passed=True, gates=[location_gate], summary=lambda: "ok")
        validator = SimpleNamespace(validate=lambda **kw: result)
        # No raise — the gate passed.
        _run_final_validation_gate(
            validator_cls=lambda **kw: validator,
            config=self._mk_config(skip_audio=True),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(1)]),
            regular_lf_paths=[big],
            metadata_per_lf={1: {}},
        )
