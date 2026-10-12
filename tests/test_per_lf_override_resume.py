"""Per-LF override resume-preservation — a non-empty ``--per-lf-location`` must
win over the saved (empty) ``per_lf_overrides`` on resume.

Root cause: the operator re-ran a PAUSED drive with
``--per-lf-location '1=Bristol to Abingdon via I-81, VA'`` to correct a
GPS-derived label, but ``_try_resume`` reloaded the saved state (whose
``per_lf_overrides`` was empty from the original run) OVER the CLI value, so the
override was silently dropped and the wrong "Bristol to Hayter" burned. Same bug
class as the old ``--region`` resume stomp: re-apply the non-empty CLI override
after the merge so operator intent wins; an absent override preserves saved.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.drive_orchestrator import preserve_per_lf_overrides_on_resume
from gomoveshift.pipeline.drive_state_model import (
    _INT_KEY_DICT_FIELDS,
    PER_LF_OVERRIDE_MIRROR_FIELDS,
)


class TestPerLfOverrideMirrorFieldsIsSingleSourced:
    """PER_LF_OVERRIDE_MIRROR_FIELDS is the one declared list of Phase-A-derived
    per-LF label/text fields that must inherit a resumed --per-lf-location
    override — drive_orchestrator.py must import it, not re-type it, so a
    future per-LF label field added to PipelineState/_INT_KEY_DICT_FIELDS but
    left off this list fails a test instead of silently recurring the bug
    (stale derived label survives a Phase-D-only resume)."""

    def test_mirror_fields_matches_the_known_three(self):
        assert set(PER_LF_OVERRIDE_MIRROR_FIELDS) == {
            "per_lf_locations",
            "per_lf_ost_text",
            "per_lf_titles",
        }

    def test_mirror_fields_is_a_subset_of_int_key_dict_fields(self):
        # Catches a typo/rename that would silently drift the two lists apart.
        assert set(PER_LF_OVERRIDE_MIRROR_FIELDS) <= set(_INT_KEY_DICT_FIELDS)


class TestPerLfOverrideResumePreservation:
    def test_cli_override_wins_over_saved_empty(self):
        # Saved run had no overrides; this invocation supplies them.
        state = SimpleNamespace(per_lf_overrides={})
        preserve_per_lf_overrides_on_resume(state, {1: "Bristol to Abingdon via I-81, VA"})
        assert state.per_lf_overrides == {1: "Bristol to Abingdon via I-81, VA"}

    def test_cli_override_wins_over_saved_nonempty(self):
        state = SimpleNamespace(per_lf_overrides={1: "old label"})
        preserve_per_lf_overrides_on_resume(state, {1: "new label", 2: "second"})
        assert state.per_lf_overrides == {1: "new label", 2: "second"}

    def test_empty_cli_preserves_saved(self):
        # Operator did NOT pass --per-lf-location → handler builds {} → saved wins.
        state = SimpleNamespace(per_lf_overrides={1: "saved label"})
        preserve_per_lf_overrides_on_resume(state, {})
        assert state.per_lf_overrides == {1: "saved label"}

    def test_none_cli_preserves_saved(self):
        state = SimpleNamespace(per_lf_overrides={3: "saved"})
        preserve_per_lf_overrides_on_resume(state, None)
        assert state.per_lf_overrides == {3: "saved"}


class TestPerLfOverrideReachesDerivedLabels:
    """A Phase-D-only resume with ``--per-lf-location
    '1=Goleen, County Cork'`` updated ``per_lf_overrides`` only. Phase D reads the
    Phase-A-derived ``per_lf_locations`` / ``per_lf_ost_text`` / ``per_lf_titles``
    (Phase A does not re-run), so the saved 'To Goleen, County Cork' still became
    the title and hard-failed the metadata gate. A non-empty CLI override must
    reach every populated derived map."""

    def test_override_rewrites_populated_derived_maps(self):
        old = "To Goleen, County Cork"
        state = SimpleNamespace(
            per_lf_overrides={},
            per_lf_locations={1: old, 2: "Mizen Head, County Cork"},
            per_lf_ost_text={1: old, 2: "Mizen Head, County Cork"},
            per_lf_titles={1: old, 2: "Mizen Head, County Cork"},
        )
        preserve_per_lf_overrides_on_resume(state, {1: "Goleen, County Cork"})
        for field in PER_LF_OVERRIDE_MIRROR_FIELDS:
            assert getattr(state, field) == {
                1: "Goleen, County Cork",
                2: "Mizen Head, County Cork",
            }, field

    def test_unpopulated_derived_maps_stay_unset(self):
        # Fresh/early resume: Phase A has not derived labels yet — it will
        # consume per_lf_overrides itself, so nothing is seeded here.
        state = SimpleNamespace(
            per_lf_overrides={}, per_lf_locations=None, per_lf_ost_text={}, per_lf_titles=None
        )
        preserve_per_lf_overrides_on_resume(state, {1: "Goleen, County Cork"})
        assert state.per_lf_locations is None
        assert state.per_lf_ost_text == {}
        assert state.per_lf_titles is None

    def test_empty_cli_leaves_derived_maps(self):
        state = SimpleNamespace(per_lf_overrides={}, per_lf_locations={1: "saved"})
        preserve_per_lf_overrides_on_resume(state, {})
        assert state.per_lf_locations == {1: "saved"}
