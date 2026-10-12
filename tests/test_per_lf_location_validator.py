"""Direct coverage for the ``--per-lf-location`` argparse validator.

This module is the dedicated unit-test file for the validator. Two things to
note about the framing:

1. **Mental model.** A first draft might summarise
   the validator as enforcing three rules — "duplicate indices rejected,
   labels equal to ``--location`` rejected, every GPS-less LF must have
   an override". A line-by-line re-read against
   ``_validate_per_lf_location_args`` in ``gomoveshift/cli/parser.py`` shows the actual contract is
   different:

       Rule 1 (all-equal): reject only when EVERY override label equals
       ``--location`` verbatim. Single-of-many equality is intentionally
       allowed (see the ``_validate_per_lf_location_args`` docstring).
       Rule 2 (specificity floor): when overrides are set, ``--location``
       must have >= 2 comma-separated components.
       Rule 3 (comma-suffix): each per-LF label must end with
       ``--location`` as a comma-suffix.

   Duplicate indices are *intentionally* accepted (handlers/drive.py
   reduces the list-of-tuples to a dict with last-write-wins). GPS-less
   coverage is enforced post-discovery in the PhaseBC entry gate, not
   here, because LF indices aren't known until clip discovery runs (see the
   "Per-LF location is total — no drive-wide fallback" contract).

2. **Test discipline.** The validator raises ``argparse.ArgumentTypeError``
   directly (it is invoked post-parse from ``cli/handlers/drive.py``,
   not as an argparse ``type=`` callback). Tests therefore assert with
   ``pytest.raises(argparse.ArgumentTypeError)`` against the message body,
   matching the in-repo style established by ``tests/test_drive_cli.py``.

Coverage target: 100% line coverage on ``_validate_per_lf_location_args``
and ``_per_lf_location_kv``.

References:
- ``gomoveshift/cli/parser.py`` (``_per_lf_location_kv`` — shape parser)
- ``gomoveshift/cli/parser.py`` (``_validate_per_lf_location_args``)
- ``gomoveshift/cli/handlers/drive.py`` (call site + duplicate-index reduction)
"""

from __future__ import annotations

import argparse

import pytest

from gomoveshift.cli.parser import (
    _per_lf_location_kv,
    _validate_per_lf_location_args,
)


# ---------------------------------------------------------------------------
# _per_lf_location_kv — argparse ``type=`` shape parser
# ---------------------------------------------------------------------------


class TestPerLfLocationKvShape:
    """Shape-level rejection: every malformed ``INDEX=LABEL`` must raise.

    These mirror the failure cases ``cli/parser.py:_per_lf_location_kv``
    enumerates explicitly — missing ``=``, non-integer key, idx < 1,
    empty label, and the whitespace-trimming pass-through.
    """

    def test_accepts_well_formed_kv(self):
        assert _per_lf_location_kv("1=Alpha to Beta via Route 9, Example County") == (
            1,
            "Alpha to Beta via Route 9, Example County",
        )

    def test_trims_whitespace_around_key_and_value(self):
        assert _per_lf_location_kv("  2  =  Route B  ") == (2, "Route B")

    def test_rejects_missing_equals(self):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _per_lf_location_kv("no-equals-sign")
        assert "INDEX=LABEL" in str(exc.value)

    def test_rejects_non_integer_key(self):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _per_lf_location_kv("abc=Some Route")
        assert "INDEX must be an integer" in str(exc.value)

    def test_rejects_zero_index(self):
        # LF indices are 1-based — matches ``RegularLFPlan.index``.
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _per_lf_location_kv("0=Route")
        assert ">= 1" in str(exc.value)

    def test_rejects_negative_index(self):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _per_lf_location_kv("-3=Route")
        assert ">= 1" in str(exc.value)

    def test_rejects_empty_label(self):
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _per_lf_location_kv("1=")
        assert "non-empty" in str(exc.value)

    def test_rejects_whitespace_only_label(self):
        # Whitespace is trimmed before the emptiness check — so "  " -> ""
        # and the non-empty rule still fires.
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _per_lf_location_kv("1=   ")
        assert "non-empty" in str(exc.value)


# ---------------------------------------------------------------------------
# _validate_per_lf_location_args — cross-flag rules
# ---------------------------------------------------------------------------


class TestValidatorHappyPath:
    """Well-formed invocations must pass cleanly."""

    def test_single_override_with_two_component_location(self):
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[(1, "Alpha to Beta via Route 9, Example County, Exampleland")],
        )
        _validate_per_lf_location_args(ns)  # no raise

    def test_multiple_overrides_all_comma_suffix(self):
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[
                (1, "Alpha to Beta, Example County, Exampleland"),
                (2, "Beta to Gamma, Example County, Exampleland"),
            ],
        )
        _validate_per_lf_location_args(ns)  # no raise

    def test_no_overrides_supplied_returns_silently(self):
        # Validator is a no-op when ``per_lf_locations`` is ``None`` or
        # empty — drive-wide-only invocations are the GPS-bearing common
        # case and don't trip any cross-flag rule.
        ns = argparse.Namespace(location="Exampleland", per_lf_locations=None)
        _validate_per_lf_location_args(ns)  # no raise

        ns2 = argparse.Namespace(location="Exampleland", per_lf_locations=[])
        _validate_per_lf_location_args(ns2)  # no raise

    def test_empty_drive_wide_with_overrides_returns_silently(self):
        # When ``--location`` is empty, the cross-flag specificity /
        # comma-suffix checks don't fire (the PhaseBC coverage gate is
        # the appropriate enforcement point further downstream).
        ns = argparse.Namespace(
            location="",
            per_lf_locations=[(1, "Alpha to Beta Pass, Example County, Exampleland")],
        )
        _validate_per_lf_location_args(ns)  # no raise


class TestValidatorRule1AllEqualRejection:
    """Rule 1: reject only when EVERY override label equals
    ``--location`` verbatim. Single-of-many equality is allowed."""

    def test_rejects_when_every_override_equals_location(self):
        ns = argparse.Namespace(
            location="Exampleland",
            per_lf_locations=[(1, "Exampleland"), (2, "Exampleland")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        assert "more specific" in str(exc.value)

    def test_rejects_when_single_override_equals_location(self):
        # A single override equal to ``--location`` still trips the rule
        # — "all" over a 1-element set is also "every".
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[(1, "Example County, Exampleland")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        assert "more specific" in str(exc.value)

    def test_accepts_mixed_overrides_one_equals_one_does_not(self):
        # Docstring at ``_validate_per_lf_location_args`` explicitly: "Mixing some
        # overrides equal to ``--location`` with others that aren't is
        # *allowed* (gate 1 only fires on ALL == location)."
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[
                (1, "Example County, Exampleland"),
                (2, "Alpha to Beta, Example County, Exampleland"),
            ],
        )
        _validate_per_lf_location_args(ns)  # no raise


class TestValidatorRule2SpecificityFloor:
    """Rule 2: when overrides are set, ``--location`` must
    have >= 2 comma-separated components."""

    def test_rejects_single_component_drive_wide(self):
        ns = argparse.Namespace(
            location="Exampleland",
            per_lf_locations=[(1, "Alpha to Beta Pass, Example County, Exampleland")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        msg = str(exc.value)
        assert ">= 2 components" in msg
        # Suggestion is the longest common comma-suffix of the per-LF
        # labels' tails — for one label the tail is everything after the
        # first comma-component.
        assert "Example County, Exampleland" in msg

    def test_rejection_message_omits_suggestion_when_tails_share_nothing(self):
        # Per-LF tails have no common trailing components → suggestion
        # derives to the empty string → message omits the
        # ``Suggested:`` tail. Exercises the falsy branch of
        # ``if suggested else ""`` inside rule 2.
        ns = argparse.Namespace(
            location="Exampleland",
            per_lf_locations=[
                (1, "Rome, Italy"),
                (2, "Madrid, Spain"),
            ],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        msg = str(exc.value)
        assert ">= 2 components" in msg
        assert "Suggested:" not in msg

    def test_rejection_message_includes_suggestion_when_derivable(self):
        # The rule-2 error message appends ``Suggested: --location ...``
        # iff ``_suggest_drive_wide_from_per_lf`` returns a non-empty
        # string. For a single-component per-LF label the derivation
        # falls back to the label itself; the suggestion is therefore
        # present and equals ``--location 'Exampleland-Themed Route'``.
        # Exercises the truthy branch of ``if suggested else ""``.
        ns = argparse.Namespace(
            location="Exampleland",
            per_lf_locations=[(1, "Exampleland-Themed Route")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        msg = str(exc.value)
        assert ">= 2 components" in msg
        assert "Suggested: --location 'Exampleland-Themed Route'" in msg


class TestValidatorRule3CommaSuffix:
    """Rule 3: every per-LF label must end with ``--location``
    as a comma-suffix at the comma boundary."""

    def test_rejects_when_drive_wide_is_not_comma_suffix(self):
        # "Beta Pass" lives inside the first comma-component of per-LF
        # ("Alpha to Beta Pass"), so "Beta Pass, Example County, Exampleland" is
        # NOT a comma-suffix of the per-LF label at the comma boundary.
        ns = argparse.Namespace(
            location="Beta Pass, Example County, Exampleland",
            per_lf_locations=[(1, "Alpha to Beta Pass, Example County, Exampleland")],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        assert "comma-suffix" in str(exc.value)

    def test_rejection_message_includes_suggestion_derived_from_per_lf(self):
        ns = argparse.Namespace(
            location="Beta Pass, Example County, Exampleland",
            per_lf_locations=[
                (1, "Alpha to Delta, Example County, Exampleland"),
                (2, "Delta to Gamma, Example County, Exampleland"),
            ],
        )
        with pytest.raises(argparse.ArgumentTypeError) as exc:
            _validate_per_lf_location_args(ns)
        msg = str(exc.value)
        assert "comma-suffix" in msg
        # Suggested drive-wide is the longest common comma-suffix of
        # the per-LF tails — both labels share ``Example County, Exampleland``.
        assert "Suggested: --location 'Example County, Exampleland'" in msg

    def test_accepts_drive_wide_that_is_exact_comma_suffix_tail(self):
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[(1, "Alpha to Beta, Example County, Exampleland")],
        )
        _validate_per_lf_location_args(ns)  # no raise


# ---------------------------------------------------------------------------
# Deliberate-behavior pins
# ---------------------------------------------------------------------------


class TestValidatorDeliberateBehaviors:
    """Pins for behaviors that look like bugs but are intentional, so a
    future change can't silently flip them.

    Each test cites the source comment that documents the intent.
    """

    def test_duplicate_index_is_silently_accepted_last_write_wins(self):
        """``handlers/drive.py`` documents the contract:

            # Reduce list-of-tuples → dict so downstream consumers see a
            # ``{lf_index: label}`` shape. Last-write-wins on duplicate
            # indices (operator passes ``--per-lf-location 1=A
            # --per-lf-location 1=B`` → B wins; cheap predictable
            # behavior).

        The validator therefore must NOT reject duplicate indices —
        that's the handler's reduction step's job, and it's not a
        rejection but a deterministic merge. A future change that adds
        a duplicate-index rejection would break operator workflows that
        currently rely on the override-rewrite pattern.
        """
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[
                (1, "Alpha to Beta, Example County, Exampleland"),
                (1, "Alpha to Beta Pass, Example County, Exampleland"),  # same idx, last wins
            ],
        )
        _validate_per_lf_location_args(ns)  # no raise

    def test_cross_lf_duplicate_labels_are_accepted(self):
        """``_validate_per_lf_location_args`` docstring: "Cross-LF duplicate labels are
        *allowed*." Two LFs that happen to be on the same road segment
        should not fight the validator. The all-equal rule (rule 1)
        only fires when EVERY label also equals ``--location`` — that's
        a different invariant."""
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[
                (1, "Beta Pass loop, Example County, Exampleland"),
                (2, "Beta Pass loop, Example County, Exampleland"),
            ],
        )
        _validate_per_lf_location_args(ns)  # no raise

    def test_validator_does_not_enforce_gps_less_coverage(self):
        """The "Per-LF location is total — no drive-wide fallback" contract
        pins the layering: every GPS-less LF must have
        an override, but the *coverage* check (do we have one override
        per GPS-less LF index?) lives at the PhaseBC entry gate, NOT
        in this validator. Argparse-time can't know which LF indices
        are GPS-less because clip discovery hasn't run yet.

        This test asserts the validator's contract by negative: an
        invocation with overrides for LF 1 only (no LF 2 override) is
        accepted at argparse-time even though LF 2 might be GPS-less
        downstream. PhaseBC will catch that case later with a
        ``ValueError`` from ``_assert_per_lf_locations_resolvable``.
        """
        ns = argparse.Namespace(
            location="Example County, Exampleland",
            per_lf_locations=[(1, "Alpha to Beta, Example County, Exampleland")],
        )
        # No raise — coverage isn't this validator's job.
        _validate_per_lf_location_args(ns)  # no raise


class TestValidatorRule0LeadingPreposition:
    """A label such as ``--per-lf-location '1=To Alpha, County Beta'`` passed
    launch, burned, then hard-failed the pre-upload metadata gate hours later
    (``title: starts with a bare preposition``) — the YouTube title derives
    from the burned OST, and the gate refuses a leading preposition. Refuse
    the label at launch, before any encode, with the same pattern."""

    @pytest.mark.parametrize(
        "label", ["To Alpha, County Beta", "Through Gamma, Beta", "Around Delta, Beta"]
    )
    def test_rejects_leading_preposition_even_without_location(self, label):
        ns = argparse.Namespace(location="", per_lf_locations=[(1, label)])
        with pytest.raises(argparse.ArgumentTypeError, match="bare preposition"):
            _validate_per_lf_location_args(ns)

    def test_accepts_place_label_and_inner_to(self):
        ns = argparse.Namespace(
            location="",
            per_lf_locations=[(1, "Alpha, County Beta"), (2, "Bristol to Abingdon, VA")],
        )
        _validate_per_lf_location_args(ns)  # no raise
