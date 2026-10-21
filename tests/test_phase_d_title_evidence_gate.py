"""GMS-17b: Phase D refuses a long-form title whose place claims its LF's
location-evidence record does not back.

An "A to B" title needs both endpoints entered by the trace in travel order.
No record behind such a title is a refusal (logged ``E``), never a skip. A
title with no A-to-B claim has nothing to check.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import _build_metadata_per_lf
from gomoveshift.pipeline.location_evidence import (
    TitleEvidenceError,
    refuse_unsupported_title,
)

from _state_helpers import make_state
from test_location_evidence import _record

_LOGGER = "gomoveshift.pipeline.location_evidence"


class TestRefuseUnsupportedTitle:
    def test_a_backed_title_passes(self):
        refuse_unsupported_title("Alder to Dogwood, VA in 4K", _record(), lf_index=1)

    def test_a_dict_record_from_state_is_read(self):
        refuse_unsupported_title("Alder to Dogwood, VA in 4K", _record().to_dict(), lf_index=1)

    def test_no_record_behind_a_claim_is_refused_and_logged_e(self, caplog):
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            with pytest.raises(TitleEvidenceError, match="no location evidence record"):
                refuse_unsupported_title("Alder to Dogwood, VA in 4K", None, lf_index=3)
        assert any(
            "TITLE-EVIDENCE E" in r.getMessage() and "LF 3" in r.getMessage()
            for r in caplog.records
        )

    def test_endpoints_against_the_travel_order_are_refused(self):
        with pytest.raises(TitleEvidenceError, match="travel order"):
            refuse_unsupported_title("Dogwood to Alder, VA in 4K", _record(), lf_index=1)

    def test_an_endpoint_the_trace_never_entered_is_refused(self):
        with pytest.raises(TitleEvidenceError, match="'Elmwood' is never entered"):
            refuse_unsupported_title("Alder to Elmwood, VA in 4K", _record(), lf_index=1)

    def test_a_title_with_no_a_to_b_claim_has_nothing_to_check(self):
        refuse_unsupported_title("Ash County, VA in 4K", None, lf_index=1)


def _build(tmp_path, title, evidence):
    gen = MagicMock()
    gen.generate.side_effect = lambda **kw: {"title": title, "description": "d"}
    with patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    ):
        return _build_metadata_per_lf(
            meta_gen=gen,
            drive_plan=SimpleNamespace(
                regular_lfs=[SimpleNamespace(index=1, total_duration=1800.0, segments=[])]
            ),
            regular_lf_paths=["/v/lf.mp4"],
            state=make_state(per_lf_location_evidence=evidence),
            location="x",
            publish_day_full="Wednesday",
            project_base="p",
            output_base=tmp_path,
        )


class TestPhaseDTitlePath:
    def test_phase_d_refuses_an_unbacked_title_before_writing_metadata(self, tmp_path):
        with pytest.raises(TitleEvidenceError):
            _build(tmp_path, "Alder to Dogwood, VA in 4K", {})
        assert not list(tmp_path.glob("*_youtube.json"))

    def test_phase_d_ships_a_backed_title(self, tmp_path):
        out = _build(tmp_path, "Alder to Dogwood, VA in 4K", {1: _record().to_dict()})
        assert out[1]["title"] == "Alder to Dogwood, VA in 4K"
