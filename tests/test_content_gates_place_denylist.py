"""content_gates hard-fails shipped metadata carrying a denylisted place
name -- belt and braces for a substitution ``title_builder``
missed."""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.validation.content_gates import check_metadata_quality
from gomoveshift.pipeline.validation.place_corrections import PlaceCorrections


def _base_metadata(**overrides):
    meta = {
        "title": "Beaufort to Port Royal, SC",
        "description": "A scenic drive through Beaufort County, South Carolina.",
        "location": "Beaufort County, South Carolina",
        "tags": ["Beaufort", "South Carolina"],
    }
    meta.update(overrides)
    return meta


_DENY = PlaceCorrections(denylist=("Frogmore Depot",))


class TestDenylistedPlaceNameGate:
    def test_clean_metadata_passes_with_denylist_configured(self):
        assert check_metadata_quality(_base_metadata(), place_corrections=_DENY).passed

    @pytest.mark.parametrize(
        "overrides",
        [
            {"title": "Beaufort to Frogmore Depot, SC"},
            {"description": "A drive out to Frogmore Depot and back."},
            {"tags": ["Beaufort", "Frogmore Depot"]},
        ],
    )
    def test_denylisted_name_hard_fails(self, overrides):
        result = check_metadata_quality(_base_metadata(**overrides), place_corrections=_DENY)
        assert not result.passed and "Frogmore Depot" in result.message

    def test_empty_ledger_is_a_strict_noop(self):
        meta = _base_metadata(title="Beaufort to Frogmore Depot, SC")
        assert check_metadata_quality(meta, place_corrections=PlaceCorrections()).passed
