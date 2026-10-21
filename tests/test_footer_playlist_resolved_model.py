"""F-PL: the footer playlist line follows the persisted ResolvedLocation.

The footer region comes from ``region_resolver.derive_region`` off the LF's
canonical ``ResolvedLocation``. These tests drive the real entry point
(``generate()``) with the model as production hands it over: built as a
``ResolvedLocation`` (the shape ``build_resolved_location`` returns), written
into ``PipelineState`` and reloaded through ``to_dict`` / ``from_dict`` (the
resume path). The label tails are invented non-county places, so a label
re-parse cannot reach a configured region key; only the model can.
"""

from __future__ import annotations

import json

import pytest

from gomoveshift.config import Config
from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline import longform_metadata
from gomoveshift.pipeline.drive_state_model import PipelineState
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator
from gomoveshift.pipeline.region_resolver import derive_region

PLAYLISTS = {
    "ireland": "https://www.youtube.com/playlist?list=PLsyntheticIE",
    "usa": "https://www.youtube.com/playlist?list=PLsyntheticUS",
}

# (configured region key, display word, label with a non-county tail, model)
CASES = [
    (
        "ireland",
        "Ireland",
        "Brindlemoor to Quillhaven, Harrow Peninsula",
        ResolvedLocation(
            dominant_county="County Harrow", dominant_state=None,
            country_code="Ireland", is_us=False,
        ),
    ),
    (
        "usa",
        "USA",
        "Brindlemoor to Quillhaven, Harrow Hollow",
        ResolvedLocation(
            dominant_county="Harrow County", dominant_state="Virginia",
            country_code="US", is_us=True,
        ),
    ),
]  # fmt: skip


class _StubGenerator(LongFormMetadataGenerator):
    def _init_seo_manager(self):
        self._manager = None
        self._optimizer = None
        self._seo_degraded = True

    def _init_ledger(self):
        self._ledger = None
        self._ledger_path = None


@pytest.fixture(autouse=True)
def _footer_config(monkeypatch):
    cfg = Config.model_validate({"drive": {"playlist_url_by_region": PLAYLISTS}})
    monkeypatch.setattr(longform_metadata, "_get_config", lambda config_path=None: cfg)


def _reloaded(model: ResolvedLocation) -> ResolvedLocation:
    """The model as resume hands it back: state → JSON → state."""
    state = PipelineState(per_lf_resolved_location={1: model})
    loaded = PipelineState.from_dict(json.loads(json.dumps(state.to_dict())))
    return loaded.per_lf_resolved_location[1]


@pytest.mark.parametrize(("key", "display", "label", "model"), CASES)
def test_reloaded_model_derives_the_configured_region_key(key, display, label, model):
    reloaded = _reloaded(model)
    assert isinstance(reloaded, ResolvedLocation)
    assert derive_region(label) != key  # the label alone cannot reach it
    assert derive_region(label, resolved=reloaded) == key


@pytest.mark.parametrize(("key", "display", "label", "model"), CASES)
def test_generate_emits_the_playlist_line_for_the_reloaded_model(
    tmp_path, key, display, label, model
):
    result = _StubGenerator(seo_storage_path=tmp_path).generate(
        location=label,
        video_duration=1800.0,
        project_name="t",
        recording_date="2025-09-14",
        resolved_location=_reloaded(model),
    )
    assert f"▶ Full {display} playlist: {PLAYLISTS[key]}" in result["description"]
    other = next(url for k, url in PLAYLISTS.items() if k != key)
    assert other not in result["description"]


def test_a_raw_provenance_dict_is_not_a_region_signal():
    """A plain dict (e.g. the provenance sidecar's summary) carries no
    attributes, so it falls through to the label: no playlist key."""
    label = CASES[0][2]
    assert derive_region(label, resolved={"country_code": "Ireland"}) != "ireland"
