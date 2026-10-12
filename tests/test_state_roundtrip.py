"""``ResolvedLocation`` model + ``PipelineState`` wiring.

Covers the four state-model registrations for ``per_lf_resolved_location``
(int-key survival, dataclass-value rehydration both in-process and after a
JSON round-trip, missing-key OLD-pipeline backward-compat) and the
resume-path point provenance: ``_build_gps_points_by_lf`` rebuilds a
non-empty ``gps_points_by_lf`` from the persisted ``gps_per_episode`` after
a state round-trip with no in-memory points, so the voting step can vote (not
bare-mean) on resume.
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.models import (
    DrivePlan,
    LFSegment,
    RegularLFPlan,
    ResolvedLocation,
)
from gomoveshift.pipeline.drive_state_model import (
    _INT_KEY_DICT_FIELDS,
    _dict_value_dataclass_fields_map,
    PipelineState,
)


def _make_resolved() -> ResolvedLocation:
    return ResolvedLocation(
        dominant_county="Beaufort County",
        dominant_state="SC",
        country_code="US",
        is_us=True,
        ordered_place_sequence=["Beaufort County, SC", "Jasper County, SC"],
        regions=["Beaufort County, SC", "Jasper County, SC"],
        vote={"Beaufort County, SC": 0.8, "Jasper County, SC": 0.2},
        start_label="Beaufort, SC",
        end_label="Hardeeville, SC",
        endpoints={"start": [32.43, -80.67], "end": [32.29, -81.08]},
        destination="Donnelley Wildlife Management Area",
    )


def _make_drive_plan() -> DrivePlan:
    seg = LFSegment(
        source_file_index=0,
        source_path=Path("/src/ep0.mp4"),
        # Wide window so the ``_build_gps_points_by_lf`` time gate admits every
        # trackpoint regardless of the GPS offset the rebuild path uses.
        start_time=0.0,
        end_time=1_000_000.0,
        original_start_time=10.0,
        original_end_time=40.0,
        composite_score=0.9,
        gps_center_lat=32.4,
        gps_center_lon=-80.7,
        terrain_type="coastal",
        contiguous_with_previous=False,
    )
    lf = RegularLFPlan(
        index=1,
        segments=[seg],
        montage_peaks=[],
        shorts_target=14,
        region="usa",
        total_duration=30.0,
    )
    return DrivePlan(
        source_files=[Path("/src/ep0.mp4")],
        processed_files=[Path("/proc/ep0.mp4")],
        regular_lfs=[lf],
        score_threshold=0.5,
        total_source_duration=100.0,
        total_highlight_duration=30.0,
    )


# ── State-model registrations ────────────────────────────────────────────


def test_per_lf_resolved_location_registered_int_key_field():
    """Field is in the int-key-dict list (keys survive)."""
    assert "per_lf_resolved_location" in _INT_KEY_DICT_FIELDS


def test_per_lf_resolved_location_registered_dict_value_dataclass():
    """Field maps to ResolvedLocation in the real registry."""
    assert _dict_value_dataclass_fields_map().get("per_lf_resolved_location") is ResolvedLocation


def test_per_lf_resolved_location_default_none():
    """Field declared with a None default (fresh runs)."""
    assert PipelineState().per_lf_resolved_location is None


def test_resolved_location_roundtrips_int_keys_and_dataclass():
    """Int keys + ResolvedLocation values survive JSON round-trip."""
    state = PipelineState()
    state.per_lf_resolved_location = {1: _make_resolved(), 2: _make_resolved()}

    restored = PipelineState.from_dict(state.to_dict())

    # int keys survive (not stringified)
    assert set(restored.per_lf_resolved_location) == {1, 2}
    for k in restored.per_lf_resolved_location:
        assert isinstance(k, int)
    # values are ResolvedLocation instances, not plain dicts
    rl = restored.per_lf_resolved_location[1]
    assert isinstance(rl, ResolvedLocation)
    assert rl.dominant_county == "Beaufort County"
    assert rl.is_us is True
    assert rl.vote == {"Beaufort County, SC": 0.8, "Jasper County, SC": 0.2}
    assert rl.regions == ["Beaufort County, SC", "Jasper County, SC"]
    assert rl.endpoints == {"start": [32.43, -80.67], "end": [32.29, -81.08]}
    # The destination POI round-trips on the canonical model.
    assert rl.destination == "Donnelley Wildlife Management Area"


def test_in_process_update_rehydrates_dataclass():
    """``update`` from a raw dict also rehydrates the value."""
    state = PipelineState()
    raw = {"per_lf_resolved_location": {"3": _make_resolved().__dict__}}
    state.update(raw)
    assert isinstance(state.per_lf_resolved_location[3], ResolvedLocation)


def test_old_pipeline_state_without_field_loads_and_recomputes():
    """A state file predating the field (no key) loads; field stays None → recompute-on-resume."""
    legacy = {
        "region": "usa",
        "publish_day": "tue",
        # NOTE: no per_lf_resolved_location key at all
    }
    restored = PipelineState.from_dict(legacy)
    assert restored.per_lf_resolved_location is None  # routes to recompute path


# ── resume-path point provenance ───────────────────────────────────


def test_gps_points_rebuilt_from_gpx_after_roundtrip(tmp_path):
    """``gps_per_episode`` is NOT persisted into the
    state JSON — on resume it's rebuilt from the on-disk ``.gpx`` paths in
    ``gpx_files``, and ``_build_gps_points_by_lf`` still recovers a non-empty
    per-LF trace so the voting step can vote, not bare-mean."""
    from gomoveshift.pipeline.gps_enrichment import (
        _build_gps_points_by_lf,
        rebuild_gps_per_episode,
    )

    gpx = tmp_path / "ep0.gpx"
    gpx.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">\n'
        "  <trk><trkseg>\n"
        '    <trkpt lat="32.40" lon="-80.70"><time>2026-04-01T12:00:00Z</time></trkpt>\n'
        '    <trkpt lat="32.41" lon="-80.71"><time>2026-04-01T12:00:25Z</time></trkpt>\n'
        '    <trkpt lat="32.42" lon="-80.72"><time>2026-04-01T12:00:38Z</time></trkpt>\n'
        "  </trkseg></trk>\n"
        "</gpx>\n"
    )

    drive_plan = _make_drive_plan()
    # Fresh-run shape: phase_a would have populated gps_per_episode in memory.
    fresh = rebuild_gps_per_episode([gpx])
    state = PipelineState()
    state.drive_plan = drive_plan
    state.gpx_files = [gpx]
    state.gps_per_episode = fresh

    # Sanity: points reconstructable from the LIVE state.
    live_points = _build_gps_points_by_lf(state.drive_plan, state.gps_per_episode)
    assert live_points.get(1)

    # Round-trip the state to JSON and back — drops gps_per_episode but
    # keeps gpx_files, the recovery source.
    d = state.to_dict()
    assert "gps_per_episode" not in d
    restored = PipelineState.from_dict(d)
    assert restored.gps_per_episode is None
    assert restored.gpx_files, "gpx paths must survive so resume can rebuild"

    rebuilt = _build_gps_points_by_lf(
        restored.drive_plan, rebuild_gps_per_episode(restored.gpx_files)
    )
    assert rebuilt.get(1), "resume must rebuild a non-empty per-LF GPS trace"
    assert len(rebuilt[1]) == 3


def test_resume_gpx_reparse_failure_hard_fails_when_episode_had_gps(tmp_path):
    """On resume a ``.gpx`` for an episode that HAD GPS at ingest and no
    longer re-parses is a derivation FAILURE, not a GPS-less episode. Degrading
    it to ``(None, 0.0)`` would make a GPS trace look GPS-less and silently SKIP
    the pre-encode far-road ``location_invariant`` gate — ``rebuild_gps_per_episode`` must
    HARD FAIL when the episode's index is in ``had_gps_indices``."""
    import pytest

    from gomoveshift.pipeline.gps_enrichment import rebuild_gps_per_episode

    missing = tmp_path / "vanished.gpx"  # recorded path but the file is gone
    with pytest.raises(RuntimeError, match="GPX re-parse"):
        rebuild_gps_per_episode([missing], had_gps_indices={0})


def test_resume_gpx_reparse_failure_tolerated_without_ingest_gps(tmp_path):
    """The narrowing that fixes the over-broad regression: a re-parse failure for
    an episode with NO recorded ingest GPS (index absent from
    ``had_gps_indices``) degrades to ``(None, 0.0)`` — never a hard fail. This is
    the contract missing/None-path fixtures encode."""
    from gomoveshift.pipeline.gps_enrichment import rebuild_gps_per_episode

    missing = tmp_path / "vanished.gpx"
    assert rebuild_gps_per_episode([None, missing]) == [(None, 0.0), (None, 0.0)]
    # Even with evidence for a DIFFERENT episode, the failing one is tolerated.
    assert rebuild_gps_per_episode([missing], had_gps_indices={5}) == [(None, 0.0)]


def test_episodes_with_ingest_gps_maps_resolved_model_to_gpx_by_date():
    """The wiring that makes the gate genuinely fire: an LF with a real resolved
    model on a given date marks that date's episode(s) as had-GPS; a run with no
    resolved models yields the empty set (so GPS-less/legacy resumes tolerate)."""
    from types import SimpleNamespace

    from gomoveshift.pipeline.gps_enrichment import episodes_with_ingest_gps

    resolved = {1: ResolvedLocation(dominant_county="Beaufort County")}
    state = SimpleNamespace(
        per_lf_resolved_location=resolved,
        per_lf_recording_date={1: "2026-03-25"},
        episode_dates=["2026-03-24", "2026-03-25"],
    )
    assert episodes_with_ingest_gps(state) == {1}
    # No resolved models → nothing is protected → tolerate everything.
    assert episodes_with_ingest_gps(SimpleNamespace()) == set()


def test_update_does_not_clobber_saved_true_with_default_false_bool():
    """``PipelineState.update(other)`` (PipelineState branch) must not let a
    freshly-constructed ``other`` (default ``skip_upload=False``) overwrite a
    saved ``True`` on ``self``. The old ``val is not None`` guard skipped only
    ``None``, so a default-``False`` bool clobbered a meaningful ``True``."""
    saved = PipelineState()
    object.__setattr__(saved, "skip_upload", True)

    fresh = PipelineState()  # skip_upload defaults to False
    assert fresh.skip_upload is False

    saved.update(fresh)
    assert saved.skip_upload is True, "default-False bool must not clobber saved True"


def test_update_applies_explicit_non_default_value():
    """The merge still applies a non-default value (regression guard)."""
    target = PipelineState()
    other = PipelineState()
    object.__setattr__(other, "skip_upload", True)

    target.update(other)
    assert target.skip_upload is True
