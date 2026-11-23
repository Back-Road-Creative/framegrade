"""Tests for OST audit sidecar formatter + writer.

The operator must see a per-day audit artefact listing every LF's OST
text, YouTube title, thumbnail-text source, and recording date BEFORE
any OST burn-in re-encode commits pixels.

``format_ost_audit`` is a pure formatter; ``write_ost_audit`` writes the
JSON sidecar and returns the path. Mirrors ``endscreen_sidecar``'s
shape (pure formatter + writer separated from gate logic).
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

from gomoveshift.pipeline import ost_audit_sidecar as sidecar
from gomoveshift.pipeline.encoding_constants import SHORTS_WIDTH
from gomoveshift.pipeline.ost_audit_sidecar import (
    compute_ost_audit_hash,
    format_ost_audit,
    write_ost_audit,
)

from _state_helpers import make_state


def test_format_ost_audit_full_data_single_lf():
    state = make_state(
        per_lf_ost_text={1: "Glenfable Peninsula toward Gortmore, County Kerry"},
        per_lf_titles={
            1: "Highlights from 30min — Glenfable Peninsula toward Gortmore, County Kerry"
        },
        per_lf_recording_date={1: "2025-09-14"},
        per_lf_locations={1: "County Kerry"},
    )
    payload = format_ost_audit("2025-09-14", state, [1])
    assert payload["date"] == "2025-09-14"
    assert "generated_at" in payload
    assert len(payload["lfs"]) == 1
    lf = payload["lfs"][0]
    assert lf["lf_index"] == 1
    assert lf["ost_text"] == "Glenfable Peninsula toward Gortmore, County Kerry"
    assert (
        lf["youtube_title"]
        == "Highlights from 30min — Glenfable Peninsula toward Gortmore, County Kerry"
    )
    # thumbnail_text mirrors ost_text by construction (single source of truth).
    assert lf["thumbnail_text"] == "Glenfable Peninsula toward Gortmore, County Kerry"
    assert lf["recording_date"] == "2025-09-14"


def test_format_ost_audit_missing_thumbnail_proxy_falls_back_to_per_lf_locations():
    """When per_lf_ost_text lacks an entry, per_lf_locations is the fallback
    for both ost_text and thumbnail_text — same priority cascade as the
    OST burn path in run_day_pipeline."""
    state = make_state(
        per_lf_ost_text={},  # canonical missing
        per_lf_titles={2: "Coastal drive — County Clare"},
        per_lf_recording_date={2: "2025-09-15"},
        per_lf_locations={2: "County Clare"},
    )
    payload = format_ost_audit("2025-09-15", state, [2])
    lf = payload["lfs"][0]
    assert lf["lf_index"] == 2
    assert lf["ost_text"] == "County Clare"
    assert lf["thumbnail_text"] == "County Clare"
    assert lf["youtube_title"] == "Coastal drive — County Clare"
    assert lf["recording_date"] == "2025-09-15"


def test_format_ost_audit_multi_lf_day_ordered_by_index():
    state = make_state(
        per_lf_ost_text={
            1: "Ballymock to Glenfable via N86, County Kerry",
            2: "Glenfable to Tralee via R559, County Kerry",
            3: "Tralee to Limerick via N69, County Kerry",
        },
        per_lf_titles={
            1: "Driving — Ballymock to Glenfable",
            2: "Driving — Glenfable to Tralee",
            3: "Driving — Tralee to Limerick",
        },
        per_lf_recording_date={1: "2025-09-14", 2: "2025-09-14", 3: "2025-09-14"},
        per_lf_locations={},
    )
    payload = format_ost_audit("2025-09-14", state, [3, 1, 2])
    indices = [lf["lf_index"] for lf in payload["lfs"]]
    assert indices == [1, 2, 3]
    assert payload["lfs"][0]["ost_text"].startswith("Ballymock")
    assert payload["lfs"][2]["ost_text"].startswith("Tralee")


def test_write_ost_audit_creates_parent_dir_and_returns_path(tmp_path):
    output_base = tmp_path / "out" / "project"
    payload = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "x",
                "youtube_title": "x",
                "thumbnail_text": "x",
                "recording_date": "2025-09-14",
            }
        ],
    }
    path = write_ost_audit(output_base, "2025-09-14", payload)
    assert path == output_base / "ost_audit_2025-09-14.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["date"] == "2025-09-14"
    assert loaded["lfs"][0]["lf_index"] == 1


def test_compute_ost_audit_hash_is_stable_across_generated_at_changes():
    """Hash must reflect the LF content, not the timestamp — otherwise
    a re-prompt fires on every gate entry even when nothing changed."""
    p1 = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "A",
                "youtube_title": "T",
                "thumbnail_text": "A",
                "recording_date": "2025-09-14",
            }
        ],
    }
    p2 = dict(p1, generated_at="2026-05-10T15:00:00Z")
    assert compute_ost_audit_hash(p1) == compute_ost_audit_hash(p2)


def test_compute_ost_audit_hash_changes_when_ost_text_changes():
    p1 = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "A",
                "youtube_title": "T",
                "thumbnail_text": "A",
                "recording_date": "2025-09-14",
            }
        ],
    }
    p2 = {
        "date": "2025-09-14",
        "generated_at": "2026-05-10T14:00:00Z",
        "lfs": [
            {
                "lf_index": 1,
                "ost_text": "B",
                "youtube_title": "T",
                "thumbnail_text": "B",
                "recording_date": "2025-09-14",
            }
        ],
    }
    assert compute_ost_audit_hash(p1) != compute_ost_audit_hash(p2)


def test_the_postburn_proof_reads_frames_at_native_shorts_width(tmp_path, monkeypatch):
    """The post-burn check counts glyph strokes, so it must keep reading frames at
    the shorts' own width — a downscale smears the very strokes it counts. Pinned
    because the sampler is shared with ``blur_gate``, whose spot-check samples at a
    far smaller width."""
    short = tmp_path / "short_01.mp4"
    short.write_bytes(b"x")
    cmds = []

    def fake_run(cmd, **kwargs):
        cmds.append(cmd)
        return SimpleNamespace(returncode=0, stdout=b"\x80" * (SHORTS_WIDTH * 16))

    monkeypatch.setattr(subprocess, "run", fake_run)
    present, why = sidecar.ost_text_present_in_pixels(short)

    assert len(cmds) == 3  # one read per sample time inside the title's plateau
    assert f"scale={SHORTS_WIDTH}:-2,format=gray" in cmds[0]
    # Flat mid-grey carries neither near-white glyphs nor their black border.
    assert present is False and "no burned text" in why


def test_shorts_audit_carries_score_fields_and_hash_ignores_them():
    from gomoveshift.pipeline.ost_audit_sidecar import (
        compute_shorts_ost_audit_hash,
        format_shorts_ost_audit,
    )

    base = {
        "rank": 1,
        "parent_lf_index": 1,
        "ost_text": "Somewhere",
        "start_sec": 10.0,
        "end_sec": 40.0,
        "regions": [],
    }
    scored = dict(base, window_mean=0.6, opening_mean=0.7, quality_bar=0.4, moment_kind="peak")
    plain = format_shorts_ost_audit("2025-09-14", [base])
    rich = format_shorts_ost_audit("2025-09-14", [scored])
    entry = rich["shorts"][0]
    assert entry["window_mean"] == 0.6
    assert entry["opening_mean"] == 0.7
    assert entry["quality_bar"] == 0.4
    assert entry["moment_kind"] == "peak"
    legacy = plain["shorts"][0]
    assert legacy["window_mean"] is None and legacy["moment_kind"] is None
    assert compute_shorts_ost_audit_hash(rich) == compute_shorts_ost_audit_hash(plain)


# ── a stale day re-confirms the CURRENT payload ─────────────────────────────

_STALE_DATE = "2025-09-16"


def _stale_state(title="Ireland (Sep 16, 2025)", stored=None, cli=None):
    return {
        "per_lf_ost_text": {1: "Ireland"},
        "per_lf_titles": {1: title},
        "per_lf_recording_date": {1: _STALE_DATE},
        "per_lf_locations": {},
        "ost_audit_passed": {_STALE_DATE: True} if stored else {},
        "ost_audit_hash": {_STALE_DATE: stored} if stored else {},
        "ost_audit_confirmed_this_run": cli,
    }


def _old_hash():
    """The hash confirmed for an OLDER base label ("Eire"): a real change, so the day is stale."""
    old = _stale_state(title="Eire")
    return compute_ost_audit_hash(format_ost_audit(_STALE_DATE, old, [1]))


def _undated_title_hash():
    """The hash confirmed for the same label before its drive date was added: not a change."""
    old = _stale_state(title="Ireland")
    return compute_ost_audit_hash(format_ost_audit(_STALE_DATE, old, [1]))


def _current_hash():
    return compute_ost_audit_hash(format_ost_audit(_STALE_DATE, _stale_state(), [1]))


def test_stale_day_is_reconfirmed_by_the_flag_and_the_hash_is_rebound(tmp_path):
    import pytest

    state = _stale_state(stored=_old_hash(), cli=True)
    sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)
    assert state["ost_audit_hash"][_STALE_DATE] == _current_hash() != _old_hash()
    state["ost_audit_confirmed_this_run"] = None  # the next run, no flag
    sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)
    # ...and a text change after the re-bind is stale again (the flag is not a blanket pass)
    state["per_lf_titles"] = {1: "Ireland (Sep 17, 2025)"}
    with pytest.raises(sidecar.OstAuditPendingError):
        sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)


def test_stale_day_with_no_route_refuses_unchanged_and_names_only_working_routes(tmp_path):
    import pytest

    state = _stale_state(stored=_old_hash())
    with pytest.raises(sidecar.OstAuditPendingError) as exc:
        sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)
    msg = str(exc.value)
    assert state["ost_audit_hash"][_STALE_DATE] == _old_hash()
    assert "--ost-audit-confirmed" in msg
    assert _current_hash() in msg  # the hash a marker must carry to pass
    assert "create " not in msg  # a bare hand-made marker does not pass a stale day
    assert "system review" not in msg  # that route does not re-confirm a stale day


def test_a_stale_day_never_passes_on_its_own(tmp_path):
    import pytest

    for stored_passed in (True, False):
        state = _stale_state(stored=_old_hash())
        state["ost_audit_passed"] = {_STALE_DATE: stored_passed}
        with pytest.raises(sidecar.OstAuditPendingError):
            sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)


def test_a_bare_confirmed_marker_does_not_pass_a_stale_day(tmp_path):
    import pytest

    (tmp_path / f"ost_audit_{_STALE_DATE}.confirmed").write_text("ok")
    state = _stale_state(stored=_old_hash())
    with pytest.raises(sidecar.OstAuditPendingError):
        sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)


def test_a_marker_stamped_with_the_current_hash_passes_a_stale_day_and_rebinds(tmp_path):
    marker = tmp_path / f"ost_audit_{_STALE_DATE}.confirmed"
    marker.write_text(_current_hash())
    state = _stale_state(stored=_old_hash())
    sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)
    assert state["ost_audit_hash"][_STALE_DATE] == _current_hash()


def test_an_unchanged_payload_still_passes_without_a_flag(tmp_path):
    state = _stale_state(title="Ireland", stored=_undated_title_hash())
    sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)


def test_a_system_review_does_not_reconfirm_a_stale_day(tmp_path):
    import pytest

    from gomoveshift.pipeline import ost_system_review as osr
    from gomoveshift.pipeline.route_claims import route_claim_outcome

    state = _stale_state(stored=_old_hash())
    state["ost_system_review_this_run"] = True
    outcomes = {1: route_claim_outcome(1, "Ireland", None, chosen="Ireland", reason="")}
    osr.record_system_reviews(state, {_STALE_DATE: {1: "Ireland"}}, outcomes)
    with pytest.raises(sidecar.OstAuditPendingError):
        sidecar.lf_gate_or_raise(tmp_path, _STALE_DATE, [1], state)
    assert state["ost_audit_hash"][_STALE_DATE] == _old_hash()
