"""Audit enrichment: per-SHORT ``median_speed_mps`` (and no ``measured_lufs``).

A human reviewer should see "this window is parked" before the irreversible
burn, so each short audit entry carries the clip's GPS ground speed — a
DIAGNOSTIC annotation excluded from the confirmation hash so a measured-value
readout never reopens the pre-burn gate (same rationale as excluding
``generated_at``).

Neither the per-SHORT nor the per-LF payload carries ``measured_lufs``:
nothing anywhere measures loudness for either surface (shorts ship muted by
policy), so the key could only ever emit ``None`` — which reads to an operator
as "measured, and it was nothing".
"""

from __future__ import annotations

from gomoveshift.pipeline.ost_audit_sidecar import (
    compute_shorts_ost_audit_hash,
    format_ost_audit,
    format_shorts_ost_audit,
)


def _short_entry(**overrides):
    entry = {
        "rank": 1,
        "parent_lf_index": 1,
        "ost_text": "County Kerry",
        "start_sec": 10.0,
        "end_sec": 48.0,
        "regions": ["County Kerry"],
    }
    entry.update(overrides)
    return entry


def test_shorts_audit_carries_enrichment_fields():
    payload = format_shorts_ost_audit(
        "2025-09-14", [_short_entry(median_speed_mps=11.2, measured_lufs=-19.5)]
    )
    short = payload["shorts"][0]
    assert short["median_speed_mps"] == 11.2
    # Never measured for a short (they ship muted): not emitted even if supplied.
    assert "measured_lufs" not in short


def test_shorts_audit_missing_enrichment_defaults_none():
    payload = format_shorts_ost_audit("2025-09-14", [_short_entry()])
    short = payload["shorts"][0]
    assert short["median_speed_mps"] is None
    assert "measured_lufs" not in short


def test_shorts_audit_hash_ignores_enrichment_fields():
    slow = format_shorts_ost_audit("2025-09-14", [_short_entry(median_speed_mps=0.1)])
    fast = format_shorts_ost_audit("2025-09-14", [_short_entry(median_speed_mps=25.0)])
    # A different measured speed must NOT change the burn-text confirmation hash.
    assert compute_shorts_ost_audit_hash(slow) == compute_shorts_ost_audit_hash(fast)


def test_dropping_measured_lufs_leaves_the_confirmation_hash_unchanged():
    """No pending OST confirmation reopens when the key is removed.

    ``measured_lufs`` was already in ``_HASH_EXCLUDED_ENTRY_KEYS``, so it never
    reached the hash input — the digest below is pinned from the payload as it
    was built BEFORE the key was dropped. An operator's outstanding
    ``shorts_ost_audit_<date>.confirmed`` therefore still matches.
    """
    before_removal = "c12c93a44160f585b0ca8cf7fbfb128d3be89e8f6826318e38513b36340b33e3"  # pragma: allowlist secret
    payload = format_shorts_ost_audit("2025-09-14", [_short_entry(median_speed_mps=11.2)])
    assert "measured_lufs" not in payload["shorts"][0]
    assert compute_shorts_ost_audit_hash(payload) == before_removal


def test_lf_audit_omits_never_measured_enrichment_fields():
    """The per-LF payload must not advertise speed/loudness it never measures.

    ``PipelineState`` declares no ``per_lf_median_speed_mps`` /
    ``per_lf_measured_lufs``, nothing anywhere writes them, and
    ``lf_gate_or_raise`` assembles its audit state without them — so the keys
    could only ever emit ``None``. A key that is structurally always ``None``
    reads to an operator as "measured, and it was nothing".
    """
    state = {
        "per_lf_ost_text": {1: "County Kerry"},
        "per_lf_titles": {1: "T"},
        "per_lf_recording_date": {1: "2025-09-14"},
        "per_lf_median_speed_mps": {1: 12.0},
        "per_lf_measured_lufs": {1: -19.5},
    }
    entry = format_ost_audit("2025-09-14", state, [1])["lfs"][0]
    assert "median_speed_mps" not in entry
    assert "measured_lufs" not in entry
    # Nothing else in the reviewed payload shifted.
    assert entry == {
        "lf_index": 1,
        "ost_text": "County Kerry",
        "youtube_title": "T",
        "thumbnail_text": "County Kerry",
        "recording_date": "2025-09-14",
    }
