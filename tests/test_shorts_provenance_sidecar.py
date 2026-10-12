"""Shorts carry the same cut-from-master identity long-forms do.

`write_lf_provenance_sidecar` proved, per long-form, which day master the cut
came from and whether that master's own privacy attestation agrees, and
`attach_master_identity_to_record` exposed that comparison so thumbnail upload
entries could reuse it rather than grow a drifting copy. Shorts were left out --
so the surface with the *largest* audience per file, and the one that ships to
four platforms through a CSV, had no way to say which master it came from.

`write_shorts_provenance_sidecar` closes that, through the same helper.

**Records; never gates.** Whether a mismatch should HOLD a delivery is the operator's
decision (the same one the LF sidecar deliberately leaves open). Nothing here
can fail a run: every absent input is null-WITH-REASON, never "verified".

One thing is deliberately NOT inherited from the LF: a short's day comes from
its OWN path (`.../day_<DATE>/shorts/short_<rank>.mp4`), not from its LF's
recording date. A day-first LF is assembled from one calendar day, but the
short's path is the direct evidence, and preferring an indirect lookup over
direct evidence is how a far-corner short ends up attributed to the wrong
master.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from gomoveshift.pipeline.lf_provenance_sidecar import (
    SHORTS_PROVENANCE_FILENAME,
    build_shorts_provenance_payload,
    write_shorts_provenance_sidecar,
)

DATE = "2026-02-24"
OTHER_DATE = "2026-02-25"


def _write_attestation(stages_base, date, master_fingerprint=None):
    day_dir = stages_base / f"day_{date}"
    day_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "gms.privacy_attestation/v1", "date": date}
    if master_fingerprint is not None:
        payload["master_fingerprint"] = master_fingerprint
    (day_dir / f"day_{date}_privacy_attestation.json").write_text(json.dumps(payload))


def _short(stages_base, date, rank):
    return str(stages_base / f"day_{date}" / "shorts" / f"short_{rank:02d}.mp4")


def _state(stages_base, shorts_per_lf, *, per_day_completed=None, recording_dates=None):
    return SimpleNamespace(
        stages_base=str(stages_base),
        shorts_per_lf=shorts_per_lf,
        per_day_completed=per_day_completed or {},
        per_lf_recording_date=recording_dates or {},
    )


def test_every_short_carries_the_master_it_was_cut_from(tmp_path):
    stages = tmp_path / "_stages"
    _write_attestation(stages, DATE, master_fingerprint="123:456")
    state = _state(
        stages,
        {0: [_short(stages, DATE, 1), _short(stages, DATE, 2)]},
        per_day_completed={DATE: {"master_fingerprint": "123:456"}},
    )

    shorts = build_shorts_provenance_payload(state)["shorts"]
    assert [s["rank"] for s in shorts] == [1, 2]
    for entry in shorts:
        assert entry["lf_index"] == 0
        assert entry["day"] == DATE
        assert entry["cut_from_master_fingerprint"] == "123:456"


def test_a_mismatch_is_recorded_with_its_reason_and_never_raises(tmp_path):
    stages = tmp_path / "_stages"
    _write_attestation(stages, DATE, master_fingerprint="999:999")
    state = _state(
        stages,
        {0: [_short(stages, DATE, 1)]},
        per_day_completed={DATE: {"master_fingerprint": "123:456"}},
    )

    entry = write_and_read(tmp_path, state)["shorts"][0]
    assert entry["master_attestation_verified"] is False
    assert entry["null_reasons"]["master_attestation_verified"]


def test_a_matching_fingerprint_verifies_true(tmp_path):
    stages = tmp_path / "_stages"
    _write_attestation(stages, DATE, master_fingerprint="123:456")
    state = _state(
        stages,
        {0: [_short(stages, DATE, 1)]},
        per_day_completed={DATE: {"master_fingerprint": "123:456"}},
    )

    entry = write_and_read(tmp_path, state)["shorts"][0]
    assert entry["master_attestation_verified"] is True
    assert "master_attestation_verified" not in entry["null_reasons"]


def test_a_shorts_day_comes_from_its_own_path_not_its_lfs_recording_date(tmp_path):
    """The short's path is direct evidence; the LF's recording date is an
    indirect lookup. Preferring the lookup is how a short gets attributed to
    the wrong day's master."""
    stages = tmp_path / "_stages"
    _write_attestation(stages, OTHER_DATE, master_fingerprint="other:master")
    state = _state(
        stages,
        {0: [_short(stages, OTHER_DATE, 1)]},
        per_day_completed={
            DATE: {"master_fingerprint": "123:456"},
            OTHER_DATE: {"master_fingerprint": "other:master"},
        },
        recording_dates={0: DATE},
    )

    entry = build_shorts_provenance_payload(state)["shorts"][0]
    assert entry["day"] == OTHER_DATE
    assert entry["cut_from_master_fingerprint"] == "other:master"


def test_an_unparsable_path_falls_back_to_the_lf_date_and_says_so(tmp_path):
    stages = tmp_path / "_stages"
    _write_attestation(stages, DATE, master_fingerprint="123:456")
    state = _state(
        stages,
        {0: [str(tmp_path / "somewhere" / "clip.mp4")]},
        per_day_completed={DATE: {"master_fingerprint": "123:456"}},
        recording_dates={0: DATE},
    )

    entry = build_shorts_provenance_payload(state)["shorts"][0]
    assert entry["day"] == DATE
    assert entry["rank"] is None
    assert "day" in entry["null_reasons"]


def test_no_day_anywhere_stays_null_with_a_reason_never_verified(tmp_path):
    stages = tmp_path / "_stages"
    state = _state(stages, {0: [str(tmp_path / "clip.mp4")]})

    entry = build_shorts_provenance_payload(state)["shorts"][0]
    assert entry["day"] is None
    assert entry["master_attestation_verified"] is None
    assert entry["null_reasons"]["day"]


def test_a_missing_attestation_is_not_a_pass(tmp_path):
    stages = tmp_path / "_stages"
    state = _state(
        stages,
        {0: [_short(stages, DATE, 1)]},
        per_day_completed={DATE: {"master_fingerprint": "123:456"}},
    )

    entry = build_shorts_provenance_payload(state)["shorts"][0]
    assert entry["master_attestation_verified"] is None
    assert entry["null_reasons"]["master_attestation_verified"]


def test_no_shorts_writes_nothing_rather_than_an_empty_claim(tmp_path):
    state = _state(tmp_path / "_stages", {})
    assert write_shorts_provenance_sidecar(tmp_path, state) is None
    assert not (tmp_path / SHORTS_PROVENANCE_FILENAME).exists()


def test_the_sidecar_lands_beside_the_lf_one_and_round_trips(tmp_path):
    stages = tmp_path / "_stages"
    _write_attestation(stages, DATE, master_fingerprint="123:456")
    state = _state(
        stages,
        {1: [_short(stages, DATE, 3)]},
        per_day_completed={DATE: {"master_fingerprint": "123:456"}},
    )

    path = write_shorts_provenance_sidecar(tmp_path, state)
    assert path == tmp_path / SHORTS_PROVENANCE_FILENAME
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == 1
    assert doc["shorts"][0]["lf_index"] == 1
    assert doc["shorts"][0]["rank"] == 3


def test_a_write_failure_returns_none_rather_than_failing_the_run(tmp_path, monkeypatch):
    """The day master is already encoded and the shorts already exist; a record
    that can abort the run is a record that decides an outcome."""
    stages = tmp_path / "_stages"
    state = _state(stages, {0: [_short(stages, DATE, 1)]})

    from gomoveshift.pipeline import lf_provenance_sidecar as mod

    def _boom(self, *a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(mod.Path, "write_text", _boom)
    assert write_shorts_provenance_sidecar(tmp_path, state) is None


def write_and_read(tmp_path, state) -> dict:
    path = write_shorts_provenance_sidecar(tmp_path, state)
    assert path is not None
    return json.loads(path.read_text())
