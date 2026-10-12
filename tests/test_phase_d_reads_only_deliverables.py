"""Phase D reads rendered paths and measurements from ``Deliverables``."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_phases import phase_bc_day_first as bc
from gomoveshift.pipeline.drive_phases import phase_d, phase_d_gates
from gomoveshift.pipeline.media_api import (
    Deliverables,
    Measurements,
    deliverables_for_publish,
    publish_lf_paths,
)
from gomoveshift.pipeline.short_planner import ShortRange

PHASE_D_DIR = Path(phase_d.__file__).parent


def _deliverables(root: Path, date: str, lf_indices: list[int]) -> Deliverables:
    day = root / f"day_{date}"
    (day / "regular_lfs").mkdir(parents=True)
    (day / "shorts").mkdir()
    lfs = [day / "regular_lfs" / f"lf_{i:02d}.mp4" for i in lf_indices]
    shorts = [day / "shorts" / f"short_{n:02d}.mp4" for n in (1, 2)]
    for p in (*lfs, *shorts):
        p.write_bytes(b"x")
    (day / f"day_{date}_master.mp4").write_bytes(b"x")
    sidecar = day / f"ost_audit_{date}.json"
    sidecar.write_text("{}")
    return Deliverables(
        date=date,
        master_path=day / f"day_{date}_master.mp4",
        regular_lf_paths=lfs,
        regular_lf_indices=lf_indices,
        short_paths=shorts,
        short_ranges=[ShortRange(1.0, 9.0, 0.5, 1, "peak"), ShortRange(20.0, 31.0, 0.4, 2)],
        master_record={"master_duration": 12.5},
        planned_shorts_count=2,
        shorts_target_in_force=1,
        sidecars={"ost_audit": sidecar},
        measurements=Measurements(sharpen_amount={str(lfs[0]): 0.3}),
    )


def _state(tmp_path: Path, days: list[Deliverables]) -> SimpleNamespace:
    state = SimpleNamespace(
        stages_base=str(tmp_path), output_base=str(tmp_path), per_day_completed={}
    )
    for d in days:
        bc._record_day_and_save(d, state, None)
    return state


@pytest.fixture
def two_days(tmp_path):
    # recorded out of order on purpose
    later = _deliverables(tmp_path, "2026-05-02", [3, 4])
    earlier = _deliverables(tmp_path, "2026-05-01", [1, 2])
    return tmp_path, earlier, later, _state(tmp_path, [later, earlier])


class _NoFlatReads(SimpleNamespace):
    @property
    def regular_lf_paths(self):
        raise AssertionError("Phase D read state.regular_lf_paths")


def test_phase_d_reads_only_deliverables(two_days):
    tmp_path, earlier, later, state = two_days
    got = deliverables_for_publish(state)
    assert [d.date for d in got] == ["2026-05-01", "2026-05-02"]
    assert got == [earlier, later]

    guarded = _NoFlatReads(**vars(state), skip_shorts_quality_gate=False, shorts_per_lf={})
    # gate: empty shorts_per_lf while LFs exist is refused, via Deliverables
    with pytest.raises(RuntimeError, match="shorts_per_lf is empty"):
        phase_d_gates._assert_shorts_quality_gate(guarded, validator_cls=object)
    # reconcile scopes to the Deliverables' dates: both days' 2 shorts seen
    with pytest.raises(RuntimeError, match="4 exist on disk"):
        phase_d._reconcile_shorts_per_lf_against_disk(guarded)


def test_resume_rebuilds_deliverables_from_persisted_record(two_days):
    import json

    _, earlier, later, state = two_days
    reloaded = SimpleNamespace(**json.loads(json.dumps(vars(state), default=str)))
    rebuilt = deliverables_for_publish(reloaded)
    assert len(rebuilt) == 2
    for want, have in zip([earlier, later], rebuilt):
        assert have == want
        assert have.regular_lf_indices == want.regular_lf_indices
        assert have.measurements == want.measurements
        assert have.sidecars == want.sidecars
        assert have.short_ranges == want.short_ranges


def test_order_follows_day_artifacts_and_flat_list_matches_pre_change(tmp_path):
    later = _deliverables(tmp_path, "2026-05-02", [3, 4])
    earlier = _deliverables(tmp_path, "2026-05-01", [1, 2])
    state = _state(tmp_path, [later, earlier])
    state.day_artifacts = [later, earlier]  # skipped/fresh order, not date order
    assert [d.date for d in deliverables_for_publish(state)] == ["2026-05-02", "2026-05-01"]
    # what Phase B/C stored before: LF paths sorted by LF index
    pre_change = [str(p) for p in (*earlier.regular_lf_paths, *later.regular_lf_paths)]
    assert publish_lf_paths(state) == pre_change
    state.day_artifacts = [earlier]  # unlisted dates follow, in date order
    assert [d.date for d in deliverables_for_publish(state)] == ["2026-05-01", "2026-05-02"]


def test_legacy_state_without_records_falls_back_to_lf_paths(tmp_path):
    d = _deliverables(tmp_path, "2026-05-01", [1, 2])
    state = SimpleNamespace(
        per_day_completed={}, regular_lf_paths=[str(p) for p in d.regular_lf_paths]
    )
    got = deliverables_for_publish(state)
    assert [x.date for x in got] == ["2026-05-01"]
    assert got[0].regular_lf_indices == [1, 2]


@pytest.mark.parametrize("name", ["phase_d.py", "phase_d_gates.py"])
def test_phase_d_has_no_state_attribute_read_of_rendered_paths(name):
    tree = ast.parse((PHASE_D_DIR / name).read_text())
    bad = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"regular_lf_paths", "short_paths"}
        and isinstance(node.value, ast.Name)
        and node.value.id == "state"
    ]
    assert bad == []
