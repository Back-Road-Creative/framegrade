"""End-to-end seam proof: Phase A evidence -> strict gate, with NOTHING stubbed.

Every other sign-verify test isolates one half. The wiring tests stub the
comparator; the comparator tests hand-build their own hits. That leaves the
join untested — and the join is where this feature fails SILENTLY rather than
loudly: if the hit's index space and the plan segment's index space disagree,
``bucket_hits_by_lf`` matches nothing, every verdict is "no-signal", and the
gate reports a clean bill of health precisely because it can no longer see
anything. A gate that never fires looks identical to a gate with nothing to
report.

So these tests run the REAL ``sign_pass`` -> ``sign_evidence`` -> ``sign_verify``
-> ``_assert_sign_verify`` chain and assert the gate actually raises on a
planted contradiction. The only patched seam is the frame grab / OCR pair (no
video on disk) and the plan builder (no scoring arrays) — the index spaces the
join depends on are real on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from gomoveshift.analysis.sign_evidence import SignClaim, SignHit, hits_by_date_to_dicts
from gomoveshift.analysis.sign_verify import bucket_hits_by_lf, verify_all
from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline.drive_phases import phase_bc_day_first


@dataclass
class _Seg:
    """Minimal stand-in for an LFSegment: the three fields bucketing reads."""

    source_file_index: int
    start_time: float
    end_time: float


@dataclass
class _LF:
    index: int
    segments: list


@dataclass
class _Plan:
    regular_lfs: list


class _Drive:
    """``config.drive`` stub carrying only the sign-verify knobs."""

    def __init__(self, strict: bool = True, floor: int = 2) -> None:
        self.sign_verify = True
        self.sign_verify_strict = strict
        self.sign_verify_min_corroborating_hits = floor


class _Config:
    def __init__(self, **kw) -> None:
        self.drive = _Drive(**kw)


class _State:
    def __init__(self, hits_by_date, resolved_by_lf) -> None:
        self.sign_hits_by_date = hits_by_date
        self.per_lf_resolved_location = resolved_by_lf


def _hit(episode_index: int, t_local: float, county: str, ts_day: Optional[float] = None):
    """A real SignHit carrying a real county claim, in the GLOBAL index space."""
    text = f"{county} County"
    return SignHit(
        episode_index=episode_index,
        t_local=t_local,
        ts_day=t_local if ts_day is None else ts_day,
        date="2026-07-20",
        raw_text=text,
        ocr_conf=0.9,
        claims=[SignClaim(kind="county_boundary", value=county, raw_text=text)],
    )


# LF 0 owns episode 0 [0, 100]; LF 1 owns episode 1 [0, 100]. Two episodes so a
# hit assigned to the wrong index space lands in the wrong LF (or nowhere).
def _plan() -> _Plan:
    return _Plan(
        regular_lfs=[
            _LF(index=0, segments=[_Seg(0, 0.0, 100.0)]),
            _LF(index=1, segments=[_Seg(1, 0.0, 100.0)]),
        ]
    )


def _resolved_colleton() -> ResolvedLocation:
    return ResolvedLocation(
        dominant_county="Colleton",
        dominant_state="South Carolina",
        regions=["Colleton, South Carolina"],
        vote={"Colleton, South Carolina": 1.0},
    )


def _persisted(hits_by_date) -> dict:
    """Round-trip through the state serializer shape, as Phase A persists it."""
    return hits_by_date_to_dicts(hits_by_date)


class TestBucketingSeam:
    """The join Phase A and the comparator each assume about the other."""

    def test_hits_reach_their_own_lf_through_the_state_round_trip(self):
        hits = {"2026-07-20": [_hit(0, 10.0, "Fableport"), _hit(1, 10.0, "Mockston")]}
        buckets = bucket_hits_by_lf(_persisted(hits), _plan())

        assert sorted(buckets) == [0, 1]
        assert [c.value for c in buckets[0][0].parsed_claims()] == ["Fableport"]
        assert [c.value for c in buckets[1][0].parsed_claims()] == ["Mockston"]

    def test_a_day_local_index_space_is_NOT_silently_accepted(self):
        """The regression that would make the gate blind.

        ``_partition_lf_plans_by_date`` remaps segment ``source_file_index``
        from global to day-local IN PLACE. Bucketing against a partitioned plan
        must not quietly succeed: episode 1's hit would land in LF 0 (whose
        segment was remapped to day-local 0), attributing one LF's evidence to
        another. Assert the hit does NOT reach LF 1 — so if anyone ever hands
        this a partitioned plan, a test fails instead of the gate going quiet.
        """
        partitioned = _Plan(regular_lfs=[_LF(index=0, segments=[_Seg(0, 0.0, 100.0)])])
        hits = {"2026-07-20": [_hit(1, 10.0, "Mockston")]}

        buckets = bucket_hits_by_lf(_persisted(hits), partitioned)

        assert buckets.get(1) is None
        assert not buckets.get(0), "episode-1 evidence must not be credited to LF 0"


class TestStrictGateFiresForReal:
    """``_assert_sign_verify`` with the real comparator underneath."""

    @pytest.fixture(autouse=True)
    def _plan_from_state(self, monkeypatch):
        """Skip the scoring arrays; keep the GLOBAL index space real."""
        monkeypatch.setattr(
            phase_bc_day_first,
            "_build_drive_plan_from_state",
            lambda config, state: _plan(),
        )

    def test_two_contradicting_signs_hard_fail_before_any_encode(self):
        hits = {"2026-07-20": [_hit(0, 10.0, "Fableport"), _hit(0, 80.0, "Fableport")]}
        state = _State(_persisted(hits), {0: _resolved_colleton()})

        with pytest.raises(RuntimeError) as exc:
            phase_bc_day_first._assert_sign_verify(state, _Config())

        msg = str(exc.value)
        assert "SIGN-VERIFY-CONFLICT" in msg
        assert "LF 0" in msg
        assert "Fableport" in msg, "the operator must be told WHICH county the signs claimed"
        assert "Colleton" in msg, "...and what the label resolved to"

    def test_one_sign_is_not_enough_to_block_an_encode(self):
        """The conservatism rule, proven at the gate rather than in isolation."""
        hits = {"2026-07-20": [_hit(0, 10.0, "Fableport")]}
        state = _State(_persisted(hits), {0: _resolved_colleton()})

        phase_bc_day_first._assert_sign_verify(state, _Config())

    def test_agreeing_signs_do_not_block(self):
        hits = {"2026-07-20": [_hit(0, 10.0, "Colleton"), _hit(0, 80.0, "Colleton")]}
        state = _State(_persisted(hits), {0: _resolved_colleton()})

        phase_bc_day_first._assert_sign_verify(state, _Config())

    def test_flag_off_never_blocks_even_on_a_real_conflict(self):
        hits = {"2026-07-20": [_hit(0, 10.0, "Fableport"), _hit(0, 80.0, "Fableport")]}
        state = _State(_persisted(hits), {0: _resolved_colleton()})

        phase_bc_day_first._assert_sign_verify(state, _Config(strict=False))

    def test_one_lfs_conflict_does_not_implicate_its_sibling(self):
        """Cross-LF leak check with the real bucketer in the loop."""
        hits = {
            "2026-07-20": [
                _hit(1, 10.0, "Fableport"),
                _hit(1, 80.0, "Fableport"),
                _hit(0, 10.0, "Colleton"),
            ]
        }
        state = _State(_persisted(hits), {0: _resolved_colleton(), 1: _resolved_colleton()})

        with pytest.raises(RuntimeError) as exc:
            phase_bc_day_first._assert_sign_verify(state, _Config())

        msg = str(exc.value)
        assert "LF 1" in msg
        assert "LF 0" not in msg, "LF 0's signs agreed — it must not be named"

    def test_a_gps_less_lf_with_no_model_is_a_no_op(self):
        hits = {"2026-07-20": [_hit(0, 10.0, "Fableport"), _hit(0, 80.0, "Fableport")]}
        state = _State(_persisted(hits), {0: None})

        phase_bc_day_first._assert_sign_verify(state, _Config())


class TestVerifyAllOverThePersistedShape:
    def test_raw_state_dicts_drive_the_same_verdicts_as_typed_hits(self):
        """Persistence must not change a verdict — resume parity."""
        hits = {"2026-07-20": [_hit(0, 10.0, "Fableport"), _hit(0, 80.0, "Fableport")]}
        resolved = {0: _resolved_colleton()}

        typed = verify_all(hits, _plan(), resolved)
        persisted = verify_all(_persisted(hits), _plan(), resolved)

        assert typed[0].status == persisted[0].status
        assert persisted[0].status == "conflicts"
