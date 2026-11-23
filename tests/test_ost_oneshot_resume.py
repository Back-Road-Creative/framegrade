"""A one-shot flag never replays through a resume recipe, and resume accepts every hash the
gate accepts.

* ``--ost-audit-confirmed`` and ``--fresh`` are stripped from a recipe by the DESTINATION the
  real drive parser resolves them to (argparse accepts any unique prefix and ``name=value``).
* The resume drift check and ``lf_gate_or_raise`` share one rule for the hashes a confirmation
  may hold, so a day the gate passes on a legacy hash never stalls the next resume.
"""

from __future__ import annotations

import copy
import json

import pytest

from gomoveshift.cli.parser import _build_parser
from gomoveshift.health.resume import read_recipe, write_recipe
from gomoveshift.pipeline import ost_audit_sidecar as sidecar
from gomoveshift.pipeline import state_invariants as si
from gomoveshift.pipeline.ost_audit_sidecar import compute_ost_audit_hash, format_ost_audit

CONFIRM = "--ost-audit-confirmed"
FRESH = "--fresh"
DAY = "2025-09-20"
COUNTY = "County Cork, Ireland"
LEGACY_SPLIT = {1: f"{COUNTY} (Part 1 of 2)", 2: f"{COUNTY} (Part 2 of 2)"}


def _replayed(tmp_path, argv):
    path = write_recipe(tmp_path, ["drive", "/src", *argv], cwd="/", sha="x", now_iso="t")
    assert path is not None
    return read_recipe(path)["argv"][3:]


def _parse(argv):
    try:
        return _build_parser().parse_known_args(argv)[0]
    except SystemExit:  # an ambiguous prefix is a parse error: nothing to replay
        return None


def _accepted_spellings(flag, dest):
    """Every spelling of ``flag`` the real parser resolves to ``dest`` as given: the exact
    flag, each of its prefixes, and the ``=`` form of each."""
    out = []
    for n in range(3, len(flag) + 1):
        for spelling in (flag[:n], flag[:n] + "=1"):
            args = _parse(["drive", "/src", spelling])
            if args is not None and getattr(args, dest, None) not in (None, False):
                out.append(spelling)
    return out


@pytest.mark.parametrize("flag,dest", [(CONFIRM, "ost_audit_confirmed"), (FRESH, "fresh")])
def test_no_spelling_of_a_one_shot_flag_is_written_or_replayed(tmp_path, flag, dest):
    spellings = _accepted_spellings(flag, dest)
    assert flag in spellings and len(spellings) >= 3  # the exact form and real abbreviations
    for spelling in spellings:
        sub = tmp_path / spelling.replace("-", "_").replace("=", "_eq_")
        path = write_recipe(sub, ["drive", "/src", spelling], cwd="/", sha=None, now_iso="t")
        assert spelling not in json.loads(path.read_text(encoding="utf-8"))["argv"], spelling
        args = _parse(read_recipe(path)["argv"][3:])
        assert not getattr(args, dest), spelling  # replayed argv parses to the flag unset
        # the ``name=value`` form of the same spelling is dropped too
        eq = _replayed(sub / "eq", [spelling + "=1", "--ost-system-review"])
        assert eq == ["drive", "/src", "--ost-system-review"], spelling


@pytest.mark.parametrize("flag,dest", [(CONFIRM, "ost_audit_confirmed"), (FRESH, "fresh")])
def test_a_recipe_already_on_disk_with_an_abbreviated_one_shot_flag_replays_without_it(
    tmp_path, flag, dest
):
    from gomoveshift.health.resume import RECIPE_NAME

    path = tmp_path / RECIPE_NAME
    argv = ["py", "-m", "gomoveshift", "drive", "/src", *_accepted_spellings(flag, dest)]
    path.write_text(json.dumps({"argv": argv, "cwd": "/"}), encoding="utf-8")
    assert read_recipe(path)["argv"] == argv[:5]


def test_an_unrelated_flag_sharing_a_prefix_is_kept(tmp_path):
    kept = _replayed(tmp_path, ["--ost-system-review", "--ost-audit-c", "--fresh"])
    assert kept == ["drive", "/src", "--ost-system-review"]
    assert _parse(kept).ost_system_review is True


# ── resume accepts every hash the gate accepts ─────────────────────────────


def _state(titles, ost=None, stored=None, date=DAY):
    ost = ost or {k: COUNTY for k in titles}
    return {
        "per_lf_ost_text": dict(ost),
        "per_lf_titles": dict(titles),
        "per_lf_recording_date": {k: date for k in ost},
        "per_lf_locations": {},
        "ost_audit_passed": {date: True} if stored else {},
        "ost_audit_hash": {date: stored} if stored else {},
    }


def _legacy_day(tmp_path):
    """A day the DEPLOYED code confirmed: dedup titles, raw hash, part-key-less sidecar."""
    st = _state(LEGACY_SPLIT)
    payload = format_ost_audit(DAY, st, [1, 2])
    stored = compute_ost_audit_hash(payload, normalise=False)
    st.update(
        {
            "ost_audit_passed": {DAY: True},
            "ost_audit_hash": {DAY: stored},
            "output_base": str(tmp_path),
        }
    )
    for e in payload["lfs"]:
        e.pop("part_index", None)
        e.pop("part_count", None)
    (tmp_path / f"ost_audit_{DAY}.json").write_text(json.dumps(payload))
    return st, stored


def test_resume_does_not_call_drift_on_a_day_the_gate_passes_on_its_legacy_hash(tmp_path):
    date = "2025-09-16"
    st = _state({1: "Ireland (Sep 16, 2025)"}, ost={1: "Ireland"}, date=date)
    payload = format_ost_audit(date, st, [1])
    (tmp_path / f"ost_audit_{date}.json").write_text(json.dumps(payload))
    st.update(
        {
            "ost_audit_passed": {date: True},
            "ost_audit_hash": {date: compute_ost_audit_hash(payload, normalise=False)},
            "output_base": str(tmp_path),
        }
    )
    sidecar.lf_gate_or_raise(tmp_path, date, [1], copy.deepcopy(st))
    si._check_ost_audit_hash_drift(st)  # no raise: the gate accepts it, so does resume


def test_two_consecutive_resumes_after_a_legacy_pass_neither_raise_nor_stall(tmp_path):
    st, stored = _legacy_day(tmp_path)
    si._check_ost_audit_hash_drift(copy.deepcopy(st))  # resume 1: invariants
    sidecar.lf_gate_or_raise(tmp_path, DAY, [1, 2], st)  # resume 1: passes, re-stamps
    assert st["ost_audit_hash"][DAY] != stored
    sidecar_now = json.loads((tmp_path / f"ost_audit_{DAY}.json").read_text())
    assert compute_ost_audit_hash(sidecar_now) == st["ost_audit_hash"][DAY]  # disk == stamp
    si._check_ost_audit_hash_drift(copy.deepcopy(st))  # resume 2: invariants, no drift
    sidecar.lf_gate_or_raise(tmp_path, DAY, [1, 2], st)  # resume 2: passes, no pending
    assert st["ost_audit_passed"][DAY] is True


def test_a_legacy_hash_still_refuses_a_changed_burn_on_resume(tmp_path):
    st, _ = _legacy_day(tmp_path)
    st["per_lf_ost_text"] = {1: "County Kerry, Ireland", 2: "County Kerry, Ireland"}
    with pytest.raises(sidecar.OstAuditPendingError):
        sidecar.lf_gate_or_raise(tmp_path, DAY, [1, 2], st)


def test_the_drift_check_still_raises_when_the_stored_hash_matches_no_accepted_form(tmp_path):
    st, _ = _legacy_day(tmp_path)
    st["ost_audit_hash"][DAY] = "0" * 64
    with pytest.raises(Exception, match="drift"):
        si._check_ost_audit_hash_drift(st)


def test_the_drift_message_does_not_tell_an_unattended_run_to_pass_the_flag(tmp_path):
    st, _ = _legacy_day(tmp_path)
    st["ost_audit_hash"][DAY] = "0" * 64
    with pytest.raises(Exception, match="drift") as exc:
        si._check_ost_audit_hash_drift(st)
    assert CONFIRM not in str(exc.value)


# ── the end-of-options separator, ambiguous prefixes, and a refused write ──


def test_a_bare_double_dash_is_never_stripped_and_ends_option_parsing():
    from gomoveshift.health.resume import strip_one_shot

    assert strip_one_shot(["drive", "--", "-odd-dir"]) == ["drive", "--", "-odd-dir"]
    # nothing after the first ``--`` is an option, so a positional spelled like a flag stays
    assert strip_one_shot(["drive", "--", "--fresh"]) == ["drive", "--", "--fresh"]
    assert strip_one_shot(["drive", CONFIRM, "--", "--fresh"]) == ["drive", "--", "--fresh"]


@pytest.mark.parametrize("prefix", ["--ost", "--os"])
def test_an_ambiguous_prefix_that_could_be_a_one_shot_flag_is_stripped(prefix):
    from gomoveshift.health.resume import strip_one_shot

    # grounding: --ost-audit-confirmed (one-shot) and --ost-system-review share the prefix
    assert _parse(["drive", "/src", prefix]) is None  # argparse: ambiguous
    assert strip_one_shot(["drive", "/src", prefix, "--ost-system-review"]) == [
        "drive",
        "/src",
        "--ost-system-review",
    ]
    assert strip_one_shot(["drive", "/src", prefix + "=1"]) == ["drive", "/src"]


def test_a_refused_recipe_write_removes_the_previous_runs_recipe(tmp_path, monkeypatch):
    from gomoveshift.health import resume

    first = write_recipe(tmp_path, ["drive", "/old-src"], cwd="/", sha=None, now_iso="x")
    assert first is not None and read_recipe(first) is not None

    def boom():
        raise RuntimeError("no parser")

    monkeypatch.setattr(resume, "_drive_options", boom)
    refused = write_recipe(
        tmp_path, ["drive", "/new-src", "--location", "X"], cwd="/", sha=None, now_iso="y"
    )
    assert refused is None
    assert not first.exists()  # heal finds no recipe, never the earlier run's argv
    assert read_recipe(first) is None


def test_a_refused_recipe_write_removes_a_leftover_tmp_file(tmp_path, monkeypatch):
    from gomoveshift.health import resume

    tmp = tmp_path / (resume.RECIPE_NAME + ".tmp")
    tmp.write_text("{}")  # an earlier run's half-written recipe
    keep = tmp_path / "keep.json"
    keep.write_text("{}")

    def boom():
        raise RuntimeError("no parser")

    monkeypatch.setattr(resume, "_drive_options", boom)
    assert (
        write_recipe(
            tmp_path, ["drive", "/new-src", "--location", "X"], cwd="/", sha=None, now_iso="y"
        )
        is None
    )
    assert not tmp.exists()  # a replay can never pick up the leftover
    assert keep.exists()
