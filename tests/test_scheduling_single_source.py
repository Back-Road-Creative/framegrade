"""Every slot decision lives in ``seo_manager.scheduling``.

GMS keeps thin wrappers (``content_ledger.next_free_slot``, ``schedule_report``,
Rule 8, Phase D's LF 2..N guard) that translate its config + ledger into what
``seo_manager`` decides over. These tests pin that each wrapper asks seo_manager, and an
AST guard that weekday arithmetic does not grow back outside the allowlist.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import schedule_constants
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger

EST = schedule_constants.EST
PKG = Path(__file__).resolve().parents[1] / "gomoveshift"
_SEO_SPEC = importlib.util.find_spec("seo_manager")
SEO_PKG = Path(_SEO_SPEC.origin).parent if _SEO_SPEC and _SEO_SPEC.origin else None


@pytest.fixture
def restore_schedule_module():
    saved = (
        schedule_constants.LONG_FORM_DAYS,
        schedule_constants.LONG_FORM_TIME,
        schedule_constants.SHORTS_TIMES,
    )
    yield
    (
        schedule_constants.LONG_FORM_DAYS,
        schedule_constants.LONG_FORM_TIME,
        schedule_constants.SHORTS_TIMES,
    ) = saved
    from gomoveshift.pipeline import scheduler as _sched

    _sched.LONG_FORM_DAYS, _sched.LONG_FORM_TIME, _sched.SHORTS_TIMES = saved


def _lf(date: str, region: str | None, project: str = "other") -> ContentEntry:
    return ContentEntry(
        date=date,
        time="16:00:00",
        content_type="longform",
        title="Booked",
        project_name=project,
        status="scheduled",
        source="pipeline",
        region=region,
    )


def _ledger(tmp_path: Path, *entries: ContentEntry) -> ContentLedger:
    ledger = ContentLedger(entries=list(entries))
    ledger.save(tmp_path / ".content_ledger.json")
    return ledger


# -- (a) each wrapper asks seo_manager ---------------------------------------


def test_next_free_slot_wrapper_asks_seo_manager(tmp_path):
    from gomoveshift.pipeline.content_ledger import next_free_slot

    sentinel = datetime(2031, 1, 2, 16, 0, tzinfo=EST)
    with patch("seo_manager.scheduling.slots.next_free_slot", return_value=sentinel) as spy:
        got = next_free_slot(_ledger(tmp_path), Config(), "usa", publish_day="tue")
    assert got == sentinel
    assert spy.call_count == 1


def test_next_free_slot_still_refuses_non_longform(tmp_path):
    from gomoveshift.pipeline.content_ledger import next_free_slot

    with pytest.raises(ValueError, match="long-form slots only"):
        next_free_slot(_ledger(tmp_path), Config(), "usa", "shorts")


def test_rule_8_asks_seo_manager_and_keeps_its_message(monkeypatch):
    from gomoveshift.pipeline import quality_gates

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({1}))
    slots = [SimpleNamespace(date="2026-10-13"), SimpleNamespace(date="2026-10-14")]
    state = {"drive_schedule": SimpleNamespace(lf_slots=slots)}
    with patch(
        "seo_manager.scheduling.slots.validate_anchor",
        wraps=__import__(
            "seo_manager.scheduling.slots", fromlist=["validate_anchor"]
        ).validate_anchor,
    ) as spy:
        errors = quality_gates.check_schedule_dates_in_long_form_days(state)
    assert spy.call_count == 1
    assert errors == ["lf_slot dates fall outside LONG_FORM_DAYS=[1]: 2026-10-14 (weekday=2)"]


def test_rule_8_unparseable_date_message_unchanged(monkeypatch):
    from gomoveshift.pipeline import quality_gates

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({1}))
    state = {"drive_schedule": SimpleNamespace(lf_slots=[SimpleNamespace(date="soon")])}
    assert quality_gates.check_schedule_dates_in_long_form_days(state) == [
        "lf_slot dates fall outside LONG_FORM_DAYS=[1]: soon (unparseable)"
    ]


def test_report_wrapper_builds_through_seo_manager(tmp_path):
    from gomoveshift.pipeline.schedule_report import build_schedule_report

    import seo_manager.scheduling.report as seo_report

    with patch.object(seo_report, "build", wraps=seo_report.build) as spy:
        report = build_schedule_report(
            _ledger(tmp_path), Config(), after=datetime(2026, 8, 10, 9, 0, tzinfo=EST)
        )
    assert spy.call_count == 1
    assert {"as_of", "next_available", "queue_ends", "scheduled_longforms"} <= set(report)


def test_scheduler_has_no_base_date_of_its_own():
    """The drive scheduler lays shorts out from a base date it is HANDED; with none it
    refuses rather than pick a ledger-blind LF day itself."""
    from gomoveshift.pipeline.scheduler import PipelineScheduler

    with pytest.raises(ValueError, match="start_date is required"):
        PipelineScheduler().generate_drive_schedule(
            regular_lfs=[SimpleNamespace(index=1)], region="usa", publish_day="wed"
        )


def test_next_lf_interval_asks_seo_manager():
    """The shorts-window end (the cadence day after an LF) is seo_manager's answer."""
    from gomoveshift.pipeline.scheduler import PipelineScheduler

    sched = PipelineScheduler(long_form_days={2})
    lf = datetime(2031, 1, 1, 16, 0, tzinfo=EST)  # a Wednesday
    with patch(
        "seo_manager.scheduling.slots.next_free_slot",
        wraps=__import__("seo_manager.scheduling.slots", fromlist=["x"]).next_free_slot,
    ) as spy:
        got = sched._get_next_long_form_publish(lf)
    assert spy.call_count == 1
    assert got == datetime.combine(lf.date() + timedelta(days=7), sched.long_form_time, tzinfo=EST)


def test_metadata_slot_without_a_ledger_asks_seo_manager(monkeypatch):
    from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

    monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({1}))
    gen = LongFormMetadataGenerator.__new__(LongFormMetadataGenerator)
    gen._ledger = None
    sentinel = datetime(2031, 1, 7, 16, 0, tzinfo=EST)
    with patch("seo_manager.scheduling.slots.next_free_slot", return_value=sentinel) as spy:
        slot = gen.get_next_publish_slot("longform")
    assert spy.call_count == 1
    assert slot["date"] == "2031-01-07"


def test_phase_d_lf_dates_guard_asks_seo_manager_and_keeps_message(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _assert_lf_dates_free

    import seo_manager.scheduling.slots as seo_slots

    ledger = _ledger(tmp_path, _lf("2026-10-20", "usa"))
    with patch.object(seo_slots, "assert_free", wraps=seo_slots.assert_free) as spy:
        with pytest.raises(RuntimeError, match="LF 2 of 2 would publish on 2026-10-20"):
            _assert_lf_dates_free(ledger, Config(), "usa", "2026-10-13", 2)
    assert spy.call_count >= 1
    _assert_lf_dates_free(ledger, Config(), "ireland", "2026-10-13", 2)  # other region: free


def _store_with(tmp_path: Path, project: str, *rows):
    from seo_manager.scheduling.store import ScheduleStore

    st = ScheduleStore(tmp_path / "schedule.sqlite3", EST, ["ireland", "usa"])
    st.init()
    for idx, day in rows:
        st.book(project, idx, "ireland", datetime.fromisoformat(day).replace(hour=14, tzinfo=EST))


def test_phase_d_lf_dates_guard_exempts_the_runs_own_store_rows(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _assert_lf_dates_free

    ledger = _ledger(tmp_path)
    _store_with(tmp_path, "mizen_s5_s7", (1, "2026-10-11"), (2, "2026-10-18"))
    _assert_lf_dates_free(ledger, Config(), "ireland", "2026-10-11", 2, project="mizen_s5_s7")


def test_phase_d_lf_dates_guard_refuses_another_projects_store_row(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _assert_lf_dates_free

    ledger = _ledger(tmp_path)
    _store_with(tmp_path, "rival", (1, "2026-10-18"))
    with pytest.raises(RuntimeError, match="booked") as exc:
        _assert_lf_dates_free(ledger, Config(), "ireland", "2026-10-11", 2, project="mizen_s5_s7")
    assert "content ledger" not in str(exc.value)


# -- (d) boundary guards ------------------------------------------------------

# Files that still do weekday arithmetic of their own, and why. Only the first entry is a
# slot resolver (the ONE documented holdout: the superlong day), and it is allowed only inside
# the named functions -- new weekday arithmetic elsewhere in that module fails. The rest are
# display or analytics. A value of ``None`` allows the whole file. New weekday arithmetic
# anywhere else fails.
WEEKDAY_ALLOWLIST: dict[str, tuple[str, frozenset | None]] = {
    # the one slot-resolver holdout, keyed by (file, enclosing function)
    "pipeline/schedule_constants.py": (
        "GMS superlong resolver (documented holdout)",
        frozenset({"_related_superlong_weekday", "resolve_superlong_publish_day"}),
    ),
    # not slot decisions: display or analytics
    "console/pubqueue_calendar.py": ("calendar week anchor for the queue view", None),
    "pipeline/social_generator.py": ("weekday label in post copy", None),
    "tracking/slot_performance.py": ("analytics, reads past slots", None),
    "tracking/evidence_export.py": (
        "evidence record of a past post's local weekday, not a slot decision",
        frozenset({"_slot"}),
    ),
}
# Files whose slot decisions moved to seo_manager: must hold NO weekday arithmetic.
WRAPPER_FILES = (
    "pipeline/content_ledger.py",
    "pipeline/schedule_report.py",
    "pipeline/quality_gates.py",
    "pipeline/state_invariants.py",
    "pipeline/drive_phases/phase_d_scheduling.py",
    "pipeline/scheduler.py",
    "pipeline/seo_slots.py",
    "cli/handlers/drive.py",
)
_WEEKDAY_ATTRS = {"weekday", "isoweekday"}


def _is_mod_7(n: ast.AST) -> bool:
    return (
        isinstance(n, ast.BinOp)
        and isinstance(n.op, ast.Mod)
        and isinstance(n.right, ast.Constant)
        and n.right.value == 7
    )


def _enclosing_function(tree: ast.AST, lineno: int) -> str | None:
    """Name of the innermost function whose body spans ``lineno``."""
    best, span = None, None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = n.end_lineno or n.lineno
            if n.lineno <= lineno <= end and (span is None or end - n.lineno < span):
                best, span = n.name, end - n.lineno
    return best


def _weekday_calls(path: Path, allowed_funcs: frozenset | None = None) -> list[int]:
    """Lines doing weekday arithmetic: ``.weekday()``/``.isoweekday()`` calls and ``% 7``.

    Lines inside a function named in ``allowed_funcs`` are not reported.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = sorted(
        n.lineno
        for n in ast.walk(tree)
        if _is_mod_7(n)
        or (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in _WEEKDAY_ATTRS
        )
    )
    if allowed_funcs is None:
        return lines
    return [ln for ln in lines if _enclosing_function(tree, ln) not in allowed_funcs]


def test_the_guard_sees_mod_7_slot_arithmetic(tmp_path):
    src = tmp_path / "x.py"
    src.write_text("def f(a, b):\n    return (a - b) % 7\n", encoding="utf-8")
    assert _weekday_calls(src) == [2]


def test_no_weekday_arithmetic_outside_the_allowlist():
    offenders = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        _why, funcs = WEEKDAY_ALLOWLIST.get(rel, ("", frozenset()))
        if funcs is None:
            continue
        lines = _weekday_calls(path, funcs)
        if lines:
            offenders[rel] = lines
    assert not offenders, (
        "weekday arithmetic outside the allowlist; slot decisions belong in "
        f"seo_manager.scheduling: {offenders}"
    )


def test_allowlist_is_keyed_by_function_not_whole_module(tmp_path):
    src = tmp_path / "m.py"
    src.write_text(
        "def ok(d):\n    return d.weekday()\n\ndef sneaky(d):\n    return d.weekday()\n",
        encoding="utf-8",
    )
    assert _weekday_calls(src, frozenset({"ok"})) == [5]
    sc = WEEKDAY_ALLOWLIST["pipeline/schedule_constants.py"][1]
    assert sc == {"_related_superlong_weekday", "resolve_superlong_publish_day"}


def test_allowlist_has_no_stale_entries():
    stale = [r for r in WEEKDAY_ALLOWLIST if not _weekday_calls(PKG / r)]
    assert not stale, f"allowlisted files with no weekday arithmetic left (remove them): {stale}"


@pytest.mark.parametrize("rel", WRAPPER_FILES)
def test_wrapper_files_hold_no_weekday_arithmetic(rel):
    assert rel not in WEEKDAY_ALLOWLIST
    assert not _weekday_calls(PKG / rel)


def test_wrapper_files_do_not_rebuild_the_slot_search():
    """The deleted long-form search loop and occupancy test must not come back."""
    for rel in ("pipeline/content_ledger.py", "pipeline/schedule_report.py"):
        src = (PKG / rel).read_text(encoding="utf-8")
        assert "candidate.weekday" not in src and "def _occupied" not in src, rel


def test_seo_manager_never_imports_gomoveshift():
    if SEO_PKG is None or not SEO_PKG.is_dir():
        pytest.skip("seo_manager is not installed")
    bad = []
    for path in SEO_PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(n.split(".")[0] == "gomoveshift" for n in names):
                bad.append(f"{path.relative_to(SEO_PKG)}:{node.lineno}")
    assert not bad, bad


def test_the_superlong_resolver_stays_documented():
    """The GMS superlong day resolver is the one allowlisted holdout (seo_manager's
    ``superlong_day`` is a single value; GMS resolves per region and per related part)."""
    assert "pipeline/schedule_constants.py" in WEEKDAY_ALLOWLIST
    assert callable(schedule_constants.resolve_superlong_publish_day)
    doc = (Path(__file__).resolve().parents[1] / "docs" / "unattended-cadence.md").read_text()
    assert "superlong" in doc.lower() and "holdout" in doc.lower()


def test_upload_without_lf_slots_fails_closed_instead_of_a_monday_fallback(tmp_path):
    from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

    for schedule in (None, SimpleNamespace(lf_slots=[])):
        with pytest.raises(RuntimeError, match="lf_slots"):
            esc._upload_lfs_to_youtube(
                drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
                regular_lf_paths=["lf.mp4"],
                metadata_per_lf={},
                drive_schedule=schedule,
                youtube_uploads={},
                state=SimpleNamespace(youtube_uploads={}),
                output_base=tmp_path,
                config=Config(),
                save_state=lambda s: None,
                skip_youtube_upload=False,
                youtube_upload_enabled=True,
            )


def test_phase_d_has_no_monday_publish_fallback():
    src = (PKG / "pipeline/drive_phases/phase_d.py").read_text(encoding="utf-8")
    assert "plan_weekly_mondays" not in src


# -- LF 2..N dates: seo_manager's series rule, no arithmetic in GMS -------------------

import seo_manager.scheduling.slots as _seo_slots  # noqa: E402


def _spaced_14(lf1, count):
    from datetime import date as _d

    base = _d.fromisoformat(str(lf1))
    return [base + timedelta(days=14 * i) for i in range(count)]


def test_drive_schedule_lf_dates_come_from_seo_manager_series():
    """Patch the series rule to a fortnight: the scheduler must follow it, not add 7 days."""
    from gomoveshift.pipeline.scheduler import PipelineScheduler

    with patch.object(_seo_slots, "lf_series_dates", side_effect=_spaced_14) as spy:
        sched = PipelineScheduler(long_form_days={1})
        out = sched.generate_drive_schedule(
            regular_lfs=[SimpleNamespace(index=i) for i in (1, 2, 3)],
            region="usa",
            publish_day="tue",
            start_date="2026-10-13",
        )
    assert spy.call_count >= 1
    lf_dates = (
        [s.date for s in out["lf_slots"]]
        if isinstance(out, dict)
        else [s.date for s in out.lf_slots]
    )
    assert lf_dates == ["2026-10-13", "2026-10-27", "2026-11-10"]


def test_drive_schedule_lf_dates_equal_the_series_rule_for_a_tue_fri_drive():
    from gomoveshift.pipeline.scheduler import PipelineScheduler

    sched = PipelineScheduler(long_form_days={1, 4})
    out = sched.generate_drive_schedule(
        regular_lfs=[SimpleNamespace(index=i) for i in (1, 2, 3)],
        region="usa",
        publish_day="fri",
        start_date="2026-10-09",
    )
    lf_dates = [s.date for s in (out["lf_slots"] if isinstance(out, dict) else out.lf_slots)]
    assert lf_dates == [d.isoformat() for d in _seo_slots.lf_series_dates("2026-10-09", 3)]


def test_phase_d_lf_dates_guard_checks_the_series_rule_dates(tmp_path):
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _assert_lf_dates_free

    ledger = _ledger(tmp_path, _lf("2026-10-27", "usa"))  # a fortnight on, not a week
    with patch.object(_seo_slots, "lf_series_dates", side_effect=_spaced_14):
        with pytest.raises(RuntimeError, match="LF 2 of 2 would publish on 2026-10-27"):
            _assert_lf_dates_free(ledger, Config(), "usa", "2026-10-13", 2)


def _interval_arithmetic(path: Path) -> list[int]:
    """Lines building a ``timedelta`` from the LF interval or a literal 7 days / any weeks."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for n in ast.walk(tree):
        if not (
            isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == "timedelta"
        ):
            continue
        for a in n.args[:1]:  # timedelta(7) / timedelta(LF_PUBLISH_INTERVAL_DAYS): days
            names = {x.id for x in ast.walk(a) if isinstance(x, ast.Name)}
            consts = {x.value for x in ast.walk(a) if isinstance(x, ast.Constant)}
            if "LF_PUBLISH_INTERVAL_DAYS" in names or 7 in consts:
                out.append(n.lineno)
        for kw in n.keywords:
            names = {x.id for x in ast.walk(kw.value) if isinstance(x, ast.Name)}
            consts = {x.value for x in ast.walk(kw.value) if isinstance(x, ast.Constant)}
            if (
                kw.arg == "weeks"
                or "LF_PUBLISH_INTERVAL_DAYS" in names
                or (kw.arg == "days" and 7 in consts)
            ):
                out.append(n.lineno)
    return out


# The contested-share window (scheduler.py) sizes a date range; it is not the series rule.
_INTERVAL_NAME_ALLOWED = {"pipeline/scheduler.py"}


def _interval_name_uses(path: Path) -> list[int]:
    """Lines that read ``LF_PUBLISH_INTERVAL_DAYS`` at all (it can reach a timedelta via a variable)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "LF_PUBLISH_INTERVAL_DAYS"
    ]


@pytest.mark.parametrize("rel", [r for r in WRAPPER_FILES if r not in _INTERVAL_NAME_ALLOWED])
def test_wrapper_files_never_read_the_lf_interval(rel):
    assert not _interval_name_uses(PKG / rel)


def test_the_interval_guard_sees_series_arithmetic(tmp_path):
    src = tmp_path / "x.py"
    src.write_text(
        "from datetime import timedelta\n"
        "a = d + timedelta(days=i * LF_PUBLISH_INTERVAL_DAYS)\n"
        "b = d + timedelta(days=7)\n"
        "c = d + timedelta(weeks=1)\n"
        "e = d + timedelta(days=1)\n"
        "f = d + timedelta(7)\n"
        "g = d + timedelta(LF_PUBLISH_INTERVAL_DAYS)\n",
        encoding="utf-8",
    )
    assert _interval_arithmetic(src) == [2, 3, 4, 6, 7]
    names = tmp_path / "y.py"
    names.write_text(
        "step = LF_PUBLISH_INTERVAL_DAYS\nd2 = d + timedelta(step)\n", encoding="utf-8"
    )
    assert _interval_name_uses(names) == [1]


@pytest.mark.parametrize("rel", WRAPPER_FILES)
def test_wrapper_files_hold_no_lf_interval_arithmetic(rel):
    assert not _interval_arithmetic(PKG / rel), (
        "the LF series (LF n = LF1 + 7*(n-1)) is seo_manager.scheduling.slots.lf_series_dates"
    )
