"""CLI surface for ``gomoveshift reconcile-playlists`` — argparse wiring,
dispatch, exit codes, dry-run vs --apply.

The diff/apply engine itself is covered in test_playlist_reconciler.py; this
file pins the thin plumbing that makes it runnable from cron, which the
apply-glue change deliberately scoped out.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import gomoveshift.pipeline.playlist_reconciler as playlist_reconciler
from gomoveshift.cli.handlers.reconcile_playlists import (
    cmd_reconcile_playlists,
    run_reconcile_playlists,
)


def _reconciler_config():
    return SimpleNamespace(
        max_inserts_per_run=40,
        newest_first=True,
        backfill_since="2026-01-01",
        overrides_path="",
    )


def test_subparser_is_registered():
    """Confirm ``gomoveshift reconcile-playlists`` is dispatched by the
    top-level parser, not a typo."""
    from gomoveshift.cli import _COMMAND_MAP, _build_parser

    assert "reconcile-playlists" in _COMMAND_MAP
    parser = _build_parser()
    args = parser.parse_args(["reconcile-playlists"])
    assert args.command == "reconcile-playlists"
    assert args.apply is False
    args = parser.parse_args(["reconcile-playlists", "--apply"])
    assert args.apply is True


def test_dry_run_prints_plan_and_exits_zero(monkeypatch, capsys):
    calls = {}

    def fake_reconcile(service, config, *, apply, now=None):
        calls["apply"] = apply
        return {"planned": 3, "applied": 0, "skipped": 0, "unroutable": 1}

    monkeypatch.setattr(playlist_reconciler, "reconcile", fake_reconcile)
    rc = run_reconcile_playlists(
        apply=False, reconciler_config=_reconciler_config(), service=object()
    )
    assert rc == 0
    assert calls["apply"] is False
    out = capsys.readouterr().out
    assert "planned=3" in out
    assert "unroutable=1" in out
    assert "dry-run" in out


def test_apply_passes_through_and_reports(monkeypatch, capsys):
    def fake_reconcile(service, config, *, apply, now=None):
        assert apply is True
        return {"planned": 2, "applied": 2, "skipped": 0, "unroutable": 0}

    monkeypatch.setattr(playlist_reconciler, "reconcile", fake_reconcile)
    rc = run_reconcile_playlists(
        apply=True, reconciler_config=_reconciler_config(), service=object()
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "applied=2" in out
    assert "dry-run" not in out


def test_missing_credentials_exit_clean(monkeypatch, capsys):
    """A creds/libs failure is a clean exit-2 message for the cron log, not a
    traceback."""
    import gomoveshift.pipeline._yt_creds as yt_creds

    def boom(*_a, **_k):
        raise RuntimeError("no creds here")

    monkeypatch.setattr(yt_creds, "load_youtube_credentials", boom)
    rc = run_reconcile_playlists(apply=False, reconciler_config=_reconciler_config())
    assert rc == 2
    err = capsys.readouterr().err
    assert "YouTube client" in err


def test_cmd_adapter_threads_reconciler_config(monkeypatch):
    """The dispatch adapter must hand the handler the ``playlist_reconciler``
    sub-config, not the whole Config."""
    import gomoveshift.cli.handlers.reconcile_playlists as mod

    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(mod, "run_reconcile_playlists", fake_run)
    sub = _reconciler_config()
    args = SimpleNamespace(apply=True, credentials_dir=None, ledger=Path("mine.jsonl"))
    config = SimpleNamespace(playlist_reconciler=sub)
    assert cmd_reconcile_playlists(args, config) == 0
    assert seen["ledger"] == Path("mine.jsonl")
    assert seen["reconciler_config"] is sub
    assert seen["apply"] is True


# --- --prune / --undo / --audit ---------------------------------------------------


def _summary(**extra):
    return {"planned": 1, "applied": 0, "skipped": 0, "unroutable": 0, **extra}


def _pruned(**kw):
    return {"planned": 0, "applied": 0, "skipped": 0, "deferred": 0, "errors": 0, "rows": [], **kw}


def test_prune_flags_are_parsed_with_safe_defaults():
    from gomoveshift.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["reconcile-playlists"])
    # include_nonpublic None = "use config playlist_reconciler.prune_include_nonpublic"
    assert (args.prune, args.audit, args.include_nonpublic, args.undo) == (
        False,
        False,
        None,
        None,
    )
    assert args.max_deletes == 150
    args = parser.parse_args(
        ["reconcile-playlists", "--prune", "--max-deletes", "5", "--include-nonpublic"]
    )
    assert (args.prune, args.max_deletes, args.include_nonpublic) == (True, 5, True)
    off = parser.parse_args(["reconcile-playlists", "--prune", "--no-include-nonpublic"])
    assert off.include_nonpublic is False
    args = parser.parse_args(
        ["reconcile-playlists", "--credentials-dir", "x", "--prune", "--ledger", "l.jsonl"]
    )
    assert (args.credentials_dir, args.ledger) == (Path("x"), Path("l.jsonl"))


def test_nightly_apply_calls_reconcile_without_prune_kwargs(monkeypatch):
    seen = {}

    def fake_reconcile(service, config, **kw):
        seen.update(kw)
        return _summary()

    monkeypatch.setattr(playlist_reconciler, "reconcile", fake_reconcile)
    run_reconcile_playlists(apply=True, reconciler_config=_reconciler_config(), service=object())
    assert set(seen) == {"apply", "now"}


def test_prune_dry_run_lists_rows_and_summary(monkeypatch, capsys):
    row = {
        "reason": "unsorted",
        "playlist_title": "Unsorted (fallback)",
        "video_id": "v1",
        "item_id": "it1",
        "position": 3,
    }
    seen = {}

    def fake_reconcile(service, config, **kw):
        seen.update(kw)
        return _summary(prune=_pruned(planned=1, skipped=2, deferred=4, rows=[row]))

    monkeypatch.setattr(playlist_reconciler, "reconcile", fake_reconcile)
    rc = run_reconcile_playlists(
        apply=False,
        reconciler_config=_reconciler_config(),
        service=object(),
        prune=True,
        max_deletes=7,
        include_nonpublic=True,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert (seen["prune"], seen["max_deletes"], seen["include_nonpublic"]) == (True, 7, True)
    assert "ledger" not in seen
    assert "RECONCILE_PRUNE unsorted playlist='Unsorted (fallback)' video=v1 item=it1" in out
    assert "Prune: planned=1 applied=0 skipped=2 deferred=4 errors=0" in out
    assert "dry-run" in out


def test_prune_apply_passes_a_ledger_and_exits_1_on_errors(monkeypatch, capsys, tmp_path):
    seen = {}

    def fake_reconcile(service, config, **kw):
        seen.update(kw)
        return _summary(prune=_pruned(planned=2, applied=1, errors=1))

    monkeypatch.setattr(playlist_reconciler, "reconcile", fake_reconcile)
    ledger = tmp_path / "l.jsonl"
    rc = run_reconcile_playlists(
        apply=True,
        reconciler_config=_reconciler_config(),
        service=object(),
        prune=True,
        ledger=ledger,
    )
    assert rc == 1
    assert seen["ledger"] == ledger
    assert str(ledger) in capsys.readouterr().out


@pytest.mark.parametrize("apply", [True, False])
def test_prune_counts_are_the_last_output_line(monkeypatch, capsys, tmp_path, apply):
    """The cron logs ``tail -1``: it must carry planned/applied/deferred/errors."""
    monkeypatch.setattr(
        playlist_reconciler,
        "reconcile",
        lambda service, config, **kw: _summary(
            prune=_pruned(planned=3, applied=2, skipped=1, deferred=5, errors=1)
        ),
    )
    run_reconcile_playlists(
        apply=apply,
        reconciler_config=_reconciler_config(),
        service=object(),
        prune=True,
        ledger=tmp_path / "l.jsonl",
    )
    last = capsys.readouterr().out.rstrip("\n").splitlines()[-1]
    assert last == "Prune: planned=3 applied=2 skipped=1 deferred=5 errors=1"


def test_audit_prints_one_line_per_check(monkeypatch, capsys):
    monkeypatch.setattr(
        playlist_reconciler,
        "reconcile",
        lambda service, config, **kw: _summary(audit={"unsorted_items": 10, "dual_shorts": 0}),
    )
    rc = run_reconcile_playlists(
        apply=False, reconciler_config=_reconciler_config(), service=object(), audit=True
    )
    assert rc == 0
    assert capsys.readouterr().out.splitlines() == [
        "RECONCILE_AUDIT unsorted_items=10",
        "RECONCILE_AUDIT dual_shorts=0",
    ]


def test_undo_restores_from_the_ledger(monkeypatch, capsys, tmp_path):
    seen = {}

    def fake_undo(service, ledger):
        seen["ledger"] = ledger
        return {"restored": 3, "skipped": 1, "errors": 0}

    monkeypatch.setattr(playlist_reconciler, "undo_deletes", fake_undo)
    monkeypatch.setattr(
        playlist_reconciler, "reconcile", lambda *a, **k: pytest.fail("undo must not reconcile")
    )
    rc = run_reconcile_playlists(
        apply=False,
        reconciler_config=_reconciler_config(),
        service=object(),
        undo=tmp_path / "l.jsonl",
    )
    assert rc == 0
    assert seen["ledger"] == tmp_path / "l.jsonl"
    assert "Undo: restored=3 skipped=1 errors=0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "flags",
    [["--audit", "--undo", "l.jsonl"], ["--undo", "l.jsonl", "--prune"], ["--audit", "--prune"]],
)
def test_prune_undo_audit_are_mutually_exclusive(flags):
    from gomoveshift.cli import _build_parser

    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args(["reconcile-playlists", *flags])
    assert exc.value.code == 2


def test_include_nonpublic_flag_is_tri_state_through_the_handler(monkeypatch):
    seen = {}

    def fake_reconcile(service, config, **kw):
        seen.update(kw)
        return _summary(prune=_pruned())

    monkeypatch.setattr(playlist_reconciler, "reconcile", fake_reconcile)
    run_reconcile_playlists(
        apply=False, reconciler_config=_reconciler_config(), service=object(), prune=True
    )
    assert seen["include_nonpublic"] is None  # config decides
    run_reconcile_playlists(
        apply=False,
        reconciler_config=_reconciler_config(),
        service=object(),
        prune=True,
        include_nonpublic=False,
    )
    assert seen["include_nonpublic"] is False


def _summary_with_unrouted(n):
    return {
        "planned": 0,
        "applied": 0,
        "skipped": 0,
        "unroutable": n,
        "unrouted": [{"video_id": f"v{i:02d}", "title": "t", "kind": "real_lf"} for i in range(n)],
    }


def test_summary_line_names_unrouted_video_ids(monkeypatch, capsys):
    monkeypatch.setattr(playlist_reconciler, "reconcile", lambda *a, **k: _summary_with_unrouted(2))
    run_reconcile_playlists(apply=True, reconciler_config=_reconciler_config(), service=object())
    line = capsys.readouterr().out.splitlines()[0]
    assert "unroutable=2 [v00,v01]" in line


def test_summary_line_caps_unrouted_ids_at_25(monkeypatch, capsys):
    monkeypatch.setattr(
        playlist_reconciler, "reconcile", lambda *a, **k: _summary_with_unrouted(30)
    )
    run_reconcile_playlists(apply=True, reconciler_config=_reconciler_config(), service=object())
    line = capsys.readouterr().out.splitlines()[0]
    assert "v24" in line and "v25" not in line and "and 5 more" in line
