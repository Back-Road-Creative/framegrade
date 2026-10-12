"""Tests for preflight SEO ownership repair.

Detects an ownership mismatch on SEO state files at preflight, attempts
``os.chown`` repair, and on PermissionError fails fast with the exact
``sudo chown <user>:<group> <files>`` command. The owner account is
configuration, never a built-in default: ``GMS_SEO_OWNER_USER`` /
``GMS_SEO_OWNER_GROUP`` (or the ``target_user`` / ``target_group``
arguments) only name it in the repair hint; the check itself always runs and
the hint shows placeholders while either is unset. The tests use fictional
account names.

Files covered:
  - ``<seo_storage_path>/.content_ledger.json``
  - ``<seo_storage_path>/<project>/{.lock,events.jsonl,index.json}``
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

OWNER_USER = "pipeline-owner"
OWNER_GROUP = "pipeline-group"


@pytest.fixture(autouse=True)
def _owner_account_configured(monkeypatch):
    """Configure the fictional owner account for every test below (it names
    the repair hint). ``TestOwnerConfiguration`` clears these to test the
    placeholder hint."""
    monkeypatch.setenv("GMS_SEO_OWNER_USER", OWNER_USER)
    monkeypatch.setenv("GMS_SEO_OWNER_GROUP", OWNER_GROUP)


def _make_seo_tree(root: Path, project: str = "gomoveshift") -> tuple[Path, Path]:
    """Create the two SEO state files; return (ledger, lock) paths."""
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / ".content_ledger.json"
    ledger.write_text("{}")
    project_dir = root / project
    project_dir.mkdir()
    lock = project_dir / ".lock"
    lock.write_text("")
    return ledger, lock


class TestCheckSeoOwnership:
    """check_seo_ownership: detect mismatch, repair, or fail with sudo cmd."""

    def test_ownership_matches_passes(self, tmp_path):
        """SEO files owned by current user → preflight passes (idempotent)."""
        from gomoveshift.pipeline.preflight import check_seo_ownership

        ledger, lock = _make_seo_tree(tmp_path)
        # Files just created — owned by current uid by definition.
        check_seo_ownership(tmp_path, project="gomoveshift")  # no raise

    def test_missing_files_pass(self, tmp_path):
        """Fresh install (no SEO files yet) → preflight is a no-op."""
        from gomoveshift.pipeline.preflight import check_seo_ownership

        # Empty dir: no .content_ledger.json, no project subdir.
        check_seo_ownership(tmp_path, project="gomoveshift")  # no raise

    def test_mismatch_with_chown_success_repairs_silently(self, tmp_path):
        """Different owner + os.chown succeeds → preflight passes (repaired)."""
        from gomoveshift.pipeline import preflight

        ledger, lock = _make_seo_tree(tmp_path)
        my_uid = os.geteuid()
        # Pretend the files are owned by another account.
        foreign_uid = my_uid + 1000

        original_stat = os.stat
        chown_calls: list[tuple[str, int, int]] = []

        def fake_stat(p, *args, **kwargs):
            r = original_stat(p, *args, **kwargs)
            sp = str(p)
            if sp.endswith(".content_ledger.json") or sp.endswith(".lock"):

                class _S:
                    st_uid = foreign_uid
                    st_gid = foreign_uid
                    # Strip S_IWGRP so the group-write fallback
                    # does NOT short-circuit and the chown branch fires
                    # under all umask conditions.
                    st_mode = r.st_mode & ~0o020
                    st_size = r.st_size
                    st_mtime = r.st_mtime
                    st_atime = r.st_atime
                    st_ctime = r.st_ctime
                    st_ino = r.st_ino
                    st_dev = r.st_dev
                    st_nlink = r.st_nlink

                return _S()
            return r

        def fake_chown(path, uid, gid):
            chown_calls.append((str(path), uid, gid))
            # Pretend repair worked.

        with (
            patch.object(preflight.os, "stat", fake_stat),
            patch.object(preflight.os, "chown", fake_chown),
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        # Both files should have been chown'd to current uid.
        assert len(chown_calls) == 2
        for _, uid, _ in chown_calls:
            assert uid == my_uid

    def test_mismatch_with_chown_permission_error_raises_with_sudo_command(self, tmp_path):
        """Different owner + chown fails (cross-user, no privs) → fail fast."""
        from gomoveshift.pipeline import preflight

        ledger, lock = _make_seo_tree(tmp_path)
        my_uid = os.geteuid()
        foreign_uid = my_uid + 1000

        original_stat = os.stat

        def fake_stat(p, *args, **kwargs):
            r = original_stat(p, *args, **kwargs)
            sp = str(p)
            if sp.endswith(".content_ledger.json") or sp.endswith(".lock"):

                class _S:
                    st_uid = foreign_uid
                    st_gid = foreign_uid
                    # Strip S_IWGRP so the group-write fallback
                    # does NOT short-circuit and the chown branch fires
                    # under all umask conditions.
                    st_mode = r.st_mode & ~0o020
                    st_size = r.st_size
                    st_mtime = r.st_mtime
                    st_atime = r.st_atime
                    st_ctime = r.st_ctime
                    st_ino = r.st_ino
                    st_dev = r.st_dev
                    st_nlink = r.st_nlink

                return _S()
            return r

        def fake_chown(path, uid, gid):
            raise PermissionError(1, "Operation not permitted", str(path))

        with (
            patch.object(preflight.os, "stat", fake_stat),
            patch.object(preflight.os, "chown", fake_chown),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        msg = str(excinfo.value)
        # Error must include both repair options and absolute paths.
        assert "sudo chown" in msg
        assert f"{OWNER_USER}:{OWNER_GROUP}" in msg
        assert "chmod g+w" in msg
        assert str(ledger) in msg
        assert str(lock) in msg


# ── group-write fallback + subdir walk ──────────────────────────────────


def _stat_override(real_stat, suffixes, *, st_uid, st_gid, mode_mask, mode_set=0):
    """Build a fake_stat that overrides uid/gid/mode for paths matching
    any of ``suffixes``. ``mode_mask`` is AND'd against the real mode
    before ``mode_set`` is OR'd in, so callers control group-write bit
    cleanly regardless of umask.
    """

    def fake_stat(p, *args, **kwargs):
        r = real_stat(p, *args, **kwargs)
        sp = str(p)
        if any(sp.endswith(s) for s in suffixes):

            class _S:
                pass

            s = _S()
            s.st_uid = st_uid
            s.st_gid = st_gid
            s.st_mode = (r.st_mode & mode_mask) | mode_set
            s.st_size = r.st_size
            s.st_mtime = r.st_mtime
            s.st_atime = r.st_atime
            s.st_ctime = r.st_ctime
            s.st_ino = r.st_ino
            s.st_dev = r.st_dev
            s.st_nlink = r.st_nlink
            return s
        return r

    return fake_stat


class TestGroupWriteFallback:
    """A file passes preflight when group-writable + group-member, even with
    a foreign uid, so a shared-group deployment needs no repeated
    ``sudo chown`` repairs.
    """

    def test_group_write_with_membership_passes(self, tmp_path, caplog):
        """Foreign uid + S_IWGRP set + gid in os.getgroups() → PASS, no chown."""
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        my_uid = os.geteuid()
        foreign_uid = my_uid + 1000
        foreign_gid = 99999  # any gid; we patch getgroups to include it

        chown_calls: list = []

        def fake_chown(path, uid, gid):
            chown_calls.append((str(path), uid, gid))

        suffixes = (".content_ledger.json", ".lock")
        fake_stat = _stat_override(
            os.stat,
            suffixes,
            st_uid=foreign_uid,
            st_gid=foreign_gid,
            mode_mask=0o777,
            mode_set=0o020,  # ensure S_IWGRP
        )

        import logging as _logging

        with (
            patch.object(preflight.os, "stat", fake_stat),
            patch.object(preflight.os, "chown", fake_chown),
            patch.object(preflight.os, "getgroups", lambda: [my_uid, foreign_gid]),
            caplog.at_level(_logging.INFO, logger="gomoveshift.pipeline.preflight"),
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        assert chown_calls == [], "no chown should be attempted on group-write path"
        assert any("group-write fallback" in rec.message for rec in caplog.records)

    def test_group_write_without_membership_fails(self, tmp_path):
        """Foreign uid + S_IWGRP set BUT gid NOT in os.getgroups() → FAIL."""
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        my_uid = os.geteuid()
        foreign_uid = my_uid + 1000

        suffixes = (".content_ledger.json", ".lock")
        fake_stat = _stat_override(
            os.stat,
            suffixes,
            st_uid=foreign_uid,
            st_gid=99999,
            mode_mask=0o777,
            mode_set=0o020,
        )

        with (
            patch.object(preflight.os, "stat", fake_stat),
            patch.object(
                preflight.os,
                "chown",
                lambda *a: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")),
            ),
            patch.object(preflight.os, "getgroups", lambda: [my_uid]),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        msg = str(excinfo.value)
        assert "chown" in msg and "chmod g+w" in msg

    def test_group_member_without_write_bit_fails(self, tmp_path):
        """Foreign uid + group-member BUT mode lacks S_IWGRP → FAIL."""
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        my_uid = os.geteuid()
        foreign_uid = my_uid + 1000
        foreign_gid = 99999

        suffixes = (".content_ledger.json", ".lock")
        fake_stat = _stat_override(
            os.stat,
            suffixes,
            st_uid=foreign_uid,
            st_gid=foreign_gid,
            mode_mask=0o777 & ~0o020,  # strip S_IWGRP
        )

        with (
            patch.object(preflight.os, "stat", fake_stat),
            patch.object(
                preflight.os,
                "chown",
                lambda *a: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")),
            ),
            patch.object(preflight.os, "getgroups", lambda: [my_uid, foreign_gid]),
            pytest.raises(RuntimeError),
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")


class TestSubdirWalk:
    """Coverage spans the ledger plus the per-project
    {.lock,events.jsonl,index.json}: events.jsonl and index.json are
    inspected alongside .lock.
    """

    def test_walks_extra_state_files(self, tmp_path):
        """events.jsonl and index.json are inspected alongside .lock."""
        from gomoveshift.pipeline import preflight

        ledger, lock = _make_seo_tree(tmp_path)
        events = lock.parent / "events.jsonl"
        index = lock.parent / "index.json"
        events.write_text("")
        index.write_text("{}")

        my_uid = os.geteuid()
        foreign_uid = my_uid + 1000

        # All four files foreign-owned, no group-write — every one must
        # land in the failure block.
        suffixes = (".content_ledger.json", ".lock", "events.jsonl", "index.json")
        fake_stat = _stat_override(
            os.stat,
            suffixes,
            st_uid=foreign_uid,
            st_gid=foreign_uid,
            mode_mask=0o777 & ~0o020,
        )

        with (
            patch.object(preflight.os, "stat", fake_stat),
            patch.object(
                preflight.os,
                "chown",
                lambda *a: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")),
            ),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        msg = str(excinfo.value)
        assert str(events) in msg
        assert str(index) in msg
        assert str(ledger) in msg
        assert str(lock) in msg

    def test_walks_extra_state_files_pass_with_group_write(self, tmp_path):
        """All four files group-writable + group-member → PASS, no raise."""
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        events = tmp_path / "gomoveshift" / "events.jsonl"
        index = tmp_path / "gomoveshift" / "index.json"
        events.write_text("")
        index.write_text("{}")

        my_uid = os.geteuid()
        foreign_uid = my_uid + 1000
        foreign_gid = 99999

        suffixes = (".content_ledger.json", ".lock", "events.jsonl", "index.json")
        fake_stat = _stat_override(
            os.stat,
            suffixes,
            st_uid=foreign_uid,
            st_gid=foreign_gid,
            mode_mask=0o777,
            mode_set=0o020,
        )

        with (
            patch.object(preflight.os, "stat", fake_stat),
            patch.object(preflight.os, "getgroups", lambda: [my_uid, foreign_gid]),
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")  # no raise


def _foreign_owned_stat():
    """A fake ``os.stat`` making the SEO state files foreign-owned and not
    group-writable, so an enabled check must try to repair and then fail."""
    foreign = os.geteuid() + 1000
    return _stat_override(
        os.stat,
        (".content_ledger.json", ".lock", "events.jsonl", "index.json"),
        st_uid=foreign,
        st_gid=foreign,
        mode_mask=0o777 & ~0o020,
    )


def _deny_chown(*_args):
    raise PermissionError(1, "Operation not permitted")


class TestOwnerConfiguration:
    """No built-in owner account: the check always runs, and the owner
    variables only shape the repair hint."""

    @pytest.fixture(autouse=True)
    def _unset(self, monkeypatch):
        monkeypatch.delenv("GMS_SEO_OWNER_USER", raising=False)
        monkeypatch.delenv("GMS_SEO_OWNER_GROUP", raising=False)

    def test_unset_still_detects_and_hints_with_placeholders(self, tmp_path):
        """Nothing configured -> the check still runs, tries the repair, and
        fails with placeholders plus the variables that name the accounts."""
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        chown_calls: list = []

        def deny(*a):
            chown_calls.append(a)
            raise PermissionError(1, "Operation not permitted")

        with (
            patch.object(preflight.os, "stat", _foreign_owned_stat()),
            patch.object(preflight.os, "chown", deny),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        assert chown_calls, "repair must be attempted without a configured owner"
        msg = str(excinfo.value)
        assert "sudo chown <owner-user>:<owner-group>" in msg
        assert "GMS_SEO_OWNER_USER" in msg and "GMS_SEO_OWNER_GROUP" in msg
        assert "chmod g+w" in msg

    def test_unset_passes_when_files_are_writable(self, tmp_path):
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        preflight.check_seo_ownership(tmp_path, project="gomoveshift")  # no raise

    def test_unset_repairs_silently_when_chown_succeeds(self, tmp_path):
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        chown_calls: list = []

        with (
            patch.object(preflight.os, "stat", _foreign_owned_stat()),
            patch.object(preflight.os, "chown", lambda *a: chown_calls.append(a)),
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")  # no raise

        assert len(chown_calls) == 2

    def test_only_user_set_keeps_placeholders(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        monkeypatch.setenv("GMS_SEO_OWNER_USER", OWNER_USER)

        with (
            patch.object(preflight.os, "stat", _foreign_owned_stat()),
            patch.object(preflight.os, "chown", _deny_chown),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        msg = str(excinfo.value)
        assert "sudo chown <owner-user>:<owner-group>" in msg
        assert OWNER_USER not in msg

    def test_blank_env_values_count_as_unset(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        monkeypatch.setenv("GMS_SEO_OWNER_USER", "  ")
        monkeypatch.setenv("GMS_SEO_OWNER_GROUP", "")

        with (
            patch.object(preflight.os, "stat", _foreign_owned_stat()),
            patch.object(preflight.os, "chown", _deny_chown),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        assert "sudo chown <owner-user>:<owner-group>" in str(excinfo.value)

    def test_env_variables_name_the_accounts_in_the_failure(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        monkeypatch.setenv("GMS_SEO_OWNER_USER", "env-user")
        monkeypatch.setenv("GMS_SEO_OWNER_GROUP", "env-group")

        with (
            patch.object(preflight.os, "stat", _foreign_owned_stat()),
            patch.object(preflight.os, "chown", _deny_chown),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(tmp_path, project="gomoveshift")

        assert "sudo chown env-user:env-group" in str(excinfo.value)

    def test_arguments_override_the_environment(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)
        monkeypatch.setenv("GMS_SEO_OWNER_USER", "env-user")
        monkeypatch.setenv("GMS_SEO_OWNER_GROUP", "env-group")

        with (
            patch.object(preflight.os, "stat", _foreign_owned_stat()),
            patch.object(preflight.os, "chown", _deny_chown),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(
                tmp_path,
                project="gomoveshift",
                target_user="arg-user",
                target_group="arg-group",
            )

        msg = str(excinfo.value)
        assert "sudo chown arg-user:arg-group" in msg
        assert "env-user" not in msg

    def test_arguments_alone_name_the_accounts(self, tmp_path):
        from gomoveshift.pipeline import preflight

        _make_seo_tree(tmp_path)

        with (
            patch.object(preflight.os, "stat", _foreign_owned_stat()),
            patch.object(preflight.os, "chown", _deny_chown),
            pytest.raises(RuntimeError) as excinfo,
        ):
            preflight.check_seo_ownership(
                tmp_path,
                project="gomoveshift",
                target_user="arg-user",
                target_group="arg-group",
            )

        assert "sudo chown arg-user:arg-group" in str(excinfo.value)

    def test_no_operator_account_is_baked_into_the_signature(self):
        import inspect

        from gomoveshift.pipeline import preflight

        params = inspect.signature(preflight.check_seo_ownership).parameters
        assert params["target_user"].default is None
        assert params["target_group"].default is None
