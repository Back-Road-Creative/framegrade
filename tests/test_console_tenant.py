"""A runtime stamp file crosses the container-mount boundary that a
path comparison alone can't -- see gomoveshift/console/tenant.py's module
docstring for why this exists and tests/test_publishing_brand_isolation.py
for the isolation contract it protects."""

from __future__ import annotations

import os

import pytest

from gomoveshift.console import tenant


def test_fresh_storage_is_claimed_and_stamped(tmp_path):
    storage = tmp_path / "seo"
    storage.mkdir()

    tenant.claim_storage(storage, "gomoveshift")

    stamp = storage / ".waypost-tenant"
    assert stamp.exists()
    assert stamp.read_text().strip() == "gomoveshift"


def test_missing_storage_directory_is_created_for_a_fresh_tenant(tmp_path):
    storage = tmp_path / "does-not-exist-yet" / "seo"

    tenant.claim_storage(storage, "gomoveshift")

    assert (storage / ".waypost-tenant").read_text().strip() == "gomoveshift"


def test_same_brand_restart_is_accepted_without_rewriting_the_stamp(tmp_path, monkeypatch):
    storage = tmp_path / "seo"
    storage.mkdir()
    tenant.claim_storage(storage, "gomoveshift")
    stamp = storage / ".waypost-tenant"
    before_mtime = stamp.stat().st_mtime_ns

    def _boom(*_a, **_k):
        raise AssertionError("stamp should not be rewritten on a same-brand restart")

    monkeypatch.setattr(os, "replace", _boom)
    tenant.claim_storage(storage, "gomoveshift")

    assert stamp.stat().st_mtime_ns == before_mtime
    assert stamp.read_text().strip() == "gomoveshift"


def test_a_different_brand_is_refused_and_nothing_is_mutated(tmp_path):
    storage = tmp_path / "seo"
    storage.mkdir()
    tenant.claim_storage(storage, "gomoveshift")

    with pytest.raises(tenant.TenantMismatchError) as excinfo:
        tenant.claim_storage(storage, "examplebrand")

    message = str(excinfo.value)
    assert "gomoveshift" in message
    assert "examplebrand" in message
    stamp = storage / ".waypost-tenant"
    assert stamp.read_text().strip() == "gomoveshift"
    assert not (storage / ".waypost-tenant.tmp").exists()


def test_damaged_stamp_is_treated_as_unknown_and_refused(tmp_path):
    storage = tmp_path / "seo"
    storage.mkdir()
    (storage / ".waypost-tenant").write_text("   \n")

    with pytest.raises(tenant.TenantMismatchError):
        tenant.claim_storage(storage, "gomoveshift")


def test_atomic_write_leaves_no_tmp_residue_on_failure(tmp_path, monkeypatch):
    storage = tmp_path / "seo"
    storage.mkdir()

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", _boom)

    with pytest.raises(OSError):
        tenant.claim_storage(storage, "gomoveshift")

    assert list(storage.iterdir()) == []
