"""Two editors cannot lose each other's corrections.

`seo._edit` reads the ledger, checks the caller's base revision, mutates and
replaces atomically. Atomic replacement makes a *torn* file impossible; it does
nothing about a *lost* one. Nothing held the file between the read and the
replace, so two console workers -- the shape an instance may yet grow
into, and already reachable from two tabs against one worker -- could both read
the same snapshot, both pass the revision check against it, and the second
`os.replace` would drop the first editor's rule with no error anywhere.

The optimistic revision check narrows that window; it cannot close it, because
it is itself inside the window. The lock closes it, and it is the same
`pipeline._file_lock.make_blocking_file_lock` that `content_ledger` and
`pending_journal` already use for exactly this bug class.
"""

import contextlib
import threading

import pytest
import yaml

from gomoveshift.console import seo


def _ledger(path):
    return yaml.safe_load(path.read_text()) or {}


def test_two_concurrent_saves_keep_both_edits(tmp_path, monkeypatch):
    """A barrier planted just after each editor's ``read`` forces the
    interleave that used to drop an edit. With the lock held across
    read->mutate->replace the second editor serializes behind the first and
    both rules persist; the barrier simply times out."""
    path = tmp_path / "corrections.yaml"
    barrier = threading.Barrier(2, timeout=3.0)
    real_read = seo.read

    def slow_read(p=None):
        ledger = real_read(p)
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait()
        return ledger

    monkeypatch.setattr(seo, "read", slow_read)

    a = threading.Thread(target=seo.add_correction, args=("Slygo", "Sligo", path))
    b = threading.Thread(target=seo.add_correction, args=("Kilarney", "Killarney", path))
    a.start()
    b.start()
    a.join(timeout=30)
    b.join(timeout=30)
    assert not a.is_alive() and not b.is_alive(), "editor thread deadlocked"

    wrong = {c["wrong"] for c in _ledger(path)["corrections"]}
    assert wrong == {"Slygo", "Kilarney"}, f"lost update: only {sorted(wrong)} survived"


def test_a_correction_and_a_denylist_entry_do_not_overwrite_each_other(tmp_path, monkeypatch):
    """The two writers are different functions against one file, which is
    what makes the loss quiet: neither validator can see the other's edit,
    so nothing refuses and nothing warns."""
    path = tmp_path / "corrections.yaml"
    barrier = threading.Barrier(2, timeout=3.0)
    real_read = seo.read

    def slow_read(p=None):
        ledger = real_read(p)
        with contextlib.suppress(threading.BrokenBarrierError):
            barrier.wait()
        return ledger

    monkeypatch.setattr(seo, "read", slow_read)

    a = threading.Thread(target=seo.add_correction, args=("Slygo", "Sligo", path))
    b = threading.Thread(target=seo.add_denylist_entry, args=("Atlantis", path))
    a.start()
    b.start()
    a.join(timeout=30)
    b.join(timeout=30)

    data = _ledger(path)
    assert [c["wrong"] for c in data["corrections"]] == ["Slygo"]
    assert data["denylist"] == ["Atlantis"]


def test_a_single_save_is_not_reentrant_deadlocked(tmp_path):
    """One edit takes the lock once. Pinned because the lock is reentrant
    per thread and a future nested acquire must not self-deadlock -- flock is
    tied to the open file description, so a naive second acquire in the same
    thread would block forever on itself."""
    path = tmp_path / "corrections.yaml"
    done = threading.Event()

    def edit():
        seo.add_correction("Slygo", "Sligo", path)
        done.set()

    threading.Thread(target=edit).start()
    assert done.wait(timeout=10), "a single save deadlocked on its own lock"
    assert [c["wrong"] for c in _ledger(path)["corrections"]] == ["Slygo"]


def test_a_refused_edit_releases_the_lock(tmp_path):
    """A validation refusal is the common case on this page, not the rare
    one. If it left the lock held, the next edit from any worker would hang
    with no error to explain it."""
    path = tmp_path / "corrections.yaml"
    seo.add_correction("Slygo", "Sligo", path)

    with pytest.raises(seo.LedgerInvalidError):
        seo.add_correction("Slygo", "Sligo Town", path)

    seo.add_correction("Kilarney", "Killarney", path)
    assert {c["wrong"] for c in _ledger(path)["corrections"]} == {"Slygo", "Kilarney"}


def test_a_conflicting_edit_releases_the_lock(tmp_path):
    """Same for the revision check, which raises before any mutation."""
    path = tmp_path / "corrections.yaml"
    seo.add_correction("Slygo", "Sligo", path)

    with pytest.raises(seo.LedgerConflictError):
        seo.add_correction("Kilarney", "Killarney", path, expected_revision="stale")

    seo.add_correction("Kilarney", "Killarney", path)
    assert {c["wrong"] for c in _ledger(path)["corrections"]} == {"Slygo", "Kilarney"}


def test_a_corrupt_ledger_releases_the_lock(tmp_path):
    """The refusal happens inside `read`, which is inside the lock."""
    path = tmp_path / "corrections.yaml"
    path.write_text("- not a mapping\n")

    with pytest.raises(seo.LedgerCorruptError):
        seo.add_correction("Slygo", "Sligo", path)

    assert path.read_text() == "- not a mapping\n"
    path.unlink()
    seo.add_correction("Slygo", "Sligo", path)
    assert [c["wrong"] for c in _ledger(path)["corrections"]] == ["Slygo"]


def test_the_lock_sidecar_sits_beside_the_ledger(tmp_path):
    """Beside the file, not in a temp dir: two processes only serialize if
    they name the same lock, and the sidecar has to live on the same
    filesystem the atomic replace already requires."""
    path = tmp_path / "corrections.yaml"

    seo.add_correction("Slygo", "Sligo", path)

    assert (tmp_path / "corrections.yaml.lock").exists()


def test_the_lock_does_not_change_what_a_save_writes(tmp_path):
    """The happy path is byte-identical to before: same two keys, same
    insertion order, no lock bookkeeping leaking into the ledger."""
    path = tmp_path / "corrections.yaml"

    seo.add_correction("Slygo", "Sligo", path)
    seo.add_denylist_entry("Atlantis", path)

    assert yaml.safe_load(path.read_text()) == {
        "corrections": [{"wrong": "Slygo", "correct": "Sligo"}],
        "denylist": ["Atlantis"],
    }
