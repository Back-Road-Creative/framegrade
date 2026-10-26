"""Console SEO tab: corrections-ledger read/write helpers.

Fronts the same ledger `pipeline/validation/place_corrections.py` reads for
`title_builder`/`content_gates` -- reads/writes the YAML directly
(load-modify-dump via `yaml.safe_*`) rather than reimplementing the loader,
so a save here is visible to `load_place_corrections()` immediately (the
round-trip proof for this slice).

That loader is deliberately forgiving; a *writer* cannot borrow the tolerance.
Reading `{}` from bytes it failed to parse, then dumping a fresh two-key file
over them, destroys hand-curation silently -- so this module refuses what it
cannot understand (`LedgerCorruptError`), refuses a save whose base revision
has moved (`LedgerConflictError`), refuses an edit that could never take
effect or that contradicts the rest of the ledger (`LedgerInvalidError`), and
replaces the file atomically.

Atomic replacement makes a torn file impossible; it does nothing about a lost
one. The whole read-check-mutate-replace runs under the same blocking
file-lock sidecar `content_ledger` and `pending_journal` use, because the
optimistic revision check cannot close the window it is itself inside: two
editors can both read one snapshot, both pass the check against it, and the
second replace drops the first rule with no error anywhere.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..pipeline._file_lock import make_blocking_file_lock
from ..pipeline.validation.place_corrections import default_corrections_path

#: This ledger's own reentrancy bookkeeping. A `threading.local` per lock
#: family, never shared: a ledger lock held in one thread must not mask a
#: same-named acquire for an unrelated resource in that same thread.
_LEDGER_LOCK_STATE = threading.local()

#: Blocking, reentrant, `.lock` sidecar beside the ledger. Advisory and
#: released on fd close or process death, so a crashed editor does not wedge
#: the page.
_ledger_lock = make_blocking_file_lock(_LEDGER_LOCK_STATE)


class LedgerCorruptError(RuntimeError):
    """Not a mapping, so it cannot be edited without guessing. The bytes stay."""


class LedgerConflictError(RuntimeError):
    """The ledger moved under this edit. Raised so the other edit survives."""


class LedgerInvalidError(ValueError):
    """The edit itself cannot take effect, or contradicts what is already
    here. `load_place_corrections` is deliberately forgiving -- it drops a
    blank rule and keeps only the last of two rules for the same name, both
    silently -- so a ledger can hold rows that read on the page exactly like
    working ones and do nothing, or something else. The writer refuses them
    here, upstream of the write, for the same reason it refuses corrupt
    bytes: the page cannot show what the loader quietly discarded."""


@dataclass(frozen=True)
class Ledger:
    """Lists plus `revision`, a digest of the exact bytes they came from (empty
    when there is no file). A save carries the one its form was rendered from."""

    corrections: list
    denylist: list
    revision: str


def corrections_path() -> Path:
    """``GMS_CORRECTIONS_PATH`` (read at call time, not import time, so a
    test's ``monkeypatch.setenv`` takes effect) or the pipeline default (resolved per call, so it follows
    ``GMS_DATA_HOME`` exactly as the pipeline reader does)."""
    raw = os.environ.get("GMS_CORRECTIONS_PATH")
    return Path(raw) if raw and raw.strip() else default_corrections_path()


def read(path: "Path | None" = None) -> Ledger:
    """The ledger as edit-friendly lists. Missing or empty is an empty ledger
    -- uncurated is not damage. Anything else that will not parse as a mapping
    raises `LedgerCorruptError`."""
    path = path or corrections_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return Ledger(corrections=[], denylist=[], revision="")
    try:
        loaded = yaml.safe_load(raw.decode("utf-8")) or {}
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise LedgerCorruptError(f"{path} could not be read as YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise LedgerCorruptError(
            f"{path} could not be read as a corrections ledger: "
            f"expected a mapping, found {type(loaded).__name__}"
        )
    return Ledger(
        corrections=list(loaded.get("corrections") or []),
        denylist=list(loaded.get("denylist") or []),
        revision=hashlib.sha256(raw).hexdigest()[:16],
    )


def _norm(value: str) -> str:
    """The form callers match on. `load_place_corrections` strips every side
    and `_word_re` matches case-insensitively, so "sligo " and "Sligo" are
    one name to the pipeline and must be one name here."""
    return value.strip().casefold()


def _check_correction(data: dict, wrong: str, correct: str) -> None:
    """Refuse a correction the pipeline would drop, silently override, or
    substitute into a name the metadata gate then hard-fails on."""
    if not wrong.strip() or not correct.strip():
        raise LedgerInvalidError(
            "a correction needs both a wrong name and a correct one; "
            "the pipeline drops a rule with either side blank"
        )
    if _norm(wrong) == _norm(correct):
        raise LedgerInvalidError(
            f"{wrong.strip()!r} and {correct.strip()!r} are the same name, "
            "so this rule could never change any text"
        )
    for existing in data["corrections"]:
        if not isinstance(existing, dict):
            continue
        if _norm(str(existing.get("wrong") or "")) == _norm(wrong):
            raise LedgerInvalidError(
                f"{wrong.strip()!r} is already corrected to "
                f"{str(existing.get('correct') or '').strip()!r}; the pipeline keeps only one "
                "rule per name, so edit or remove that one instead of adding a second"
            )
    for name in data["denylist"]:
        if isinstance(name, str) and _norm(name) == _norm(correct):
            raise LedgerInvalidError(
                f"{correct.strip()!r} is on the denylist, so a title corrected to it would be "
                "hard-failed by the metadata gate; remove it from the denylist first"
            )


def _check_denylist_entry(data: dict, name: str) -> None:
    """Refuse a denylist name the pipeline would drop, report twice, or that
    contradicts a correction whose result is this very name."""
    if not name.strip():
        raise LedgerInvalidError("a denylist entry needs a name")
    for existing in data["denylist"]:
        if isinstance(existing, str) and _norm(existing) == _norm(name):
            raise LedgerInvalidError(f"{existing.strip()!r} is already on the denylist")
    for correction in data["corrections"]:
        if not isinstance(correction, dict):
            continue
        if _norm(str(correction.get("correct") or "")) == _norm(name):
            raise LedgerInvalidError(
                f"{str(correction.get('wrong') or '').strip()!r} is corrected to "
                f"{name.strip()!r}; denylisting it would hard-fail every title that correction "
                "produces. Change or remove that correction first"
            )


def _write_atomic(path: Path, text: str) -> None:
    """Dump into an fsynced sibling temp file, then `os.replace` it in, so a
    failed or interrupted write leaves the previous ledger whole and no debris.
    Sibling, not `/tmp`, so the replace stays inside one filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    tmp = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _edit(path: "Path | None", expected_revision: "str | None", mutate) -> None:
    """Re-read, check the caller's base revision against the file as it is
    NOW, mutate, replace atomically -- all of it holding the ledger lock.

    The lock spans the whole read-modify-write rather than just the write.
    A second editor that gets in between the read and the replace loses
    nothing of its own; it silently destroys the first one's rule, because
    its own snapshot predates it and its validators therefore cannot see it.
    Serializing here means the second editor reads the file the first just
    wrote, so the revision check and every `_check_*` rule are deciding
    against the ledger as it actually stands.

    Blocking rather than "raise on contention": the waiting editor is a
    person who has already filled in the form, and the wait is one small
    YAML dump long. Every exit path -- a refusal from `read`, from the
    revision check, or from a validator -- leaves the lock through the
    context manager, so a refused edit never wedges the next one.
    """
    path = path or corrections_path()
    with _ledger_lock(path):
        ledger = read(path)
        if expected_revision is not None and expected_revision != ledger.revision:
            raise LedgerConflictError(
                f"{path} changed since this edit was started (expected revision "
                f"{expected_revision or 'none'}, found {ledger.revision or 'none'})"
            )
        data = {"corrections": ledger.corrections, "denylist": ledger.denylist}
        # Validated against the ledger as it is NOW, inside the same read the
        # revision check used -- a rule checked against a stale copy could still
        # collide with one that landed in between.
        mutate(data)
        _write_atomic(path, yaml.safe_dump(data, sort_keys=False))


def add_correction(
    wrong: str, correct: str, path: "Path | None" = None, expected_revision: "str | None" = None
) -> None:
    def _add(data: dict) -> None:
        _check_correction(data, wrong, correct)
        data["corrections"].append({"wrong": wrong.strip(), "correct": correct.strip()})

    _edit(path, expected_revision, _add)


def add_denylist_entry(
    name: str, path: "Path | None" = None, expected_revision: "str | None" = None
) -> None:
    def _add(data: dict) -> None:
        _check_denylist_entry(data, name)
        data["denylist"].append(name.strip())

    _edit(path, expected_revision, _add)
