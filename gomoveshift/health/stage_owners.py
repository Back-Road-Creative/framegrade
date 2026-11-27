"""Owner records for stage roots that no backlog row owns, and their expiry release.

A stage root (``<stages root>/<name>``) normally belongs to a ``drive_backlog.yaml`` row. One kept
on purpose with no row (``mizen-remeasure``, held as evidence for framegrade#398) has nothing that
says whose it is or when it may go, so it sits on the disk until the disk is full. Each such root
gets a record in ``stage_owners.json``: owner, reason, linked issue and an expiry date. Once the
expiry has passed, ``health backlog`` appends the name to the host-cleanup release list; the
release itself is host-cleanup's rule 7c (every hold it has still applies), never done here.

:func:`remeasure_holds` is the one rule every stage release here consults (the owner-record
listing and ``gomoveshift cleanup``): a project whose queued Short may still need a route
re-measure keeps its stages, whatever else says it may go.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set

SCHEMA = "gms-stage-owners/1"
DEFAULT_OWNERS_PATH = Path(__file__).with_name("stage_owners.json")
#: The release list host-cleanup rule 7c reads (workspaces ``_ops/host-cleanup.sh``,
#: ``GMS_STAGES_RELEASE``, under ``.data/host-cleanup/``): one name per line, ``#`` comments.
#: ``release_list_path`` is the only place this name is joined to a path, and a test pins the
#: literal, so the writer here and the reader there cannot drift apart unnoticed.
STAGES_RELEASE_LIST_NAME = "gms-stages-release.list"
RELEASE_LIST_DIR = "host-cleanup"
#: Rule 7c's own name test (one plain path component), applied before a name is ever listed.
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
#: ``launch.SUPERSEDED_DIR`` (launch imports this module, so the name is repeated; a test pins it).
SUPERSEDED_DIR = "_superseded"


class OwnerRecordError(ValueError):
    """An owner file that cannot be trusted; never guessed past."""


@dataclass(frozen=True)
class Owner:
    name: str
    owner: str
    reason: str
    issue: str
    expires: date


class RemeasureHold(NamedTuple):
    delivery_id: int
    sidecar: str
    #: The recording date whose ``route_remeasure.remeasure_inputs`` the re-measure reads;
    #: ``None`` when the sidecar could not be read (held anyway: it proves nothing).
    date: Optional[str]


def remeasure_holds(db_path: Path) -> Dict[str, List[RemeasureHold]]:
    """Projects whose stages must stay, each with the rows that hold them: every queued,
    non-shadow YouTube row (``queue_repair.QUEUED_YOUTUBE_SQL``, the rows ``dispatch-queue``
    repairs) whose route record has no ``bins``. Only such a row is ever re-measured (a record
    with ``bins`` is refused as done), and the re-measure reads the project's ``_stages``; a row
    that is published, failed or cancelled is never repaired again. A row whose sidecar cannot be
    read holds too. The queue opens read-only; one that cannot be read raises, so a caller
    releases nothing."""
    import sqlite3

    from ..publishing.queue_repair import QUEUED_YOUTUBE_SQL
    from ..publishing.route_remeasure import remeasure_date

    if not Path(db_path).is_file():
        raise FileNotFoundError(f"social queue {db_path} does not exist")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(QUEUED_YOUTUBE_SQL).fetchall()
    finally:
        conn.close()
    out: Dict[str, List[RemeasureHold]] = {}
    for row in rows:
        try:
            when = remeasure_date(row)
        except Exception:  # noqa: BLE001 -- an unreadable sidecar holds: it proves nothing
            when = ""
        if when is not None:
            out.setdefault(row["project_name"], []).append(
                RemeasureHold(row["id"], str(row["sidecar_path"] or ""), when or None)
            )
    return out


def release_list_path(data_root: Path) -> Path:
    return Path(data_root) / RELEASE_LIST_DIR / STAGES_RELEASE_LIST_NAME


def _record(raw: Any) -> Owner:
    if not isinstance(raw, dict):
        raise OwnerRecordError(f"owner record is not an object: {raw!r}")
    fields: Dict[str, str] = {}
    for key in ("name", "owner", "reason", "issue", "expires"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise OwnerRecordError(f"owner record {raw.get('name')!r} has no {key}")
        fields[key] = value.strip()
    if not _NAME.match(fields["name"]):
        raise OwnerRecordError(
            f"owner record name {fields['name']!r} is not a plain stage-root name"
        )
    try:
        expires = date.fromisoformat(fields["expires"])
    except ValueError as exc:
        raise OwnerRecordError(f"owner record {fields['name']!r}: bad expires: {exc}") from exc
    return Owner(fields["name"], fields["owner"], fields["reason"], fields["issue"], expires)


def load_owners(path: Path) -> List[Owner]:
    """The records in ``path``; a missing file is none. Malformed, a duplicate name or a record
    missing a field raises ``OwnerRecordError``."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OwnerRecordError(f"cannot read {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not isinstance(doc.get("roots"), list):  # fmt: skip
        raise OwnerRecordError(f"{path} is not a {SCHEMA} file")
    owners = [_record(r) for r in doc["roots"]]
    names = [o.name for o in owners]
    if len(set(names)) != len(names):
        raise OwnerRecordError(f"{path} names a root twice")
    return owners


def expired(owners: Iterable[Owner], today: date) -> List[Owner]:
    """Records whose expiry is before ``today`` (the expiry day itself is still held)."""
    return [o for o in owners if o.expires < today]


def unowned_roots(stages_root: Path, known: Set[str], owners: Iterable[Owner]) -> List[str]:
    """Stage-root dirs under ``stages_root`` that no manifest row or store drive (``known``) and
    no owner record accounts for."""
    held = set(known) | {o.name for o in owners} | {SUPERSEDED_DIR}
    root = Path(stages_root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in held)


def release_candidates(owners: Iterable[Owner], stages_root: Path, today: date) -> List[str]:
    """Expired records whose root is still on disk."""
    return [o.name for o in expired(owners, today) if (Path(stages_root) / o.name).is_dir()]


def list_for_release(list_path: Path, owners: Sequence[Owner], today: date) -> List[str]:
    """Append each expired record's name to ``list_path`` (with a comment saying why) unless it is
    already a listed line. Returns the names appended."""
    path = Path(list_path)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    have = {ln.strip() for ln in text.splitlines()}
    out: List[str] = []
    add = ""
    for o in expired(owners, today):
        if o.name in have:
            continue
        add += f"# {today.isoformat()}: {o.name} owner record expired {o.expires.isoformat()} ({o.issue}, {o.owner}: {o.reason})\n{o.name}\n"  # fmt: skip
        out.append(o.name)
    if add:
        path.parent.mkdir(parents=True, exist_ok=True)
        sep = "" if not text or text.endswith("\n") else "\n"
        path.write_text(text + sep + add, encoding="utf-8")
    return out


def default_owners(path: Optional[Path] = None) -> List[Owner]:
    return load_owners(path or DEFAULT_OWNERS_PATH)
