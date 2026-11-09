"""Duplicate detection across folders and runs.

Name-keyed dedupe (``ingest._dedupe_raw_pairs``) misses a renamed copy of a
frame. Here a frame with a capture identity (body serial plus shutter count, or
serial plus sub-second capture time) is keyed on it, so a renamed or re-offloaded
copy collapses to one keeper; a frame without one is keyed on its sha256. Only
those identity-less files are hashed. A sqlite ledger remembers keys between
runs. ``dedupe`` never writes it: the caller records after a successful run.
Originals are only ever opened read-only.
"""

from __future__ import annotations

import datetime
import hashlib
import sqlite3
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional

from .capture_meta import CaptureRecord

RAW_EXTENSIONS = frozenset({".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".dng"})
SCHEMA_VERSION = 1
_SQL_CHUNK = 500


class DupTier(str, Enum):
    SAME_SHOT = "same_shot"
    EXACT_COPY = "exact_copy"
    SEEN_BEFORE = "seen_before"


@dataclass(frozen=True)
class DupHit:
    duplicate: Path
    kept: Optional[Path]
    tier: DupTier
    key: str
    reason: str
    seen_run: Optional[str] = None


@dataclass(frozen=True)
class DedupeResult:
    unique: tuple[Path, ...]
    duplicates: tuple[DupHit, ...]
    keys: Mapping[Path, str]


def capture_identity(rec: CaptureRecord) -> Optional[str]:
    if not rec.serial:
        return None
    head = (
        f"shot:{(rec.make or '').strip().lower()}|{(rec.model or '').strip().lower()}|{rec.serial}"
    )
    if rec.shutter_count is not None:
        return f"{head}|sc:{rec.shutter_count}"
    if rec.datetime_original and rec.subsec:
        return f"{head}|t:{rec.datetime_original}.{rec.subsec}"
    return None


def sha256_file(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _keeper_order(path: Path):
    return (path.suffix.lower() not in RAW_EXTENSIONS, len(str(path)), str(path))


def dedupe(
    paths,
    records: Mapping[Path, CaptureRecord],
    *,
    ledger: "Optional[Ledger]" = None,
    reprocess: bool = False,
    hasher=sha256_file,
) -> DedupeResult:
    paths = [Path(p) for p in paths]
    key_of: dict[Path, str] = {}
    tier_of: dict[str, DupTier] = {}
    for path in paths:
        rec = records.get(path)
        identity = capture_identity(rec) if rec is not None else None
        if identity is not None:
            key_of[path], tier_of[identity] = identity, DupTier.SAME_SHOT
        else:
            key = f"sha256:{hasher(path)}"
            key_of[path], tier_of[key] = key, DupTier.EXACT_COPY

    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(key_of[path], []).append(path)
    keeper = {key: min(members, key=_keeper_order) for key, members in groups.items()}

    duplicates: list[DupHit] = []
    survivors: list[Path] = []
    for path in paths:
        key = key_of[path]
        if keeper[key] == path:
            survivors.append(path)
            continue
        tier = tier_of[key]
        duplicates.append(
            DupHit(
                duplicate=path,
                kept=keeper[key],
                tier=tier,
                key=key,
                reason=(
                    "same capture identity as the kept frame"
                    if tier is DupTier.SAME_SHOT
                    else "byte-identical to the kept file"
                ),
            )
        )

    seen = ledger.seen([key_of[p] for p in survivors]) if ledger and not reprocess else {}
    unique: list[Path] = []
    for path in survivors:
        key = key_of[path]
        if key in seen:
            duplicates.append(
                DupHit(
                    duplicate=path,
                    kept=None,
                    tier=DupTier.SEEN_BEFORE,
                    key=key,
                    reason=f"already processed in run {seen[key]}",
                    seen_run=seen[key],
                )
            )
        else:
            unique.append(path)
    order = {p: i for i, p in enumerate(paths)}
    duplicates.sort(key=lambda hit: order[hit.duplicate])
    return DedupeResult(
        unique=tuple(unique),
        duplicates=tuple(duplicates),
        keys={p: key_of[p] for p in unique},
    )


class Ledger:
    """Cross-run memory of processed shots and per-file scores (stdlib sqlite3)."""

    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        if not db_path.parent.is_dir():
            raise FileNotFoundError(f"ledger directory does not exist: {db_path.parent}")
        self._db = sqlite3.connect(db_path)
        try:
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError(f"ledger {db_path} is newer (schema {version}) than this build")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS shots (key TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
                "source_path TEXT NOT NULL, sha256 TEXT, first_seen_utc TEXT NOT NULL)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS scores (sha256 TEXT, name TEXT, value REAL, "
                "PRIMARY KEY(sha256, name))"
            )
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._db.commit()
        except BaseException:
            self._db.close()
            raise

    def seen(self, keys) -> dict[str, str]:
        keys = list(keys)
        found: dict[str, str] = {}
        for start in range(0, len(keys), _SQL_CHUNK):
            chunk = keys[start : start + _SQL_CHUNK]
            marks = ",".join("?" * len(chunk))
            # only "?" placeholders are interpolated; the keys are bound
            cursor = self._db.execute(
                f"SELECT key, run_id FROM shots WHERE key IN ({marks})",  # noqa: S608
                chunk,
            )
            found.update(cursor.fetchall())
        return found

    def record(self, run_id: str, rows) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        self._db.executemany(
            "INSERT OR IGNORE INTO shots (key, run_id, source_path, sha256, first_seen_utc) "
            "VALUES (?, ?, ?, ?, ?)",
            [(key, run_id, source, sha, now) for key, source, sha in rows],
        )
        self._db.commit()

    def get_score(self, sha256: str, name: str) -> Optional[float]:
        row = self._db.execute(
            "SELECT value FROM scores WHERE sha256 = ? AND name = ?", (sha256, name)
        ).fetchone()
        return row[0] if row else None

    def put_score(self, sha256: str, name: str, value: float) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO scores (sha256, name, value) VALUES (?, ?, ?)",
            (sha256, name, float(value)),
        )
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
