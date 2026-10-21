"""Structural guard: platform clients are imported only inside ``gomoveshift/publishing/``.

Metadata and publishing go through Waypost (``seo_manager``) and the
``publishing`` package that wraps the platform APIs. No module outside
``gomoveshift/publishing/`` may import a platform client (the YouTube, Meta,
TikTok and photo adapters, their auth helpers, ``photo_graph_client``) directly.
Existing crossings live in ALLOWLIST, one row per (module, client) with a
reason; a new crossing fails ``test_no_new_platform_client_imports`` and a row
that no longer crosses fails ``test_allowlist_has_no_stale_rows``.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1] / "gomoveshift"

CLIENTS = frozenset(
    {
        "facebook_adapter",
        "instagram_adapter",
        "meta_auth",
        "meta_comments",
        "photo_adapter",
        "photo_graph_client",
        "tiktok_adapter",
        "tiktok_auth",
        "youtube_adapter",
        "youtube_comments",
        "youtube_uploads_listing",
    }
)

DISPATCH = (
    "the `dispatch-queue` CLI drains the publishing delivery queue: it builds "
    "the handler registry, runs the TikTok pack reconcile and the first-comment "
    "drains, so it wires the adapters it hands to publishing/dispatch"
)
UPLOADS_READ = "read-only walk of the channel's uploads listing; publishes and writes nothing"
METRICS_AUTH = "loads platform credentials to read analytics; publishes and writes nothing"
RECOVER = (
    "the `queue recover-terminal` CLI (dispatch-queue's operator sibling) wires the "
    "Instagram adapter for a read-only container-status probe before it re-arms a "
    "failed_permanent row; it publishes nothing itself"
)

ALLOWLIST: dict[tuple[str, str], str] = {
    **{
        ("cli/handlers/dispatch_queue.py", c): DISPATCH
        for c in (
            "facebook_adapter",
            "instagram_adapter",
            "meta_auth",
            "meta_comments",
            "photo_adapter",
            "photo_graph_client",
            "tiktok_adapter",
            "tiktok_auth",
            "youtube_adapter",
            "youtube_comments",
        )
    },
    ("cli/handlers/queue_recover.py", "instagram_adapter"): RECOVER,
    ("cli/handlers/queue_recover.py", "meta_auth"): RECOVER,
    ("cli/handlers/queue_recover.py", "youtube_uploads_listing"): UPLOADS_READ,
    ("pipeline/content_ledger.py", "youtube_uploads_listing"): UPLOADS_READ,
    ("tracking/channel_enrol.py", "youtube_uploads_listing"): UPLOADS_READ,
    ("tracking/collector.py", "meta_auth"): METRICS_AUTH,
    ("tracking/collector.py", "tiktok_auth"): METRICS_AUTH,
}


def _imported_modules(path: Path) -> set[str]:
    """Absolute dotted names a module imports (relative imports resolved)."""
    rel = path.relative_to(PKG_ROOT.parent).with_suffix("")
    package = list(rel.parts[:-1])
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package[: len(package) - (node.level - 1)]
                base = ".".join(parent + ([base] if base else []))
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return out


def _crossings() -> set[tuple[str, str]]:
    found = set()
    for path in sorted(PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(PKG_ROOT).as_posix()
        if rel.startswith("publishing/"):
            continue
        for mod in _imported_modules(path):
            parts = mod.split(".")
            if parts[:2] == ["gomoveshift", "publishing"] and len(parts) > 2:
                if parts[2] in CLIENTS:
                    found.add((rel, parts[2]))
    return found


def test_clients_exist_under_publishing():
    missing = [c for c in CLIENTS if not (PKG_ROOT / "publishing" / f"{c}.py").exists()]
    assert not missing, f"platform client list names modules that are gone: {missing}"


def test_no_new_platform_client_imports():
    new = sorted(_crossings() - ALLOWLIST.keys())
    assert not new, (
        "platform client imported outside gomoveshift/publishing/ (route it through "
        "publishing/ or seo_manager):\n" + "\n".join(f"{m} -> {c}" for m, c in new)
    )


def test_allowlist_has_no_stale_rows():
    stale = sorted(ALLOWLIST.keys() - _crossings())
    assert not stale, "ALLOWLIST rows that no longer cross; delete them:\n" + "\n".join(
        f"{m} -> {c}" for m, c in stale
    )


def test_every_allowlist_row_has_a_reason():
    assert all(r.strip() for r in ALLOWLIST.values())


def test_scanner_resolves_relative_and_absolute_imports():
    mods = _imported_modules(PKG_ROOT / "tracking" / "channel_enrol.py")
    assert "gomoveshift.publishing.youtube_uploads_listing" in mods
    mods = _imported_modules(PKG_ROOT / "tracking" / "collector.py")
    assert "gomoveshift.publishing.meta_auth" in mods
