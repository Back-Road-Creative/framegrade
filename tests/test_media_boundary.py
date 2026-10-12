"""Structural guard: the media engine must not import publishing code.

MEDIA_SET is every ``.py`` under ``gomoveshift/pipeline/`` minus the Phase-D /
publishing modules. No MEDIA_SET module may import ``seo_manager``,
``gomoveshift.publishing``, ``gomoveshift.console``, ``gomoveshift.cli`` or any
Phase-D module. Existing violations live in ALLOWLIST with a reason; a stale
row fails ``test_allowlist_has_no_stale_rows``.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1] / "gomoveshift"

WAYPOST = (
    "Waypost-side - publishing/scheduling/place resolution; moves to waypost/ "
    "later, not an engine module"
)

# Phase-D / publishing modules (paths relative to gomoveshift/pipeline/). These
# are excluded from the scan AND are forbidden import targets.
PHASE_D_FILES: dict[str, str] = {
    f: "Phase-D / publishing module"
    for f in (
        "drive_orchestrator.py",
        "drive_phases/youtube_uploader.py",
        "longform_metadata.py",
        "title_builder.py",
        "content_ledger.py",
        "auto_enqueue.py",
    )
}
PHASE_D_DIRS = ("chapters/", "site_sync/")
PHASE_D_GLOBS = ("drive_phases/phase_d*.py",)

# Waypost-side modules: excluded from the scan, but not forbidden targets.
WAYPOST_SIDE: dict[str, str] = {
    "_yt_creds.py": WAYPOST,
    "csv_generator.py": WAYPOST,
    "social_generator.py": WAYPOST,
    "journal_publisher.py": WAYPOST,
    "playlist_reconciler.py": WAYPOST,
    "playlist_resolver.py": WAYPOST,
    "schedule_constants.py": WAYPOST,
    "schedule_report.py": WAYPOST,
    "seo_slots.py": WAYPOST,
    "region_resolver.py": WAYPOST,
    "route_profile.py": WAYPOST,
    "destination_resolver.py": WAYPOST,
    "gps_enrichment.py": WAYPOST,
    "review_gate.py": WAYPOST,
    "graph_doc.py": WAYPOST,
    "drive_phases/external_systems_coordinator.py": WAYPOST,
    "drive_phases/lf_listing.py": WAYPOST,
    "photos/scheduler.py": WAYPOST,
    "validation/content_gates.py": WAYPOST,
}

EXCLUDED_FILES: dict[str, str] = {**PHASE_D_FILES, **WAYPOST_SIDE}

FORBIDDEN_PREFIXES = (
    "seo_manager",
    "gomoveshift.publishing",
    "gomoveshift.console",
    "gomoveshift.cli",
)

# Engine cross-imports still to cut: path relative to gomoveshift/ -> reason.
# Shrinking this is the goal; growing it needs a reviewer.
ALLOWLIST: dict[str, str] = {
    "pipeline/_drive_helpers.py": (
        "imports pipeline.chapters._journey_filter; cut by moving the caller into phase_a (phase_a owns the caller)"
    ),
    "pipeline/audio_profile.py": (
        "imports publishing; cut by the MediaProfile split (MediaProfile replaces brand_policy)"
    ),
    "pipeline/day_pipeline.py": (
        "imports console.settings, pipeline.chapters.generator, pipeline.chapters.marker_merger, pipeline.title_builder, publishing.brand_policy; cut by the MediaProfile split (MediaProfile carries title/chapter/brand inputs)"
    ),
    "pipeline/drive_phases/phase_bc_day_first.py": (
        "imports pipeline.chapters.nominatim_source, pipeline.longform_metadata, pipeline.title_builder; cut by the MediaProfile split (MediaProfile carries title/chapter/brand inputs)"
    ),
    "pipeline/ingest/ost_formatter.py": (
        "imports pipeline.title_builder; cut by the MediaProfile split (MediaProfile carries title/chapter/brand inputs)"
    ),
    "pipeline/orchestrator.py": (
        "imports pipeline.longform_metadata; cut by the MediaProfile split (engine side stops importing Phase D once MediaProfile carries the publish inputs)"
    ),
    "pipeline/orchestrator_fingerprint.py": (
        "imports pipeline.drive_phases.phase_d; cut by the MediaProfile split (engine side stops importing Phase D once MediaProfile carries the publish inputs)"
    ),
    "pipeline/photos/night_products.py": (
        "imports publishing.media_fingerprint; cut by moving the import out of the engine"
    ),
    "pipeline/photos/orchestrator.py": (
        "imports pipeline.content_ledger, seo_manager.analytics.theme_rotator; cut by moving the import out of the engine"
    ),
    "pipeline/post_encoding_invariants.py": (
        "imports pipeline.drive_phases.phase_d; cut by the MediaProfile split (engine side stops importing Phase D once MediaProfile carries the publish inputs)"
    ),
    "pipeline/redrive_ost.py": (
        "imports cli.logging_setup; cut by moving the import out of the engine"
    ),
    "pipeline/volume_source.py": (
        "imports seo_manager.analytics (the shorts count is Waypost's decision, read never computed); cut by the MediaProfile split (MediaProfile carries the decided count)"
    ),
    "pipeline/route_story/metadata.py": (
        "imports pipeline.longform_metadata; cut by moving the metadata step out of the engine"
    ),
    "pipeline/route_story/privacy.py": (
        "imports publishing.dispatch; cut by moving the import out of the engine"
    ),
    "pipeline/route_story/sources.py": (
        "imports pipeline.chapters.overpass_source; cut by moving the overpass source into the GPS library"
    ),
}


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root / "pipeline").as_posix()


def _is_phase_d(path: Path, root: Path) -> bool:
    rel = _rel(path, root)
    return (
        rel in PHASE_D_FILES
        or rel.startswith(PHASE_D_DIRS)
        or any(path.match(f"{(root / 'pipeline').as_posix()}/{g}") for g in PHASE_D_GLOBS)
    )


def _is_excluded(path: Path, root: Path) -> bool:
    return _is_phase_d(path, root) or _rel(path, root) in WAYPOST_SIDE


def _module_name(path: Path, root: Path) -> str:
    return ".".join(path.relative_to(root.parent).with_suffix("").parts)


def _phase_d_modules(root: Path = PKG_ROOT) -> set[str]:
    mods = {_module_name(root / "pipeline" / f, root) for f in PHASE_D_FILES}
    for p in (root / "pipeline").rglob("*.py"):
        if _is_phase_d(p, root):
            mods.add(_module_name(p, root))
    return mods


def media_set(root: Path = PKG_ROOT) -> list[Path]:
    return sorted(p for p in (root / "pipeline").rglob("*.py") if not _is_excluded(p, root))


def _is_forbidden(name: str, phase_d: set[str]) -> bool:
    if any(name == f or name.startswith(f + ".") for f in FORBIDDEN_PREFIXES):
        return True
    return any(name == m or name.startswith(m + ".") for m in phase_d)


def _imported_names(path: Path, root: Path = PKG_ROOT) -> list[tuple[int, str]]:
    """Return (lineno, absolute dotted name) for every import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    pkg = _module_name(path, root).split(".")
    if path.name != "__init__.py":
        pkg = pkg[:-1]
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, a.name) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)]
                if node.module:
                    base = [*base, *node.module.split(".")]
            else:
                base = (node.module or "").split(".")
            base_name = ".".join(base)
            out.append((node.lineno, base_name))
            # `from pkg import module` may name a forbidden submodule
            out.extend((node.lineno, f"{base_name}.{a.name}") for a in node.names)
    return out


def find_violations(root: Path = PKG_ROOT) -> list[tuple[str, int, str]]:
    phase_d = _phase_d_modules(root)
    found: list[tuple[str, int, str]] = []
    for path in media_set(root):
        rel = path.relative_to(root).as_posix()
        hits = sorted(
            (lineno, name)
            for lineno, name in _imported_names(path, root)
            if _is_forbidden(name, phase_d)
        )
        # `from pkg.mod import x` also yields `pkg.mod.x`; keep the shortest
        # name per line so one import statement is one violation.
        for lineno, name in hits:
            if not any(ln == lineno and name.startswith(other + ".") for ln, other in hits):
                found.append((rel, lineno, name))
    return sorted(set(found))


def test_media_set_is_nonempty_and_excludes_phase_d():
    rels = {_rel(p, PKG_ROOT) for p in media_set()}
    assert len(rels) > 30
    assert "drive_orchestrator.py" not in rels
    assert "drive_phases/phase_d.py" not in rels
    assert "chapters/__init__.py" not in rels
    assert "csv_generator.py" not in rels


def test_exclusions_exist():
    for f in EXCLUDED_FILES:
        assert (PKG_ROOT / "pipeline" / f).is_file(), f
    for d in PHASE_D_DIRS:
        assert (PKG_ROOT / "pipeline" / d).is_dir(), d
    for g in PHASE_D_GLOBS:
        assert list((PKG_ROOT / "pipeline").glob(g)), g


def test_media_set_imports_no_publishing_code():
    unexpected = [v for v in find_violations() if v[0] not in ALLOWLIST]
    assert unexpected == [], (
        "media engine imports publishing code (file, line, import); fix the "
        f"import or add a reasoned ALLOWLIST row: {unexpected}"
    )


def test_allowlist_has_no_stale_rows():
    violating = {v[0] for v in find_violations()}
    stale = sorted(set(ALLOWLIST) - violating)
    assert stale == [], f"allowlist rows with no violation left: {stale}"
    assert all("; cut by " in reason for reason in ALLOWLIST.values())


def _fake_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "gomoveshift"
    for rel, src in files.items():
        f = root / "pipeline" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(src, encoding="utf-8")
    return root


def test_new_cross_import_is_detected(tmp_path):
    root = _fake_tree(
        tmp_path,
        {
            "probe.py": "from gomoveshift.publishing import brand_policy\n",
            "probe_rel.py": "from .title_builder import x\n",
            "probe_seo.py": "import seo_manager.core\n",
            "probe_dir.py": "from .drive_phases import phase_d\n",
            "clean.py": "import os\nfrom .cache import Cache\n",
            "title_builder.py": "from gomoveshift.publishing import y\n",
            "drive_phases/phase_d.py": "import seo_manager\n",
        },
    )
    found = {v[0]: v[2] for v in find_violations(root)}
    assert found == {
        "pipeline/probe.py": "gomoveshift.publishing",
        "pipeline/probe_rel.py": "gomoveshift.pipeline.title_builder",
        "pipeline/probe_seo.py": "seo_manager.core",
        "pipeline/probe_dir.py": "gomoveshift.pipeline.drive_phases.phase_d",
    }
    # None of the probes is allowlisted, so the real guard would go red.
    assert not set(found) & set(ALLOWLIST)
