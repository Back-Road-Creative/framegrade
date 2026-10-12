"""Guards: shorts volume has exactly one source (Waypost, via ``volume_source``)."""

from __future__ import annotations

import ast
from pathlib import Path

GMS = Path(__file__).resolve().parent.parent / "gomoveshift"
CONFIG_PY = GMS / "config.py"
RETIRED = ("shorts_per_lf_count", "lf_length_min_n", "lf_length_bucket_edges_sec")


def _retired_tuple_lines(tree: ast.AST) -> set[int]:
    """Line span of the one module-level ``RETIRED_CONFIG_KEYS = (...)`` assignment."""
    spans: set[int] = set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "RETIRED_CONFIG_KEYS" for t in targets):
            spans.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return spans


def _hits(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    skip = _retired_tuple_lines(tree) if path == CONFIG_PY else set()
    out = []
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute) and node.attr in RETIRED:
            name = node.attr
        elif isinstance(node, ast.keyword) and node.arg in RETIRED:
            name = node.arg
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in RETIRED:
                name = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in RETIRED:
                name = node.target.id
        elif isinstance(node, ast.Name) and node.id in RETIRED:
            name = node.id
        if name and getattr(node, "lineno", 0) not in skip:
            out.append(f"{path.relative_to(GMS.parent)}:{node.lineno} {name}")
    return out


def test_no_shorts_per_lf_default_or_lf_length_min_n_in_gms():
    hits = [h for p in sorted(GMS.rglob("*.py")) for h in _hits(p)]
    assert not hits, "retired hand-set volume keys still referenced:\n" + "\n".join(hits)


def test_min_shorts_per_lf_sourced_from_volume_policy_floor():
    from gomoveshift.config import DriveConfig
    from seo_manager.analytics import volume_policy

    assert DriveConfig().min_shorts_per_lf == volume_policy.SHORTS_FLOOR

    tree = ast.parse(CONFIG_PY.read_text())
    drive = next(
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DriveConfig"
    )
    field = next(
        n
        for n in drive.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "min_shorts_per_lf"
    )
    assert not isinstance(field.value, ast.Constant), (
        f"config.py:{field.lineno} min_shorts_per_lf default is a literal; "
        "it must come from volume_policy.SHORTS_FLOOR"
    )
