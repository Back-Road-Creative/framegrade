"""Guards: shorts hours have exactly one source (Waypost's cadence), never GMS config."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import yaml

GMS = Path(__file__).resolve().parent.parent / "gomoveshift"
CONFIG_PY = GMS / "config.py"
RETIRED = (
    "shorts_publish_times_by_region",
    "shorts_slot_pool_by_region",
    "validate_shorts_pool_region_fence",
)


def _retired_tuple_lines(tree: ast.AST) -> set[int]:
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


def _hits(path: Path, skip_retired_tuple: bool = False) -> list[str]:
    """Every code reference (attribute, keyword, string, field, name, def) to a retired name."""
    tree = ast.parse(path.read_text(), filename=str(path))
    skip = _retired_tuple_lines(tree) if skip_retired_tuple else set()
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
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in RETIRED:
            name = node.name
        elif isinstance(node, ast.Name) and node.id in RETIRED:
            name = node.id
        if name and getattr(node, "lineno", 0) not in skip:
            out.append(f"{path.name}:{node.lineno} {name}")
    return out


def _all_hits() -> list[str]:
    return [
        h for p in sorted(GMS.rglob("*.py")) for h in _hits(p, skip_retired_tuple=(p == CONFIG_PY))
    ]


def test_guard_fails_if_gms_defines_shorts_times_table():
    hits = _all_hits()
    assert not hits, "retired hand-set shorts hour tables still referenced:\n" + "\n".join(hits)


def test_guard_catches_a_planted_shorts_times_table(tmp_path):
    planted = tmp_path / "planted.py"
    planted.write_text(
        "class DriveConfig:\n"
        "    shorts_publish_times_by_region: dict = {'usa': ['10:00']}\n"
        "    def validate_shorts_pool_region_fence(self): ...\n"
        "x = cfg.drive.shorts_slot_pool_by_region\n"
    )
    found = {h.split()[-1] for h in _hits(planted)}
    assert found == set(RETIRED)


def test_retired_tables_are_listed_in_retired_config_keys():
    from gomoveshift.config import RETIRED_CONFIG_KEYS

    for key in ("shorts_publish_times_by_region", "shorts_slot_pool_by_region"):
        assert key in RETIRED_CONFIG_KEYS


def test_leftover_table_keys_are_dropped_with_a_retired_warning(tmp_path, caplog):
    from gomoveshift.config import Config

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "drive": {
                    "shorts_publish_times_by_region": {"usa": ["10:00"]},
                    "shorts_slot_pool_by_region": {"usa": ["10:00"]},
                }
            }
        )
    )
    with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
        cfg = Config.load(path)
    for key in ("shorts_publish_times_by_region", "shorts_slot_pool_by_region"):
        hits = [
            r
            for r in caplog.records
            if f"RETIRED_CONFIG_KEY key={key} (retired: Waypost decides)" in r.getMessage()
        ]
        assert len(hits) == 1, (key, [r.getMessage() for r in caplog.records])
        assert not hasattr(cfg.drive, key)


def test_schedule_report_regions_are_waypost_regions():
    from gomoveshift.config import Config
    from gomoveshift.pipeline.schedule_report import _regions
    from gomoveshift.pipeline.seo_slots import waypost_cadence

    cfg = Config()
    assert _regions(cfg) == sorted(waypost_cadence(cfg).regions)
