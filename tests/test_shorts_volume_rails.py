"""Shorts volume rails: Waypost decides, 7 is the seed, 14 is never reachable.

The number GMS plans per long-form comes only from ``pipeline/volume_source``
(Waypost's ``cadence_decisions`` log); ``DriveConfig.min_shorts_per_lf`` is
Waypost's ``SHORTS_FLOOR`` (``config.py`` ``_shorts_floor``). 14 shorts/LF is
the rate the YouTube channel was terminated at on 2026-08-27; Waypost names it
``ENFORCEMENT_SHORTS_PER_LF`` and its rails keep every decision below it.
These tests pin that through GMS's dependency, so a Waypost pin bump that
loosens a rail, or a literal 14 re-added in GMS, fails here.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml
from seo_manager.analytics import cadence_decisions as cd
from seo_manager.analytics import volume_policy as policy

from gomoveshift.pipeline import volume_source

SVC = Path(__file__).resolve().parent.parent
GMS = SVC / "gomoveshift"
ENFORCED = 14
# A name that carries a shorts-per-LF count (not durations, gaps, times).
VOLUME_NAME = re.compile(r"shorts_(per_lf|target|target_count|count)$")


def test_seed_is_seven_and_comes_from_waypost(tmp_path):
    from gomoveshift.config import DriveConfig

    assert policy.SHORTS_FLOOR == 7
    assert DriveConfig().min_shorts_per_lf == 7
    seed = volume_source.shorts_per_lf(tmp_path / volume_source.DB_NAME)
    assert (seed.value, seed.source) == (7, "seed")


def test_waypost_ceiling_stays_below_the_enforced_rate():
    assert policy.ENFORCEMENT_SHORTS_PER_LF == ENFORCED
    assert policy.SHORTS_FLOOR <= policy.SHORTS_CEILING < ENFORCED


@pytest.mark.parametrize("value", [ENFORCED, policy.SHORTS_CEILING + 1])
def test_waypost_refuses_to_record_a_shorts_count_above_its_ceiling(tmp_path, value):
    db = tmp_path / volume_source.DB_NAME
    with pytest.raises(cd.DecisionRejected):
        cd.append(
            db,
            knob=volume_source.KNOB,
            region=None,
            value=value,
            prev=policy.SHORTS_FLOOR,
            status="ok",
            source="policy",
            reason_code="TEST",
            basis="test",
            applied_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
        )
    assert volume_source.shorts_per_lf(db).value == policy.SHORTS_FLOOR


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_14_volume_sites(path: Path, root: Path = SVC) -> list[str]:
    """``name = 14``, ``name: int = 14``, ``f(name=14)``, ``{"name": 14}`` for a volume name."""
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []

    def hit(name: str | None, value: ast.AST, line: int) -> None:
        if (
            name
            and VOLUME_NAME.search(name)
            and isinstance(value, ast.Constant)
            and type(value.value) is int
            and value.value == ENFORCED
        ):
            out.append(f"{path.relative_to(root)}:{line} {name} = {ENFORCED}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                hit(_name_of(t), node.value, node.lineno)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            hit(_name_of(node.target), node.value, node.lineno)
        elif isinstance(node, ast.keyword):
            hit(node.arg, node.value, node.value.lineno)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is not None:
                    hit(_name_of(k), v, v.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args + node.args.kwonlyargs
            defaults = [None] * (len(node.args.args) - len(node.args.defaults))
            defaults += list(node.args.defaults) + list(node.args.kw_defaults)
            for a, d in zip(args, defaults):
                if d is not None:
                    hit(a.arg, d, d.lineno)
    return out


def _yaml_14_volume_keys(node, where: str) -> list[str]:
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and VOLUME_NAME.search(k) and v == ENFORCED:
                out.append(f"{where}: {k}: {ENFORCED}")
            out += _yaml_14_volume_keys(v, where)
    elif isinstance(node, list):
        for v in node:
            out += _yaml_14_volume_keys(v, where)
    return out


def test_no_literal_14_shorts_count_in_gms_code():
    hits = [h for p in sorted(GMS.rglob("*.py")) for h in _literal_14_volume_sites(p)]
    assert not hits, "a literal 14 shorts/LF count is back:\n" + "\n".join(hits)


def test_no_14_shorts_count_in_gms_yaml_config():
    files = [
        SVC / "config.yaml",
        *sorted(GMS.rglob("*.yaml")),
    ]
    hits = []
    for f in files:
        hits += _yaml_14_volume_keys(yaml.safe_load(f.read_text()), str(f.relative_to(SVC)))
    assert not hits, "a 14 shorts/LF count is back in config:\n" + "\n".join(hits)


def test_scanner_catches_each_literal_14_form(tmp_path):
    src = tmp_path / "probe.py"
    src.write_text(
        "shorts_per_lf = 14\n"
        "class C:\n    shorts_target: int = 14\n"
        "f(shorts_target=14)\n"
        "d = {'shorts_per_lf': 14}\n"
        "def g(shorts_target_count=14): ...\n"
        "shorts_target_duration_sec = 14\n"
        "shorts_per_lf_max = 14\n"
    )
    lines = sorted(h.split(":")[1].split()[0] for h in _literal_14_volume_sites(src, tmp_path))
    assert lines == ["1", "3", "4", "5", "6"]
