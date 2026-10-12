"""Contract test for ``tools.style_groundtruth_score.rank``.

This is a mechanics test, not an evaluation of selector quality. Each row below
is a hand-written scene profile paired with the exact ranking ``rank`` returns
for it today, so an unintended change to the scoring weights, the ``natural``
anchor, or the declaration-order tie-break is noticed. The expected rankings were
recorded from the shipped selector; they say what it does, not that it is right.
Judging whether the selector picks good styles needs independent labels, which
``tools/style_groundtruth_judge.py`` and ``tools/style_groundtruth_score.py``
produce from a real capture corpus.

``rank`` is the only surface exercised here -- nothing goes through
``_distinct_styled``, ``develop_variants`` or rendering. Hermetic: no corpus on
disk, no rendering, no network.

``_scene`` filters to the fields the dataclass still carries, mirroring the
scorer's ``_SCENE_FIELDS`` filter, so a field removed from ``SceneProfile`` does
not break the fixture.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gomoveshift.pipeline.photos.scene import SceneProfile  # noqa: E402
from tools.style_groundtruth_score import rank  # noqa: E402

_BASE: dict[str, Any] = {
    "exposure_class": "normal",
    "sat_profile": "normal",
    "dynamic_range": "normal",
    "warmth": "neutral",
    "haze": "clear",
    "is_near_monochrome": False,
}

_DEFAULT_ORDER = ["natural", "vivid", "moody", "golden", "mono", "soft"]

#: (scene id, scene overrides on _BASE, exact expected ``rank`` output).
RANK_CASES: list[tuple[str, dict[str, Any], list[str]]] = [
    ("neutral", {}, _DEFAULT_ORDER),
    ("foggy", {"haze": "foggy"}, ["natural", "moody", "mono", "golden", "soft", "vivid"]),
    ("warm", {"warmth": "warm"}, ["natural", "golden", "soft", "vivid", "moody", "mono"]),
    ("cool", {"warmth": "cool"}, _DEFAULT_ORDER),
    (
        "bright",
        {"exposure_class": "bright"},
        ["natural", "vivid", "soft", "moody", "golden", "mono"],
    ),
    (
        "low_key",
        {"exposure_class": "low_key"},
        ["natural", "moody", "vivid", "golden", "mono", "soft"],
    ),
    (
        "near_monochrome",
        {"is_near_monochrome": True},
        ["natural", "mono", "moody", "golden", "soft", "vivid"],
    ),
    ("hazy", {"haze": "hazy"}, ["natural", "moody", "vivid", "golden", "mono", "soft"]),
    (
        "muted_saturation",
        {"sat_profile": "muted"},
        ["natural", "soft", "moody", "golden", "mono", "vivid"],
    ),
    (
        "warm_rich_saturation",
        {"warmth": "warm", "sat_profile": "rich"},
        ["natural", "golden", "vivid", "soft", "moody", "mono"],
    ),
    # ``rank`` does not read dynamic_range, so no case varies it.
]


def _scene(raw: dict[str, Any]) -> SceneProfile:
    keep = {f.name for f in fields(SceneProfile)}
    return SceneProfile(**{k: v for k, v in raw.items() if k in keep})


@pytest.mark.parametrize(
    ("name", "overrides", "expected"), RANK_CASES, ids=[c[0] for c in RANK_CASES]
)
def test_rank_returns_the_recorded_ordering(
    name: str, overrides: dict[str, Any], expected: list[str]
) -> None:
    assert rank(_scene({**_BASE, **overrides})) == expected, (
        f"rank() output changed for the {name!r} scene -- if the scoring change is "
        "deliberate, re-record the expected ordering"
    )


@pytest.mark.parametrize(
    ("name", "overrides", "_expected"), RANK_CASES, ids=[c[0] for c in RANK_CASES]
)
def test_rank_is_natural_first_and_covers_every_style_once(
    name: str, overrides: dict[str, Any], _expected: list[str]
) -> None:
    ranking = rank(_scene({**_BASE, **overrides}))
    assert ranking[0] == "natural"
    assert sorted(ranking) == sorted(_DEFAULT_ORDER)
