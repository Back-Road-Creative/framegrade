"""The photo docs cannot claim a count the code disagrees with.

Six numbers across ``docs/pipeline-internals.md`` and the photo modules described
an older pipeline, and four of the six went stale inside three weeks. They were
later corrected. Every one was hand-written prose about a table the code already
publishes, so nothing could notice. This reads BOTH sides live — the claimed
value out of the file or the docstring by regex, the true value off the registry
itself. It hand-lists no registry: a private copy of "there are 9 masks" here
would be the same failure one layer down.

What is pinned, and what deliberately is not:

* COUNTS are pinned only where the number is cheap to keep true. ``STYLES`` (a
  new style is a rare, deliberate event, and "all eleven are registered" is a
  completeness claim a reader relies on) and ``LOOK_ALIASES`` (which is spelled
  out in THREE files, and went stale in all three at once when
  ``split_tone_midtone`` landed).
* Counts are NOT pinned for ``LOOK_KNOBS``, the ``DevelopOptions``-only knob set
  or the list-knob gate. Each moves with every knob PR — the look surface went
  ~20 to 42 in three weeks — so a pin there is a line every unrelated PR learns
  to bump mechanically, which is a worse guard than none. Those three were
  rewritten as derivations, which is the real fix, and
  ``tests/test_photo_knob_registry.py`` already pins the tables themselves
  against ``PHOTO_KNOBS``. Sub-group prose ("the after-dark four", "the ARTIFACT
  three") is not pinned either: no live structure publishes those groupings, so
  checking them would mean hand-listing them here.
* NAMES are pinned instead, for every registry on this surface. That check
  scales: it fires when a mask, look, style or look knob lands undocumented —
  which the house rule already forbids — and never fires on a reword or a count.
"""

import re
from pathlib import Path

import pytest

from gomoveshift import config as c
from gomoveshift.pipeline.photos import develop, looks, masks, styles

# The committed doc in the SOURCE tree, for the reason tests/test_pipeline_graph_doc.py
# gives: CI installs non-editable and docs/ is not shipped into site-packages.
# docs/pipeline-internals.md is a generated index now (tools/build_internals_index.py);
# the photo pipeline module content lives in its own split-out file.
DOC = Path(__file__).resolve().parents[1] / "docs" / "pipeline" / "83-photo-pipeline.md"

CARDINALS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def _flat(text: str) -> str:
    """Whitespace-folded, so a claim that wraps across lines still reads as one."""
    return re.sub(r"\s+", " ", text)


def _doc() -> str:
    return _flat(DOC.read_text(encoding="utf-8"))


def _photo_section() -> str:
    # DOC is now the split-out module file and holds exactly this one section,
    # so there is no trailing "^## " boundary -- match to end of file too.
    body = re.search(
        r"^## Photo pipeline\b.*?(?=^## |\Z)", DOC.read_text(encoding="utf-8"), re.S | re.M
    )
    assert body, f"{DOC.name} has no '## Photo pipeline' section to check claims against"
    return body.group(0)


# (where the claim lives, how to read that text, the pattern capturing the claimed
# number, what it counts, the live collection it must be the size of).
COUNT_CLAIMS = (
    (
        "docs/pipeline/83-photo-pipeline.md",
        _doc,
        r"(\w+) pure transforms",
        "photos.styles.STYLES",
        lambda: set(styles.STYLES),
    ),
    (
        "docs/pipeline/83-photo-pipeline.md",
        _doc,
        r"All (\w+) are registered in `STYLES`",
        "photos.styles.STYLES",
        lambda: set(styles.STYLES),
    ),
    (
        "docs/pipeline/83-photo-pipeline.md",
        _doc,
        r"\bthe `LOOK_ALIASES` (\w+)",
        "config.LOOK_ALIASES",
        lambda: set(c.LOOK_ALIASES),
    ),
    (
        "photos/looks.py module docstring",
        lambda: _flat(looks.__doc__ or ""),
        r"\(the (\w+) ``config\.LOOK_ALIASES`` maps\)",
        "config.LOOK_ALIASES",
        lambda: set(c.LOOK_ALIASES),
    ),
    (
        "photos/develop.py look_variant_options docstring",
        lambda: _flat(develop.look_variant_options.__doc__ or ""),
        r"``LOOK_ALIASES`` maps the (\w+) that differ",
        "config.LOOK_ALIASES",
        lambda: set(c.LOOK_ALIASES),
    ),
)

NAME_CLAIMS = (
    ("photos.masks.MASKS", lambda: set(masks.MASKS)),
    ("photos.looks.BUILTIN_LOOKS", lambda: set(looks.BUILTIN_LOOKS)),
    ("photos.styles.STYLES", lambda: set(styles.STYLES)),
    ("config.LOOK_KNOBS", lambda: set(c.LOOK_KNOBS)),
    ("config.LOOK_ALIASES", lambda: set(c.LOOK_ALIASES) | set(c.LOOK_ALIASES.values())),
)


def _documented_identifiers() -> set[str]:
    """Every identifier the photo section spells inside backticks, ``-`` folded to
    ``_`` so ``--sky-darken`` reads as the knob it names. Bare prose is excluded on
    purpose: `natural`, `chrome` and `shade` are style names AND ordinary English,
    and a check that accepted the English would be passing on nothing."""
    found: set[str] = set()
    for span in re.findall(r"`([^`\n]+)`", _photo_section()):
        found |= set(re.findall(r"[a-z][a-z0-9_]*", span.replace("-", "_")))
    return found


@pytest.mark.parametrize(
    "where, read, pattern, subject, live",
    COUNT_CLAIMS,
    ids=[f"{row[3]}-in-{row[0]}" for row in COUNT_CLAIMS],
)
def test_a_documented_count_is_the_live_count(where, read, pattern, subject, live):
    claims = re.findall(pattern, read())
    assert claims, (
        f"{where}: the {subject} claim matching /{pattern}/ is gone. If it was "
        "deliberately rewritten as a derivation, DELETE this row from COUNT_CLAIMS; "
        "never loosen the pattern until it matches something else."
    )
    names = live()
    for word in claims:
        claimed = CARDINALS.get(word.lower(), int(word) if word.isdigit() else None)
        assert claimed is not None, (
            f"{where}: the {subject} claim reads '{word}', which is no number this "
            "guard can compare — spell it as a word up to twenty, or as digits."
        )
        assert claimed == len(names), (
            f"{where} claims {subject} holds {claimed} ('{word}'); it holds "
            f"{len(names)}: {sorted(names)}."
        )


@pytest.mark.parametrize("subject, live", NAME_CLAIMS, ids=[s for s, _ in NAME_CLAIMS])
def test_every_published_name_is_documented(subject, live):
    undocumented = sorted(live() - _documented_identifiers())
    assert not undocumented, (
        f"{DOC.name}'s '## Photo pipeline' section never names {undocumented} from "
        f"{subject}. A name the code publishes and the docs never mention is exactly "
        "how the registry summaries went stale — document it in the same change."
    )
