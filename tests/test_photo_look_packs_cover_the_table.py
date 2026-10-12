"""Every name in the box is rendered by some pack suite — proved, not remembered.

``BUILTIN_LOOKS`` is merged from three family dicts, and its members are rendered
through the real ``--look`` door by FOUR suites. Two of them pin their whole
family (``tuple(_MASKED) == PACK``), so a look added there fails loudly until
someone renders it. ``_ELECTRONIC`` is split across two suites and NEITHER pins
the union: ``test_photo_electronic_pack.py`` asserts only
``set(PACK) <= set(_ELECTRONIC)``. A look added to that family therefore ships,
passes the config validator, reaches ``--look``'s help, and renders in no test.

That is not hypothetical. ``vhs`` / ``glitch_pop`` / ``leaked_film`` joined
``_ELECTRONIC`` after the pack suite had been written against four names; the subset assert is the loosening that admitted them, and the second
suite covering them was a thing someone remembered to write, not a thing the
tests required. Six look-adding PRs have landed against ``looks.py`` — the odds
of remembering every time are the thing this file removes.

It renders nothing itself. The four suites do the rendering; this one proves the
union of what they name is exactly the public table, no gaps and no duplicates.
"""

from test_photo_artifact_looks import PACK as ARTIFACT_PACK
from test_photo_cinematic_pack import CINEMATIC as CINEMATIC_PACK
from test_photo_electronic_pack import PACK as ELECTRONIC_PACK
from test_photo_masked_pack import PACK as MASKED_PACK

from gomoveshift.pipeline.photos.looks import BUILTIN_LOOKS

# The suite that renders each pack, named so a failure says where to add the test
# rather than only that one is missing.
RENDERED_BY = {
    "tests/test_photo_electronic_pack.py": ELECTRONIC_PACK,
    "tests/test_photo_artifact_looks.py": ARTIFACT_PACK,
    "tests/test_photo_cinematic_pack.py": CINEMATIC_PACK,
    "tests/test_photo_masked_pack.py": MASKED_PACK,
}


def _unrendered(table, rendered_by):
    """Names in ``table`` that no pack in ``rendered_by`` claims. Takes both as
    arguments so the gate's own logic can be tested on a fabricated table."""
    named = {name for pack in rendered_by.values() for name in pack}
    return sorted(set(table) - named)


def test_every_built_in_look_is_named_by_a_suite_that_renders_it():
    """A look in the box that no suite renders is a look nobody has ever seen."""
    unrendered = _unrendered(BUILTIN_LOOKS, RENDERED_BY)
    assert not unrendered, (
        f"{unrendered} ship in BUILTIN_LOOKS and render in no test — add each to the "
        f"PACK of the suite for its family: {sorted(RENDERED_BY)}"
    )


def test_no_suite_names_a_look_that_left_the_table():
    """A renamed or retired look leaves a dead name behind, and the suite that
    still names it fails deep inside a render with an obscure KeyError. Name it
    here instead, beside the suite that has to change."""
    dead = {
        suite: sorted(set(pack) - set(BUILTIN_LOOKS))
        for suite, pack in RENDERED_BY.items()
        if set(pack) - set(BUILTIN_LOOKS)
    }
    assert not dead, f"packs name looks the table no longer has: {dead}"


def test_no_look_is_rendered_by_two_suites():
    """Two suites rendering one look is duplicated time on the slowest tests in
    the photos tree, and leaves no single owner for its direction assertions."""
    named = [name for pack in RENDERED_BY.values() for name in pack]
    twice = sorted({name for name in named if named.count(name) > 1})
    assert not twice, f"{twice} are rendered by more than one suite"


def test_the_gate_catches_a_family_that_gained_an_unrendered_look():
    """The assertions above pass on today's table, so prove the check would NOT
    pass on the state it exists to catch — a gate that has never bitten is
    indistinguishable from one that cannot bite."""
    assert _unrendered(
        {"rendered": {}, "a_look_nobody_rendered": {}},
        {"tests/test_some_pack.py": ("rendered",)},
    ) == ["a_look_nobody_rendered"]
