"""Every console field says what it is.

A `placeholder` is not an accessible name. It disappears the moment anyone
types, screen readers treat it inconsistently, and it fails the axe `label`
check outright -- so a field whose only description is a placeholder is a
field an operator can be looking at with no idea what it wants. Several of
these ask for a filesystem path or a date key, where guessing wrong is the
whole problem.

Every console template is already clean -- this guard fixes nothing today and
is not meant to. It is the robust tier for a class of bug that is easy to
reintroduce: a field that hints what it wants in grey placeholder text instead
of naming itself. Fixing one template by hand leaves nothing behind to stop the
next one. The guard walks every control in every console template, so a new
page cannot add an unnamed field and have nothing notice.

A control is named if it has a `<label for=>` pointing at its id, an
`aria-label`, or sits directly inside its own `<label>` -- the three things
that actually produce an accessible name. There is deliberately no exemption
list: an exemption nobody rechecks is how this class got back in, so a
template that needs one has to add it, and justify it, in the PR that needs it.
"""

from __future__ import annotations

import re
from pathlib import Path

_TEMPLATES = Path(__file__).resolve().parent.parent / "gomoveshift" / "console" / "templates"

#: A control with no accessible name of its own. `hidden` carries no name and
#: needs none; `submit`/`button` are named by their own text content.
_UNNAMEABLE_TYPES = re.compile(r'type="(hidden|submit|button|image)"')

_CONTROL = re.compile(r"<(select|textarea|input)\b([^>]*)>", re.IGNORECASE)


def _unnamed(template: Path) -> list[str]:
    """Controls in ``template`` that no accessible name reaches."""
    text = template.read_text()
    labelled_ids = set(re.findall(r'<label[^>]*\bfor="([^"]+)"', text))
    missing = []
    for match in _CONTROL.finditer(text):
        attrs = match.group(2)
        if _UNNAMEABLE_TYPES.search(attrs):
            continue
        identifier = re.search(r'\bid="([^"]+)"', attrs)
        # Directly inside its own <label>: the nearest preceding <label> must
        # still be open at this point, i.e. its </label> comes after us.
        opened = text.rfind("<label", 0, match.start())
        closed = text.find("</label>", match.start())
        wrapped = opened != -1 and closed != -1 and text.find("</label>", opened) == closed
        if "aria-label" in attrs or (identifier and identifier.group(1) in labelled_ids) or wrapped:
            continue
        name = re.search(r'\bname="([^"]+)"', attrs)
        missing.append(f"<{match.group(1)} name={name.group(1) if name else '?'}>")
    return missing


def _page_templates() -> list[Path]:
    return sorted(p for p in _TEMPLATES.glob("*.html") if p.name != "_base.html")


def test_templates_exist():
    assert _page_templates(), f"no console templates found under {_TEMPLATES}"


def test_every_console_field_has_an_accessible_name():
    offenders = {}
    for template in _page_templates():
        missing = _unnamed(template)
        if missing:
            offenders[template.name] = missing
    assert not offenders, (
        "these console fields have no accessible name (a placeholder is not one -- "
        f"it vanishes on input and fails axe `label`): {offenders}"
    )


def test_the_guard_can_actually_see_an_unnamed_field(tmp_path):
    """The guard is a regex over markup, so it needs its own proof that it
    fails when it should -- otherwise a silent always-pass reads as a clean
    console."""
    page = tmp_path / "made_up.html"
    page.write_text('<form><input type="text" name="lonely" placeholder="guess"></form>')

    assert _unnamed(page) == ["<input name=lonely>"]


def test_the_guard_accepts_each_of_the_three_real_naming_forms(tmp_path):
    page = tmp_path / "named.html"
    page.write_text(
        '<label for="a">A</label><input type="text" id="a" name="a">'
        '<input type="text" name="b" aria-label="B">'
        '<label>C <input type="text" name="c"></label>'
    )

    assert _unnamed(page) == []
