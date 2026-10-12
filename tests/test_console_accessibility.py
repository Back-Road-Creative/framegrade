"""The shared base layout gets landmarks and a reachable
table scroll region.

Two defects in ``_base.html``:

1. No landmarks at all -- no ``<main>``, no ``<nav>``, no ``role=``/``aria-`` anywhere.
   Screen-reader users navigate by landmark; there were none.
2. ``.table-scroll, table { overflow-x: auto; }`` makes every table its own
   horizontally-scrolling region, but nothing makes that region focusable, so a
   keyboard-only user cannot scroll it at all (axe: scrollable-region-focusable).

The landmark/table guards above are base-only: every console page extends
``_base.html`` (see ``test_console_base_layout.py``), so fixing it once there fixes
every page. The form-control guards at the bottom of this file are NOT -- a label
belongs to the field it names, so those scan every page template individually. Both
live here so the console has one accessibility home rather than two that drift.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app

_TEMPLATES = Path(__file__).resolve().parent.parent / "gomoveshift" / "console" / "templates"
_BASE_PATH = _TEMPLATES / "_base.html"


def _base_source() -> str:
    return _BASE_PATH.read_text(encoding="utf-8")


def test_base_template_wraps_content_in_a_main_landmark():
    base = _base_source()
    assert re.search(r'<main\b[^>]*id="([\w-]+)"', base), (
        '_base.html must wrap {% block content %} in a <main id="..."> landmark'
    )
    match = re.search(r'<main\b[^>]*id="([\w-]+)"', base)
    main_id = match.group(1)
    # the content block must actually be inside the <main>, not merely adjacent to it
    main_open = base.index(match.group(0))
    content_pos = base.index("{% block content %}")
    main_close = base.index("</main>")
    assert main_open < content_pos < main_close, (
        "{% block content %} must be nested inside <main>, not outside it"
    )
    resp = TestClient(create_app()).get("/", headers={"accept": "text/html"})
    assert f'id="{main_id}"' in resp.text
    assert "<main" in resp.text and "</main>" in resp.text


def test_base_template_has_a_skip_link_targeting_main():
    base = _base_source()
    skip_match = re.search(r'<a\s+[^>]*href="#([\w-]+)"[^>]*>', base)
    assert skip_match, (
        '_base.html must have a skip link (an <a href="#...">) as the first focusable element'
    )
    target_id = skip_match.group(1)
    main_match = re.search(r'<main\b[^>]*id="([\w-]+)"', base)
    assert main_match, '_base.html must have a <main id="..."> for the skip link to target'
    assert target_id == main_match.group(1), "the skip link's href target must match the <main> id"
    # rendered response actually contains both the link and the target it points to
    resp = TestClient(create_app()).get("/", headers={"accept": "text/html"})
    assert f'href="#{target_id}"' in resp.text
    assert f'id="{target_id}"' in resp.text


def test_base_template_makes_table_scroll_regions_keyboard_reachable():
    base = _base_source()
    assert "focus-visible" in base, (
        "_base.html must define a visible :focus-visible style for the scroll region"
    )
    # progressive enhancement: base-level script gives overflowing tables tabindex+role
    assert "tabindex" in base and "role" in base, (
        "_base.html must make table scroll regions focusable (tabindex) and named (role=region)"
    )
    assert "<script" in base, (
        "the table-scroll fix must be a base-level script, since page templates "
        "cannot each be edited to add tabindex by hand"
    )


def test_table_role_lives_on_a_wrapper_not_on_the_table_element():
    """An explicit role overrides an element's implicit one: <table role="region">
    reads to a screen reader as a generic region, not a table, and its rows/cells
    lose their owning context (row/column nav, header association, "table with N
    rows"). The scroll region must be a separate wrapper element."""
    base = _base_source()
    assert "createElement" in base, (
        "a bare <table> must be wrapped in a new element before it becomes a scroll region"
    )
    assert "el.setAttribute('role'" not in base, (
        "role must never be set directly on the <table> element itself"
    )
    assert "target.setAttribute('role'" in base or re.search(
        r"\w+\.setAttribute\('role', 'region'\)", base
    ), "role must be set on the wrapper, not the table"


def test_table_scroll_css_restores_table_display_once_wrapped():
    base = _base_source()
    assert "[data-table-scroll] table" in base, (
        "once JS wraps a table in a [data-table-scroll] region, the CSS must undo the "
        "no-JS-fallback 'display: block' on the inner <table> -- display:block on a "
        "<table> is itself a semantics smell and only belongs on the no-JS fallback path"
    )


def test_heading_naming_walks_each_elements_own_siblings_not_the_pages_first_heading():
    """The old bug: document.querySelector('h1, h2') runs unscoped inside the
    forEach, so it returns the page's first heading for every table -- multiple
    tables on one page (schedule.html has 3) all get the identical accessible name."""
    base = _base_source()
    match = re.search(r"function nearestHeadingLabel\(el\)\s*\{(.*?)\n  \}", base, re.DOTALL)
    assert match, (
        "naming must be a per-element function taking the element, not a single "
        "global lookup shared by every table"
    )
    body = match.group(1)
    assert "previousElementSibling" in body, (
        "must walk the element's own preceding siblings/ancestors for its nearest "
        "heading instead of grabbing one page-wide heading for every table"
    )
    sibling_walk_pos = body.index("previousElementSibling")
    fallback_pos = body.find("document.querySelector")
    assert fallback_pos == -1 or sibling_walk_pos < fallback_pos, (
        "the page-wide querySelector fallback must run only after the sibling walk finds nothing"
    )


_SCHEDULE_REPORT_WITH_THREE_TABLES = {
    "as_of": "2026-08-10T09:00:00-04:00",
    "weekly_longform_capacity": 3,
    "designed_days": ["tue", "fri", "sun"],
    "scheduled_longforms": [
        {
            "date": "2026-08-11",
            "time": "16:00:00",
            "region": "usa",
            "title": "Booked Drive Through The Hills",
            "kind": "longform",
        }
    ],
    "scheduled_shorts_count": 4,
    "queue_ends": "2026-08-11",
    "next_available": [
        {
            "region": "usa",
            "day": "tue",
            "slot": "2026-08-18T16:00:00-04:00",
            "shorts_windows": ["14:00", "18:00"],
            "shorts_required": 2,
            "earliest_for_region": True,
        }
    ],
    "next_available_superlong": [
        {"region": "usa", "day": "sat", "slot": "2026-08-15T15:00:00-04:00"}
    ],
}


def test_multi_table_page_offers_a_distinct_preceding_heading_per_table(monkeypatch):
    """schedule.html renders 3 tables, each directly preceded by its own <h2>. This
    proves the markup gives the naming algorithm what it needs to produce distinct
    accessible names -- the algorithm test above proves it actually uses that."""
    from gomoveshift.console import schedule as console_schedule
    from gomoveshift.console.app import create_app as _create_app

    monkeypatch.setattr(
        console_schedule,
        "build_schedule_report",
        lambda ledger, config: _SCHEDULE_REPORT_WITH_THREE_TABLES,
    )
    resp = TestClient(_create_app()).get("/schedule", headers={"accept": "text/html"})
    # strip <style>/<script> blocks first: the base template's own prose comments
    # mention "<table>" and would otherwise be misread as real table tags below.
    html = re.sub(r"<(style|script)\b.*?</\1>", "", resp.text, flags=re.DOTALL)
    table_positions = [m.start() for m in re.finditer(r"<table\b", html)]
    assert len(table_positions) >= 2, "need at least 2 rendered tables to prove distinct naming"
    heading_positions = [
        (m.start(), re.sub(r"<[^>]+>", "", m.group(1)).strip())
        for m in re.finditer(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html)
    ]

    def _nearest_preceding_heading(pos: int) -> str | None:
        candidates = [text for p, text in heading_positions if p < pos]
        return candidates[-1] if candidates else None

    names = [_nearest_preceding_heading(p) for p in table_positions]
    assert all(names), f"every table needs a preceding heading to name from: {names}"
    assert len(set(names)) == len(names), f"tables must have distinct nearest headings, got {names}"


# --- Form controls: a label belongs to its field, so this half is per-page. ---

#: Jinja statements, expressions and comments are not HTML; strip them before
#: parsing so a ``{% if %}`` inside a tag cannot be mistaken for an attribute.
#:
#: The comment branch comes first so it wins at ``{#`` even when the comment
#: quotes an expression: a comment is where a template author writes ABOUT
#: markup, and that prose was being parsed as the markup it describes.
_JINJA = re.compile(r"\{#.*?#\}|\{%.*?%\}|\{\{.*?\}\}", re.DOTALL)

#: These carry their own accessible name from their value or enclosing button
#: text, so a label would be redundant rather than missing.
_SELF_LABELLING_INPUT_TYPES = {"hidden", "submit", "button", "image", "reset"}


class _ControlScanner(HTMLParser):
    """Collect controls lacking an accessible name, and forms inside paragraphs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.label_depth = 0
        self.in_paragraph = False
        self.unlabelled: list[str] = []
        self.forms_in_paragraphs: list[str] = []
        self.label_targets: set[str] = set()
        self.controls_with_ids: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attr = {k: (v or "") for k, v in attrs}
        if tag == "label":
            self.label_depth += 1
            if attr.get("for"):
                self.label_targets.add(attr["for"])
        elif tag == "p":
            self.in_paragraph = True
        elif tag == "form" and self.in_paragraph:
            self.forms_in_paragraphs.append(attr.get("action", "<no action>"))
        elif tag in {"input", "select", "textarea"}:
            if tag == "input" and attr.get("type", "text") in _SELF_LABELLING_INPUT_TYPES:
                return
            described = (
                self.label_depth > 0 or attr.get("aria-label") or attr.get("aria-labelledby")
            )
            if described:
                return
            if attr.get("id"):
                self.controls_with_ids.append((attr["id"], attr.get("name", tag)))
                return
            self.unlabelled.append(attr.get("name") or tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "label":
            self.label_depth = max(0, self.label_depth - 1)
        elif tag == "p":
            self.in_paragraph = False

    def unresolved(self) -> list[str]:
        """Controls with an id that no ``<label for=>`` ever points at."""
        return [
            name
            for control_id, name in self.controls_with_ids
            if control_id not in self.label_targets
        ]


def _scan(path: Path) -> _ControlScanner:
    scanner = _ControlScanner()
    scanner.feed(_JINJA.sub("", path.read_text(encoding="utf-8")))
    scanner.close()
    return scanner


def _pages() -> list[Path]:
    return sorted(_TEMPLATES.glob("*.html"))


def test_there_are_templates_to_scan():
    assert _pages(), f"no console templates found under {_TEMPLATES}"


def test_every_form_control_has_an_accessible_name():
    offenders: dict[str, list[str]] = {}
    for page in _pages():
        scanner = _scan(page)
        missing = scanner.unlabelled + scanner.unresolved()
        if missing:
            offenders[page.name] = missing
    assert not offenders, (
        "form controls with no label, aria-label, or matching <label for=>: "
        f"{offenders} -- a placeholder is not a label"
    )


def test_no_form_is_nested_inside_a_paragraph():
    offenders = {
        page.name: scanner.forms_in_paragraphs
        for page in _pages()
        if (scanner := _scan(page)).forms_in_paragraphs
    }
    assert not offenders, f"<form> inside <p> -- the parser closes the paragraph first: {offenders}"


# --- The scanner's own blind spot. ---
#
# `_scan` strips Jinja before handing the text to an HTML parser, so a template's
# own control flow cannot be mistaken for markup. Comments were not stripped, and a
# comment is exactly where a template author writes ABOUT markup -- "these forms sit
# at block level, never wrapped in a <p>". That sentence was parsed as a real `<p>`,
# left the scanner inside a paragraph for the rest of the file, and reported the next
# genuine form as an offender.


def _scan_text(tmp_path: Path, markup: str) -> _ControlScanner:
    page = tmp_path / "page.html"
    page.write_text(markup, encoding="utf-8")
    return _scan(page)


def test_a_comment_describing_markup_is_not_scanned_as_markup(tmp_path):
    scanner = _scan_text(
        tmp_path,
        "{# these forms sit at block level, never wrapped in a <p> #}\n"
        '<form action="/publishing/cancel"><button type="submit">Cancel</button></form>\n',
    )

    assert scanner.forms_in_paragraphs == []


def test_a_comment_cannot_hide_a_control_from_the_label_check(tmp_path):
    scanner = _scan_text(
        tmp_path,
        '{# the <label> below is deliberate #}\n<input type="text" name="journey">\n',
    )

    assert scanner.unlabelled == ["journey"]


def test_a_comment_holding_an_expression_is_stripped_whole(tmp_path):
    # The comment branch has to win at `{#` even though `{{ }}` appears inside it,
    # or the leftover `#}` lands in the parsed text.
    scanner = _scan_text(
        tmp_path,
        "{# shows {{ row.id }} inside a <p> #}\n"
        '<form action="/publishing/retry"><button type="submit">Retry</button></form>\n',
    )

    assert scanner.forms_in_paragraphs == []


def test_a_multi_line_comment_is_stripped(tmp_path):
    scanner = _scan_text(
        tmp_path,
        "{# a paragraph opens with <p>\n   and this line never closes it #}\n"
        '<form action="/seo/corrections"><button type="submit">Save</button></form>\n',
    )

    assert scanner.forms_in_paragraphs == []


def test_real_markup_outside_a_comment_is_still_caught(tmp_path):
    # The strip must not become a blanket that swallows the finding it exists to
    # protect: a form genuinely inside a paragraph is still an offender.
    scanner = _scan_text(
        tmp_path,
        "{# describing a <form> here #}\n"
        '<p>Pick one <form action="/real"><button type="submit">Go</button></form></p>\n',
    )

    assert scanner.forms_in_paragraphs == ["/real"]
