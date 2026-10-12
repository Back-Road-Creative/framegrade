"""The console's 18 hand-rolled pages share one base layout, and it is responsive.

Every console template carried its own ``<!DOCTYPE>``/``<head>`` block and not one of
them declared a viewport, so every page rendered at desktop width on a phone. A shared
``_base.html`` fixes that once; this guard keeps it fixed -- a new page that hand-rolls
its own document skips the viewport and the tokens, and nothing else would notice.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app

_TEMPLATES = Path(__file__).resolve().parent.parent / "gomoveshift" / "console" / "templates"
_BASE = "_base.html"

_OWNS_A_DOCUMENT = re.compile(r"<!DOCTYPE|<html\b|<head\b|<body\b", re.IGNORECASE)


def _page_templates() -> list[Path]:
    return sorted(p for p in _TEMPLATES.glob("*.html") if p.name != _BASE)


def test_templates_exist():
    assert _page_templates(), f"no console templates found under {_TEMPLATES}"


def test_base_template_is_responsive():
    base = (_TEMPLATES / _BASE).read_text(encoding="utf-8")
    assert 'name="viewport"' in base, f"{_BASE} must declare a viewport meta"
    assert "width=device-width" in base, (
        f"{_BASE} viewport must be width=device-width, else phones render at desktop width"
    )
    assert 'charset="utf-8"' in base, f"{_BASE} must declare a charset"


@pytest.mark.parametrize("template", _page_templates(), ids=lambda p: p.name)
def test_every_page_extends_the_base_layout(template: Path):
    html = template.read_text(encoding="utf-8")
    assert f'{{% extends "{_BASE}" %}}' in html, (
        f"{template.name} does not extend {_BASE}, so it carries its own head and "
        "misses the viewport meta and shared tokens"
    )
    stray = _OWNS_A_DOCUMENT.search(html)
    assert stray is None, (
        f"{template.name} extends {_BASE} but still declares {stray.group(0)!r}; "
        "the base owns the document shell"
    )


@pytest.mark.parametrize("path", ["/", "/does-not-exist"])
def test_rendered_pages_carry_the_viewport(path: str):
    """The static checks above prove the source; this proves what a browser receives."""
    resp = TestClient(create_app()).get(path, headers={"accept": "text/html"})
    assert "text/html" in resp.headers["content-type"]
    assert 'name="viewport"' in resp.text, (
        f"{path} rendered without a viewport meta: {resp.text[:200]}"
    )


def _nav_block(html: str) -> str:
    """Just the <nav>...</nav>, so a nav assertion cannot be satisfied by a
    link that happens to live in the page body -- the index page lists every
    tab in its own content, which would green every check below for free."""
    match = re.search(r"<nav\b.*?</nav>", html, re.DOTALL | re.IGNORECASE)
    assert match, f"no <nav> landmark in the rendered page: {html[:300]}"
    return match.group(0)


@pytest.mark.parametrize("path", ["/", "/jobs", "/does-not-exist"])
def test_every_page_carries_the_navigation_landmark(path: str):
    """Before this, the console had no navigation at all: every page ended in
    a single "Back" link to ``/``, so Publishing to Jobs meant going home
    first. A <nav> in the shared base gives every page -- the error page
    included -- the same links, and gives a screen reader a landmark to jump
    to instead of tabbing the whole document."""
    resp = TestClient(create_app()).get(path, headers={"accept": "text/html"})
    nav = _nav_block(resp.text)
    assert 'aria-label="Console sections"' in nav, (
        f"{path}'s nav is unlabelled; with more than one landmark of a kind a "
        "screen reader announces them all as just 'navigation'"
    )
    assert 'href="/jobs"' in nav, f"{path}'s nav does not link to Jobs"
    assert 'href="/settings"' in nav, f"{path}'s nav does not link to Settings"


def test_the_current_page_is_marked_in_the_nav():
    """Without aria-current a screen-reader user tabbing the nav has no way to
    tell which of the ten links is the page they are already on."""
    resp = TestClient(create_app()).get("/jobs", headers={"accept": "text/html"})
    nav = _nav_block(resp.text)
    assert re.search(r'<a[^>]*href="/jobs"[^>]*aria-current="page"', nav), (
        f"/jobs does not mark its own nav link current: {nav}"
    )
    assert nav.count('aria-current="page"') == 1, (
        f"exactly one nav link is the current page; got {nav.count(chr(34).join(['aria-current=', 'page', '']))}"
    )


def test_a_nested_route_marks_its_owning_tab_current():
    """``/drives/gates/pending`` is still the Drives tab. Matching the whole
    path would light up nothing at all on every detail page in the console."""
    resp = TestClient(create_app()).get("/drives/gates/pending", headers={"accept": "text/html"})
    assert re.search(r'<a[^>]*href="/drives"[^>]*aria-current="page"', _nav_block(resp.text)), (
        "a nested drives route does not mark Drives current"
    )


def test_the_nav_offers_every_enabled_tab_and_nothing_else():
    """The nav is built from the same brand-filtered list the tab gate uses,
    so a brand with a tab switched off is never offered a link that 404s, and
    the nav never grows a second source of truth for which tabs exist."""
    from gomoveshift.console.app import NAV_ENTRIES

    nav = _nav_block(TestClient(create_app()).get("/", headers={"accept": "text/html"}).text)
    linked = set(re.findall(r'href="/([a-z-]+)"', nav))
    assert linked == {slug for slug, _title in NAV_ENTRIES}


# --- Nothing in the shell may push the page wider than the phone it is on. ---
#
# The base already makes the body fluid and gives tables their own scroll box, and
# that is ALL it held: a <pre> log tail, a catalog id with no spaces in it, and an
# <img> bigger than the column each still pushed a 390px viewport sideways. These
# guards are structural on purpose -- they read which elements the console's pages
# actually render and demand the base contain each one, so a page added next month
# with an unguarded <pre> or <iframe> fails here instead of shipping an overflow.

#: For each element that overflows its container by default, the declarations that
#: would hold it. Satisfying ANY ONE of a tag's entries contains it: a <pre> may
#: either wrap or scroll, but it may not do neither.
_CONTAINING_DECLARATIONS: dict[str, dict[str, set[str]]] = {
    "pre": {"white-space": {"pre-wrap", "pre-line", "normal"}, "overflow-x": {"auto", "scroll"}},
    "img": {"max-width": {"100%"}},
    "video": {"max-width": {"100%"}},
    "iframe": {"max-width": {"100%"}},
    "canvas": {"max-width": {"100%"}},
    "embed": {"max-width": {"100%"}},
    "object": {"max-width": {"100%"}},
    "svg": {"max-width": {"100%"}},
}

_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENTS = re.compile(r"\{#.*?#\}|<!--.*?-->", re.DOTALL)


def _base_css_rules() -> list[tuple[list[str], dict[str, str]]]:
    """The base's <style> block as (selectors, declarations) pairs.

    Comments are stripped first: the base documents its rules in prose that
    quotes the very selectors and tags being matched for here.
    """
    css = "\n".join(_STYLE_BLOCK.findall((_TEMPLATES / _BASE).read_text(encoding="utf-8")))
    rules: list[tuple[list[str], dict[str, str]]] = []
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _CSS_COMMENT.sub("", css)):
        declarations = {}
        for declaration in body.split(";"):
            prop, sep, value = declaration.partition(":")
            if sep:
                declarations[prop.strip().lower()] = value.strip().lower()
        rules.append(([s.strip().lower() for s in selectors.split(",")], declarations))
    return rules


def _declarations_for(selector: str) -> dict[str, str]:
    """Declarations every element matching a BARE type selector gets.

    Only an unqualified selector counts: ``[data-table-scroll] table`` holds one
    table in one wrapper, and a rule that needs an ancestor cannot be relied on
    to contain the element wherever a page happens to put it.
    """
    declarations: dict[str, str] = {}
    for selectors, rule in _base_css_rules():
        if selector in selectors:
            declarations.update(rule)
    return declarations


def _pages_rendering(tag: str) -> list[str]:
    pattern = re.compile(rf"<{tag}\b", re.IGNORECASE)
    return [
        p.name
        for p in _page_templates()
        if pattern.search(_COMMENTS.sub("", p.read_text(encoding="utf-8")))
    ]


@pytest.mark.parametrize("tag", sorted(_CONTAINING_DECLARATIONS))
def test_the_base_contains_every_overflow_prone_element_a_page_renders(tag: str):
    pages = _pages_rendering(tag)
    if not pages:
        pytest.skip(f"no console template renders <{tag}>")
    declared = _declarations_for(tag)
    wanted = _CONTAINING_DECLARATIONS[tag]
    assert any(declared.get(prop) in values for prop, values in wanted.items()), (
        f"{', '.join(pages)} render(s) <{tag}>, which overflows its container by "
        f"default, and {_BASE} has no rule containing it. Add one of "
        f"{ {p: sorted(v) for p, v in wanted.items()} } to a bare `{tag}` selector "
        f"in the base -- not to the page, which is a per-page copy that goes stale. "
        f"Got: {declared or 'no bare-`' + tag + '` rule at all'}"
    )


def test_an_unbreakable_string_cannot_push_the_page_sideways():
    """Catalog ids (``photo:journey-a:2025-01-01:photo-001.jpg``) and absolute
    paths have no space to wrap at, so a <td> or <p> holding one grows past the
    viewport however fluid its parents are. ``overflow-wrap`` inherits, so the
    body is the one place this is declared."""
    declared = _declarations_for("body")
    assert declared.get("overflow-wrap") in {"break-word", "anywhere"}, (
        f"{_BASE}'s body must set overflow-wrap so an unbreakable id or path wraps "
        f"instead of widening the page; got {declared.get('overflow-wrap')!r}"
    )


@pytest.mark.parametrize("template", _page_templates(), ids=lambda p: p.name)
def test_no_page_carries_its_own_containment_style(template: Path):
    """An inline ``style="max-width:100%"`` is the per-page copy the base exists
    to delete -- the next page to render an image does not inherit a decision
    written in another template's attribute."""
    inline = re.search(r'style="[^"]*max-width', template.read_text(encoding="utf-8"))
    assert inline is None, (
        f"{template.name} contains an inline {inline.group(0)!r} rather than "
        f"relying on {_BASE}; containment belongs to the shell"
    )
