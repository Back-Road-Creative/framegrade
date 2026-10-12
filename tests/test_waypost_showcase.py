"""`bin/waypost-showcase.py`: a static, read-only showcase of the Waypost
console, built from a fixed fictional fixture -- never real tenant data.

Loaded by path: a hyphenated filename cannot be imported as a module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.console import jobs

GENERATOR = Path(__file__).resolve().parents[1] / "bin" / "waypost-showcase.py"
_spec = importlib.util.spec_from_file_location("waypost_showcase", GENERATOR)
assert _spec is not None and _spec.loader is not None, f"{GENERATOR} is not importable"
showcase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(showcase)

HOME_URL = "https://example.com/"
PRODUCT_URL = "https://example.com/waypost/"

PAGE = (
    '<!doctype html><html><head><meta name="viewport" content="width=device-width">'
    '<link rel="stylesheet" href="/static/style.css"></head><body>'
    '<a href="/jobs">Jobs</a><a href="/settings">Settings</a>'
    '<a href="https://example.invalid/x">Away</a>'
    '<form method="post" action="/jobs"><button type="submit">Launch</button></form>'
    "<script>alert(1)</script></body></html>"
)


@pytest.fixture
def page():
    bundle = {"index.html": PAGE, "jobs.html": PAGE}
    return showcase.transform_pages(bundle, "2026-09-01", "abc1234")["index.html"]


def test_a_url_becomes_one_flat_sibling_file():
    urls = ("/", "/publishing/calendar", "/schedule?journey=x")
    expected = "index.html publishing-calendar.html schedule.html"
    assert [showcase.snapshot_name(url) for url in urls] == expected.split()


def test_forms_and_buttons_are_removed(page):
    for gone in ("<form", "<button", 'type="submit"'):
        assert gone not in page, f"{gone} survived the transform"


def test_a_captured_link_is_rewritten_to_its_sibling_file(page):
    assert 'href="jobs.html"' in page


def test_an_uncaptured_or_external_link_becomes_inert_text(page):
    assert 'href="/settings"' not in page
    assert 'href="https://example.invalid/x"' not in page
    assert "showcase-unavailable" in page


def test_the_page_says_what_it_is_and_asks_not_to_be_indexed(page):
    assert 'name="robots" content="noindex, nofollow"' in page
    assert "showcase-notice" in page and "2026-09-01" in page and "abc1234" in page


def test_no_script_tag_survives(page):
    """The site serves the bundle under a CSP with no 'unsafe-inline'; an inline
    script would be blocked there, so none may ship."""
    assert not re.search(r"<script\b", page, re.IGNORECASE)


def test_viewport_meta_survives(page):
    assert 'name="viewport"' in page


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("waypost-showcase-out")
    exit_code = showcase.main(["--out", str(out), "--source", "abc1234"])
    assert exit_code == 0
    return out


def test_generator_produces_at_least_five_pages(bundle_dir):
    pages = sorted(p.name for p in bundle_dir.glob("*.html"))
    assert len(pages) >= 5
    # One page per required showcase category.
    assert "index.html" in pages
    assert "jobs.html" in pages  # jobs/queue view
    assert "publishing.html" in pages  # review view (awaiting-manual-publish rows)
    assert any(name in pages for name in ("publishing-calendar.html", "schedule.html"))


def test_manifest_matches_the_written_bundle(bundle_dir):
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["kind"] == "waypost-static-showcase"
    assert manifest["source"] == "abc1234"
    assert manifest["anchor"] == "2026-09-01"
    pages = sorted(p.name for p in bundle_dir.glob("*.html"))
    assert manifest["pages"] == len(pages)
    assert manifest["files"] == pages
    for name, digest in manifest["sha256"].items():
        actual = hashlib.sha256((bundle_dir / name).read_bytes()).hexdigest()
        assert actual == digest


def test_every_page_is_inert_and_labelled(bundle_dir):
    for html_path in bundle_dir.glob("*.html"):
        body = html_path.read_text()
        assert 'name="robots" content="noindex, nofollow"' in body
        assert 'name="viewport"' in body
        assert not re.search(r"<script\b", body, re.IGNORECASE)
        assert "<form" not in body
        assert "<button" not in body
        assert "<input" not in body
        allowed = {HOME_URL, PRODUCT_URL}
        for match in re.findall(r'href="([^"]+)"', body):
            if match.startswith(("http://", "https://")):
                assert match in allowed, f"{html_path.name}: absolute link {match}"


def test_regenerating_the_bundle_is_byte_identical(tmp_path):
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    assert showcase.main(["--out", str(out1), "--source", "abc1234"]) == 0
    assert showcase.main(["--out", str(out2), "--source", "abc1234"]) == 0
    names1 = sorted(p.name for p in out1.glob("*.html"))
    names2 = sorted(p.name for p in out2.glob("*.html"))
    assert names1 == names2
    for name in names1:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
    assert (out1 / "manifest.json").read_bytes() == (out2 / "manifest.json").read_bytes()


def test_a_process_tracked_for_another_store_leaves_the_bundle_unchanged(tmp_path, monkeypatch):
    """A finished rc=1 process tracked as job 1 of another store (left by an
    earlier test on the same xdist worker) must not be reaped into the
    showcase's seeded job 1: that rendered ``failed`` under "Needs attention" on
    the first regeneration but not the second, breaking byte-identity."""
    monkeypatch.setattr(jobs, "_PROCESSES", {})
    monkeypatch.setattr(jobs.notify, "notify_job_failed", lambda row: None)
    with monkeypatch.context() as launching:
        launching.setattr(jobs, "_validate_argv", lambda argv: None)
        launching.setattr(jobs.subprocess, "Popen", lambda *a, **k: SimpleNamespace(poll=lambda: 1))
        other = jobs.JobStore(tmp_path / "other" / "jobs.sqlite3")
        assert jobs.launch(other, "drive", ["x"], tmp_path / "other-logs") == 1

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    assert showcase.main(["--out", str(out1), "--source", "abc1234"]) == 0
    assert showcase.main(["--out", str(out2), "--source", "abc1234"]) == 0
    for html in sorted(out1.glob("*.html")):
        assert html.read_bytes() == (out2 / html.name).read_bytes(), html.name


def test_custom_source_lands_in_the_manifest(tmp_path):
    out = tmp_path / "out"
    assert showcase.main(["--out", str(out), "--source", "v9.9.9-demo"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source"] == "v9.9.9-demo"


TABLE_PAGE = PAGE.replace("<form", "<table><tr><td>x</td></tr></table><form", 1)


def test_the_return_bar_has_two_absolute_links_and_the_site_label():
    out = showcase.transform_pages({"index.html": PAGE}, "2026-09-01", "abc1234")["index.html"]
    assert f'<a class="showcase-home" href="{HOME_URL}">&larr; Home</a>' in out
    assert f'<a class="showcase-product" href="{PRODUCT_URL}">Product details</a>' in out
    assert "<strong>Fixed fictional demo</strong>" in out
    assert "Fixed demo<" not in out
    assert "<summary>About this demo</summary>" in out
    assert "abc1234" in out and "2026-09-01" in out


def test_the_return_bar_urls_are_options():
    out = showcase.transform_pages(
        {"index.html": PAGE},
        "2026-09-01",
        "abc1234",
        home_url="https://h.example/",
        site_url="https://h.example/w/",
    )["index.html"]
    assert 'href="https://h.example/"' in out and 'href="https://h.example/w/"' in out


def test_the_notice_is_a_sticky_solid_bar_with_scroll_margin():
    out = showcase.transform_pages({"index.html": PAGE}, "2026-09-01", "abc1234")["index.html"]
    assert ".showcase-notice{position:sticky;top:0;z-index:50;" in out
    assert "scroll-margin-top" in out
    assert "flex-wrap:wrap" in out


def test_every_table_sits_in_a_bounded_keyboard_reachable_container():
    out = showcase.transform_pages({"index.html": TABLE_PAGE}, "2026-09-01", "abc1234")[
        "index.html"
    ]
    assert out.count("<table") == out.count('<div class="showcase-table" tabindex="0">')
    assert out.count("</table>") == out.count("</table></div>")
    assert ".showcase-table{overflow-x:auto;max-width:100%}" in out
    assert "overflow-wrap:anywhere" in out


def test_every_exported_table_is_wrapped(bundle_dir):
    total = 0
    for html_path in bundle_dir.glob("*.html"):
        raw = html_path.read_text()
        body = re.sub(r"<!--.*?-->|<style\b.*?</style>|<script\b.*?</script>", "", raw, flags=re.S)
        wrapped = len(re.findall(r'<div class="showcase-table" tabindex="0"><table\b', body))
        assert wrapped == len(re.findall(r"<table\b", body)), html_path.name
        total += wrapped
    assert total > 0, "no exported page has a table; the check proves nothing"


def test_the_bundle_satisfies_the_static_site_contract(bundle_dir):
    """The exported bundle is a closed, flat, read-only set of fictional pages."""
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    names = sorted(p.name for p in bundle_dir.glob("*.html"))
    assert manifest["files"] == names
    assert len(names) >= 5
    for name in names:
        body = (bundle_dir / name).read_text()
        assert "Fixed fictional demo" in body, name
        assert 'href="/"' not in body, name
        assert not re.search(r'href="/', body), name
        assert not re.search(r"<(?:script|form|button|input)\b", body, re.I), name
        assert f'href="{HOME_URL}"' in body and f'href="{PRODUCT_URL}"' in body, name
