#!/usr/bin/env python3
"""Export a fixed set of Waypost console pages as a portable, read-only static showcase.

Seeded from a throwaway fixture (fictional brand, fictional jobs and a fictional
publishing queue) rather than any real tenant's data, credentials or media — the
console never sees a live deployment while this runs. Every page keeps the base
layout's own accessibility script (DOM-only, talks to nothing) and viewport meta;
this generator only removes what would let a visitor act (forms, buttons) and
rewrites internal links so the bundle is a closed, flat set of sibling files.

Deterministic: no page here carries a wall-clock value
(job timestamps and delivery timestamps are pinned to the built-in fixture
date; ``--anchor`` sets only the label and manifest date).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KIND = "waypost-static-showcase"
BRAND = "gomoveshift"
#: Placeholder link targets; pass ``--home-url`` / ``--site-url`` for a real site.
HOME_URL = "https://example.com/"
SITE_URL = "https://example.com/waypost/"

#: The as-of the bundle claims -- fixture timestamps are pinned to it so a
#: regenerated bundle is byte-identical.
ANCHOR = date(2026, 9, 1)

#: Console routes captured, one per required showcase category: index, the
#: job/queue view (plus one job's detail), the publishing review view, and
#: the schedule/calendar view. ``/schedule`` is deliberately excluded: its
#: report stamps a real wall-clock "As of" line the fixture cannot pin, so
#: the calendar view alone stands in for the schedule/calendar category.
PAGES = ("/", "/jobs", "/jobs/1", "/publishing", "/publishing/calendar")

FORM = re.compile(r"<form\b[^>]*>.*?</form>", re.IGNORECASE | re.DOTALL)
BUTTON = re.compile(r"<button\b[^>]*>.*?</button>", re.IGNORECASE | re.DOTALL)
#: Every script, including the console layout's inline table-a11y helper: the
#: bundle must work under a content-security-policy with no 'unsafe-inline', where
#: an inline script is dead, and _wrap_table already gives each table the
#: bounded, keyboard-reachable container that helper would have added.
SCRIPT = re.compile(r"<script\b.*?</script>", re.IGNORECASE | re.DOTALL)
HREF = re.compile(r'href="([^"]+)"')
#: A real table, or (matched first, left alone) a comment/style/script block that
#: merely mentions one -- the layout's CSS comments and script do.
TABLE = re.compile(
    r"(?P<inert><!--.*?-->|<style\b.*?</style>|<script\b.*?</script>)|(?P<table><table\b.*?</table>)",
    re.IGNORECASE | re.DOTALL,
)


def _wrap_table(match: "re.Match[str]") -> str:
    if match.group("table") is None:
        return match.group(0)
    return f'<div class="showcase-table" tabindex="0">{match.group("table")}</div>'


ANCHOR_TAG = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<body>.*?)</a>", re.IGNORECASE | re.DOTALL)


def snapshot_name(url: str) -> str:
    """The sibling file a route becomes; a bundle is one flat directory of them."""
    path = url.split("?", 1)[0].strip("/")
    return f"{path.replace('/', '-') or 'index'}.html"


def _static_anchor(match: "re.Match[str]", files: frozenset) -> str:
    """Rewrite a captured-page link to its sibling file; render every other
    console route (and any external/absolute-host link) as plain text."""
    attrs, body = match.group("attrs"), match.group("body")
    found = HREF.search(attrs)
    if found is None or not found.group(1).startswith("/") or found.group(1).startswith("//"):
        return f'<span class="showcase-unavailable">{body}</span>'
    candidate = snapshot_name(found.group(1))
    if candidate not in files:
        return f'<span class="showcase-unavailable">{body}</span>'
    return "<a{}>{}</a>".format(HREF.sub(f'href="{candidate}"', attrs, count=1), body)


def transform_pages(
    pages: dict,
    anchor: str,
    source: str,
    home_url: str = HOME_URL,
    site_url: str = SITE_URL,
) -> dict:
    """Remove active behaviour, label the fixture, make page links portable."""
    files = frozenset(pages)
    label = (
        '<aside class="showcase-notice" role="note">'
        f'<a class="showcase-home" href="{html.escape(home_url)}">&larr; Home</a> '
        '<span class="showcase-title">waypost &middot; Read-only demo &middot; '
        f"<strong>Fixed fictional demo</strong> &middot; as of {html.escape(anchor)}</span> "
        f'<a class="showcase-product" href="{html.escape(site_url)}">Product details</a> '
        '<details class="showcase-about"><summary>About this demo</summary>'
        f"Fictional brand, jobs and publishing queue; nothing here is a real tenant. "
        f"Source {html.escape(source)}, as of {html.escape(anchor)}.</details></aside>"
    )
    style = (
        "<style>.showcase-notice{position:sticky;top:0;z-index:50;display:flex;"
        "flex-wrap:wrap;align-items:center;gap:.25rem .75rem;box-sizing:border-box;"
        "max-height:40px;overflow:hidden;background:#fff;color:#000;"
        "border-bottom:2px solid currentColor;padding:.4rem .8rem;margin:0 0 1rem;"
        "font-size:.8rem;line-height:1.2}"
        "@media(max-width:600px){.showcase-notice{max-height:none;overflow:visible}}"
        ".showcase-about{flex-basis:100%}.showcase-about[open]{max-height:50vh;overflow:auto}"
        "[id]{scroll-margin-top:40px}"
        "body{max-width:100%;overflow-wrap:anywhere}"
        ".showcase-table{overflow-x:auto;max-width:100%}"
        ".showcase-unavailable{opacity:.6}</style>"
    )
    meta = '<meta name="robots" content="noindex, nofollow">'
    transformed: dict = {}
    for name, body in pages.items():
        body = FORM.sub("", body)
        body = BUTTON.sub("", body)
        body = SCRIPT.sub("", body)
        body = ANCHOR_TAG.sub(lambda m: _static_anchor(m, files), body)
        body = TABLE.sub(_wrap_table, body)
        body = body.replace("</head>", f"{meta}{style}</head>", 1)
        body = body.replace("<body>", f"<body>{label}", 1)
        transformed[name] = "\n".join(line.rstrip() for line in body.splitlines()) + "\n"
    return transformed


def _seed_jobs(data_dir: Path) -> None:
    """A handful of fictional, already-finished console jobs -- never a real
    launch (no subprocess is Popen'd)."""
    from gomoveshift.console.jobs import JobStore

    store = JobStore(data_dir / "jobs.sqlite3")
    anchor_iso = datetime.combine(ANCHOR, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    fixtures = (
        ("drive", ["gomoveshift", "pipeline", "/fixtures/waypost-demo-day1"], "done", 0),
        ("shorts", ["gomoveshift", "shorts", "/fixtures/waypost-demo-day1"], "done", 0),
        ("sync-site", ["gomoveshift", "sync-site", "--site-path", "/fixtures/site"], "failed", 1),
    )
    for name, argv, status, rc in fixtures:
        job_id = store.create(name, argv, log_path="", brand=BRAND)
        store.update(
            job_id,
            status=status,
            rc=rc,
            created_at=anchor_iso,
            finished_at=anchor_iso,
        )


def _seed_publishing_queue(storage_path: Path) -> None:
    """A fictional publishing queue: one row awaiting manual publish (the
    review view's whole point), one queued, one already published."""
    from gomoveshift.publishing.queue import (
        PUBLISHED_AT_OPERATOR_CONFIRMED,
        enqueue,
        open_queue_db,
        transition,
    )

    scheduled_at = datetime.combine(ANCHOR, datetime.min.time(), tzinfo=timezone.utc).isoformat()
    conn = open_queue_db(storage_path)
    try:
        awaiting = enqueue(
            conn, "waypost-demo-day1", 0, "youtube", "h-awaiting", scheduled_at=scheduled_at
        )
        transition(conn, awaiting, "queued", "awaiting_manual_publish")
        enqueue(conn, "waypost-demo-day1", 1, "instagram", "h-queued", scheduled_at=scheduled_at)
        published = enqueue(
            conn, "waypost-demo-day1", 2, "tiktok", "h-published", scheduled_at=scheduled_at
        )
        transition(
            conn,
            published,
            "queued",
            "published",
            published_at_source=PUBLISHED_AT_OPERATOR_CONFIRMED,
        )
    finally:
        conn.close()


def _apply_env(values: "dict[str, str | None]") -> None:
    """Set each name to its value; ``None`` unsets it."""
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@contextmanager
def _scoped_env(values: "dict[str, str | None]") -> Iterator[None]:
    """Apply ``values``, then put every one of those names back as found.

    ``main`` also runs in-process (the tests call it), and the data dir the
    fixture points the console at is deleted once ``main`` returns."""
    saved = {name: os.environ.get(name) for name in values}
    _apply_env(values)
    try:
        yield
    finally:
        _apply_env(saved)


def render_pages(work_dir: Path) -> dict:
    """Seed a fictional fixture, boot the console in-process, and capture
    every route in ``PAGES``. The console environment and the working
    directory are restored on return."""
    from fastapi.testclient import TestClient

    data_dir = work_dir / "console_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    seo_storage = work_dir / "seo"
    seo_storage.mkdir(parents=True, exist_ok=True)
    config_root = work_dir / "config_root"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "config.yaml").write_text(f"seo:\n  storage_path: {seo_storage}\n")

    fixture_env = {
        "GMS_CONSOLE_DATA_DIR": str(data_dir),
        "GMS_CONSOLE_BRAND": BRAND,
        "GMS_CONSOLE_TOKEN": None,
        "GMS_CONSOLE_TICKER": None,
    }
    with _scoped_env(fixture_env):
        _seed_jobs(data_dir)
        _seed_publishing_queue(seo_storage)

        cwd = Path.cwd()
        os.chdir(config_root)
        try:
            from gomoveshift.console.app import create_app

            with TestClient(create_app()) as client:
                pages = {}
                for url in PAGES:
                    response = client.get(url)
                    if response.status_code != 200:
                        raise SystemExit(f"{url} -> {response.status_code}: the page set is stale")
                    pages[snapshot_name(url)] = response.text
                return pages
        finally:
            os.chdir(cwd)


def source_revision() -> str:
    """``git describe`` against the newest tag that reaches this checkout, or a
    short commit sha when there is none."""
    describe = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "describe", "--tags"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )
    if describe.returncode == 0 and describe.stdout.strip():
        return describe.stdout.strip()
    short = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return short.stdout.strip()


def write_bundle(out: Path, pages: dict, anchor: str, source: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for stale in sorted(out.glob("*.html")):
        stale.unlink()
    for name, body in sorted(pages.items()):
        (out / name).write_text(body, encoding="utf-8")
    sha256 = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in sorted(pages)}
    manifest = {
        "kind": KIND,
        "source": source,
        "anchor": anchor,
        "pages": len(pages),
        "files": sorted(pages),
        "sha256": sha256,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the read-only Waypost console showcase bundle."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchor", type=date.fromisoformat, default=ANCHOR, metavar="YYYY-MM-DD")
    parser.add_argument("--source", default=None)
    parser.add_argument("--home-url", default=HOME_URL)
    parser.add_argument("--site-url", default=SITE_URL)
    args = parser.parse_args(argv)
    anchor = args.anchor.isoformat()
    source = args.source or source_revision()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pages = render_pages(Path(tmp))
    except Exception as exc:  # noqa: BLE001 -- a plain, non-traceback failure message
        raise SystemExit(f"waypost-showcase: {exc}") from exc
    manifest = write_bundle(
        args.out,
        transform_pages(pages, anchor, source, args.home_url, args.site_url),
        anchor,
        source,
    )
    print(f"Showcase: {manifest['pages']} read-only pages at {anchor} -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
