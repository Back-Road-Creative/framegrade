### Page layout (`console/templates/_base.html`)

Every console page extends one base template. It owns the document shell -- doctype,
charset, a `width=device-width` viewport (absent from the page templates before the base existed, so each
one rendered at desktop width on a phone) and a small token block that lets the wide
tables scroll inside the page instead of the page scrolling sideways. A child template
supplies `{% block title %}` and `{% block content %}`, and `{% block head_extra %}` for
the one page that needs more head markup (`jobs.html`'s conditional meta-refresh while a
job is running). `tests/test_console_base_layout.py` fails any page that hand-rolls its
own `<html>`/`<head>`/`<body>` instead, and asserts a rendered response actually carries
the viewport. All console templates now extend the base with no exemptions.

The base also owns the console's ONLY navigation. Before it, every page ended in a
single `<a href="/">Back</a>` and nothing else: reaching Jobs from Publishing meant going
home first, and there was no `<nav>` landmark anywhere in the console for a screen reader
to jump to. `_base.html` now renders a labelled `<nav class="console-nav"
aria-label="Console sections">` above `<main>`, after the skip link -- which is exactly
what the skip link is for, since the nav is the page chrome a keyboard user would
otherwise re-traverse on every navigation. Its links come from `nav_entries`, set once in
`create_app` as a Jinja **global** (`templates.env.globals`) rather than passed per route,
because it is brand-level configuration fixed for the life of the app and because the
error page must render it too without every route remembering to supply it. It is the
same brand-filtered list `_require_tab` gates on, so a tab a brand switched off is never
offered a link that 404s -- `test_the_nav_offers_every_enabled_tab_and_nothing_else`
compares the rendered nav's hrefs against `NAV_ENTRIES` to keep the nav from growing a
second source of truth. `aria-current="page"` marks the current tab by matching the
**first path segment**, not the whole path: `/jobs/7` and `/drives/gates/pending` are
still the Jobs and Drives tabs, and whole-path matching would mark nothing current on
every detail page. The nav wraps at narrow widths rather than scrolling sideways -- the
tab names at 390px wrap to several rows of links, where a scrolling bar would hide half of them
behind a gesture nothing announces. The per-page "Back" links are left alone; they are
harmless once a real nav exists.

The base also owns three accessibility fixes shared by every page (`tests/
test_console_accessibility.py`): `{% block content %}` is wrapped in a `<main id="main-
content">` landmark; the first focusable element in `<body>` is a skip link (`<a href="#
main-content">`) so a keyboard user doesn't traverse the whole page chrome per
navigation; and a small inline script WRAPS every bare `<table>` in a new
`[data-table-scroll]` `<div>` -- never applying `role="region"` to the `<table>` itself,
since an explicit role overrides an element's implicit one and would strip its rows/cells
of their owning context (row/column nav, header association). That wrapper gets
`tabindex="0"`, `role="region"` and an `aria-label` named from the table's own nearest
preceding `<h1>`/`<h2>`/`<h3>` (walking up through ancestors, falling back to the page's
first heading only if none is found -- so a page with several tables, like `schedule.html`'s
three, gets three distinct names, not one repeated name). This exists because
`.table-scroll, table { overflow-x: auto; }` (above) makes each table its own scrolling
region and an unfocusable scrolling region is unreachable by keyboard (axe:
`scrollable-region-focusable`); the CSS restores `display: table` on the `<table>` once
it's wrapped, and only the no-JS fallback path leaves `display: block` on a bare `<table>`.
All three live in the base rather than per page because none of the page templates can
be edited by hand for this alone without churning every one of them for a single shared
fix -- the base is the one seam that already guarantees every page gets it, the same
reason the viewport meta lives there. A visible `:focus-visible` outline on the tagged
wrapper makes the newly-reachable focus stop
visible to sighted keyboard users too.

**Every form control in every console template carries an accessible name, and a guard
keeps it that way** (`tests/test_console_field_labels.py`). A control counts as named by
exactly the three mechanisms that actually produce an accessible name -- a `<label for=>`
pointing at its `id`, an `aria-label`, or the control sitting directly inside its own
`<label>` -- and `hidden`/`submit`/`button`/`image` inputs are exempt because they carry
no name or are named by their own text. A `placeholder` is explicitly NOT a name: it
disappears the moment anyone types and fails the axe `label` check, and several console
fields ask for a filesystem path or a date key where guessing wrong is the whole problem.
The guard fixes nothing on the day it lands -- every template is already clean. It exists
because it is robust by construction: a new page cannot add an unnamed field and have the suite stay green. It carries no
exemption list on purpose (an exemption nobody rechecks goes stale), and
it proves itself in both directions -- one test feeds it an unnamed control and requires
the failure, another feeds it all three naming forms and requires silence.
