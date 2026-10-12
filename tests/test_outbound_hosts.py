"""A verified, code-generated inventory of every outbound host this service's
source can name (docs/outbound.md, tools/build_outbound_hosts.py).

Why: the hand-maintained allowlist prose in
docs/pipeline/84-network-allowlist-credential-stores.md can drift from code.
This module scans every gomoveshift/**/*.py string literal that
names a host and requires each one to be catalogued -- a new host fails loudly
here, naming the file:line, instead of shipping undocumented egress silently.
"""

from tools.build_outbound_hosts import (
    INVENTORY,
    render_doc,
    scan_hosts,
    undocumented,
)

DOC_PATH = __import__("pathlib").Path(__file__).resolve().parent.parent / "docs" / "outbound.md"


def test_every_host_named_in_code_is_in_the_inventory():
    """Every host literal the scanner finds in gomoveshift/**/*.py must be
    catalogued in INVENTORY -- a new host is a hard failure naming file:line,
    never a silent gap."""
    hosts = scan_hosts()
    missing = undocumented(hosts)
    assert not missing, (
        "hosts found in gomoveshift/**/*.py but missing from "
        "tools/build_outbound_hosts.py's INVENTORY:\n"
        + "\n".join(f"  {s.path}:{s.lineno}: {s.host!r}" for s in missing)
    )


def test_inventory_has_no_stale_entries():
    """A host removed from code but left in INVENTORY is a stale claim about
    what the service actually contacts -- keep the two in lockstep both ways."""
    hosts = {s.host for s in scan_hosts()}
    stale = sorted(set(INVENTORY) - hosts)
    assert not stale, f"INVENTORY entries no longer found in gomoveshift/**/*.py: {stale}"


def test_docs_outbound_matches_generator():
    """docs/outbound.md is generated, never hand-edited -- this is the same
    --check pattern as tools/build_internals_index.py."""
    hosts = scan_hosts()
    rendered = render_doc(hosts)
    current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.is_file() else ""
    assert current == rendered, (
        f"{DOC_PATH} is stale or hand-edited. Run:\n  python3 tools/build_outbound_hosts.py"
    )


def test_loopback_hosts_never_appear():
    hosts = {s.host for s in scan_hosts()}
    assert not hosts & {"127.0.0.1", "localhost", "::1"}
