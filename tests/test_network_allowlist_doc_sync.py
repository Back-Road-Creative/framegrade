"""Every host an adapter's ALLOWED_UPLOAD_HOSTS actually declares must appear in
docs/pipeline/84-network-allowlist-credential-stores.md (the network allowlist
module file, split out of docs/pipeline-internals.md so parallel PRs stop
colliding on one file -- see tools/build_internals_index.py). The frozensets in
publishing/{facebook,instagram,tiktok}_adapter.py are the enforced truth; the doc
section is a prose copy of that truth for humans, and a prose copy that isn't
checked against its source drifts silently (the Meta hosts shipped and were
enforced in code for weeks before the doc named them). This test makes that
drift a red test instead of a stale paragraph nobody notices.
"""

from pathlib import Path

from gomoveshift.publishing.facebook_adapter import ALLOWED_UPLOAD_HOSTS as FB_HOSTS
from gomoveshift.publishing.instagram_adapter import ALLOWED_UPLOAD_HOSTS as IG_HOSTS
from gomoveshift.publishing.tiktok_adapter import ALLOWED_UPLOAD_HOSTS as TIKTOK_HOSTS

DOC_PATH = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "pipeline"
    / "84-network-allowlist-credential-stores.md"
)
SECTION_HEADING = "## Network allowlist & credential stores"


def _allowlist_section_text() -> str:
    text = DOC_PATH.read_text()
    start = text.index(SECTION_HEADING)
    # The module file holds exactly this one section now, so there is no
    # trailing "\n## " boundary to find -- fall back to end of file.
    try:
        end = text.index("\n## ", start + len(SECTION_HEADING))
    except ValueError:
        end = len(text)
    return text[start:end]


def test_all_declared_upload_hosts_are_documented():
    """Every host any publishing adapter allows uploads to is named in the doc's
    allowlist section — nothing enforced in code may go undocumented."""
    declared_hosts = FB_HOSTS | IG_HOSTS | TIKTOK_HOSTS
    section = _allowlist_section_text()
    missing = sorted(host for host in declared_hosts if host not in section)
    assert not missing, (
        "hosts enforced by an adapter's ALLOWED_UPLOAD_HOSTS but missing from "
        f"the '{SECTION_HEADING}' section of docs/pipeline/84-network-allowlist-credential-stores.md: {missing}"
    )


def test_tiktok_api_host_is_documented():
    """The TikTok API endpoints (token, creator info, publish init, status, video
    query) sit outside the upload-host allowlist, so the first test never saw
    them; each endpoint constant's host must still be named in the doc."""
    from urllib.parse import urlparse

    from gomoveshift.publishing import tiktok_adapter, tiktok_auth
    from gomoveshift.tracking import collector

    urls = [
        tiktok_adapter.CREATOR_INFO_URL,
        tiktok_adapter.INIT_URL,
        tiktok_adapter.INBOX_INIT_URL,
        tiktok_adapter.STATUS_URL,
        tiktok_auth.TOKEN_URL,
        collector._TIKTOK_QUERY_URL,
    ]
    section = _allowlist_section_text()
    missing = sorted({urlparse(u).hostname for u in urls} - {None} - _hosts_in(section))
    assert not missing, (
        "TikTok API hosts the code calls but missing from "
        f"the '{SECTION_HEADING}' section of docs/pipeline/84-network-allowlist-credential-stores.md: {missing}"
    )


def _hosts_in(section: str) -> set[str]:
    """Bolded host names in the section (`**host**`), exact, never a substring match,
    so `open-upload.tiktokapis.com` does not vouch for `open.tiktokapis.com`."""
    import re

    return set(re.findall(r"\*\*([a-z0-9.-]+\.[a-z]{2,})\*\*", section))


def test_every_route_story_source_host_is_documented():
    """Route Story reads public map sources over the network too, so the same rule
    applies: a host the fetchers contact and the doc does not name is drift."""
    from gomoveshift.pipeline.route_story.sources import SOURCE_HOSTS

    section = _allowlist_section_text()
    missing = sorted(host for host in SOURCE_HOSTS if host not in section)
    assert not missing, (
        "hosts Route Story's source fetchers contact but missing from "
        f"the '{SECTION_HEADING}' section of docs/pipeline/84-network-allowlist-credential-stores.md: {missing}"
    )
