"""An empty library says which kind of empty it is.

The catalog reads two sources -- `content/journeys/*.md` and
`data/video_routing.yaml` -- and treats a source that is not there as that
source's own empty subset. That is the right call for the read: a site with no
videos yet must not 500. But it collapses two situations an operator has to
tell apart, and the page said the same "Nothing here yet" for both:

* the source exists and holds nothing -- normal, nothing to do;
* the source is not there at all -- this checkout has no journeys directory,
  which usually means `GMS_SITE_PATH` points at the wrong tree or the site has
  never been synced. Something IS to be done, and nothing said so.

`damaged` already separates a third case (present, would not parse). This adds
the missing one, and keeps all three exclusive: absent is not damaged, and
neither is empty.
"""

import pytest

from gomoveshift.catalog.reader import read_catalog
from gomoveshift.console import library

JOURNEY_MD = (
    "---\n"
    "days:\n"
    "- date: '2025-01-01'\n"
    "  photos:\n"
    "  - location: Place A\n"
    "    photos:\n"
    "        - photo-0001.jpg\n"
    "---\nBody\n"
)
ROUTING_YAML = "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"


def _site(tmp_path, *, journeys="full", routing="full"):
    """``full`` = present with content, ``empty`` = present and holding
    nothing, ``absent`` = not there, ``damaged`` = present and unparseable."""
    site = tmp_path / "site"
    site.mkdir(parents=True, exist_ok=True)
    jdir = site / "content" / "journeys"
    if journeys != "absent":
        jdir.mkdir(parents=True)
        if journeys == "full":
            (jdir / "journey-a.md").write_text(JOURNEY_MD)
        elif journeys == "damaged":
            (jdir / "journey-a.md").write_text("---\ndays: [unclosed\n---\n")
    if routing != "absent":
        (site / "data").mkdir(parents=True, exist_ok=True)
        target = site / "data" / "video_routing.yaml"
        if routing == "full":
            target.write_text(ROUTING_YAML)
        elif routing == "empty":
            target.write_text("routes: {}\n")
        elif routing == "damaged":
            target.write_text("routes: {unclosed\n")
    return site


def _absent_paths(result):
    return {a.path for a in result.absent}


def test_a_missing_journeys_directory_is_reported_absent(tmp_path):
    site = _site(tmp_path, journeys="absent")

    result = read_catalog(site)

    assert str(site / "content" / "journeys") in _absent_paths(result)


def test_a_journeys_directory_holding_no_files_is_not_absent(tmp_path):
    """A real, empty subset. The operator has nothing to fix."""
    site = _site(tmp_path, journeys="empty", routing="empty")

    result = read_catalog(site)

    assert result.absent == []
    assert result.entries == []


def test_a_missing_routing_file_is_reported_absent(tmp_path):
    site = _site(tmp_path, routing="absent")

    result = read_catalog(site)

    assert str(site / "data" / "video_routing.yaml") in _absent_paths(result)


def test_a_routing_file_holding_no_routes_is_not_absent(tmp_path):
    site = _site(tmp_path, routing="empty")

    result = read_catalog(site)

    assert result.absent == []


def test_a_source_that_will_not_parse_is_damaged_and_not_absent(tmp_path):
    """The three states are exclusive. A truncated routing file is there --
    calling it absent would send an operator looking for a file they have."""
    site = _site(tmp_path, routing="damaged")

    result = read_catalog(site)

    assert [d.path for d in result.damaged] == [str(site / "data" / "video_routing.yaml")]
    assert result.absent == []


def test_a_filter_never_reports_a_source_it_did_not_read(tmp_path):
    """``media_type="photo"`` never opens the routing file, so it cannot say
    anything about whether it is there."""
    site = _site(tmp_path, journeys="full", routing="absent")

    result = read_catalog(site, media_type="photo")

    assert result.absent == []


def test_both_sources_absent_are_both_reported(tmp_path):
    site = _site(tmp_path, journeys="absent", routing="absent")

    result = read_catalog(site)

    assert _absent_paths(result) == {
        str(site / "content" / "journeys"),
        str(site / "data" / "video_routing.yaml"),
    }


def test_a_fully_populated_site_reports_nothing_absent(tmp_path):
    site = _site(tmp_path)

    result = read_catalog(site)

    assert result.absent == []
    assert result.entries


@pytest.fixture
def _site_env(tmp_path, monkeypatch):
    def _make(**kwargs):
        monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path, **kwargs)))

    return _make


def test_the_page_names_the_source_that_is_not_there(_site_env):
    _site_env(journeys="absent", routing="absent")

    context = library.build_context()

    sources = {a["source"] for a in context["absent"]}
    assert sources == {"journeys", "the video routing file"}


def test_the_page_says_nothing_absent_for_a_site_that_is_merely_empty(_site_env):
    """The accept case: "Nothing here yet" has to keep meaning exactly that."""
    _site_env(journeys="empty", routing="empty")

    context = library.build_context()

    assert context["absent"] == []
    assert context["entries"] == []


def test_an_unconfigured_site_reports_no_absent_sources(_site_env, monkeypatch):
    """Nothing was read, so nothing can be said about it -- and the template
    still needs the key to render at all."""
    monkeypatch.delenv("GMS_SITE_PATH", raising=False)

    context = library.build_context()

    assert context["absent"] == []
