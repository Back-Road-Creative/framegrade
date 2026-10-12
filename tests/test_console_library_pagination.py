"""The Media library page is bounded; the catalog it describes is not.

`build_context` handed every row it read straight to the template. A site
that registers a large photo catalog across several journeys would make
`/library` build and render a table of every row on one request, growing
with every card ingested. These pin a bounded page that still tells the truth
about how much there is.

Deliberately NOT fixed here: the read itself still scans every journey file.
That is a different change -- a rebuildable index -- to introduce only
when the scan is actually the bottleneck. What this bounds
is the page.
"""

import pytest
from fastapi.testclient import TestClient

from gomoveshift.console import library
from gomoveshift.console.app import create_app

ROUTING_YAML = "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"


def _journey(n: int) -> str:
    photos = "\n".join(f"        - photo-{i:04d}.jpg" for i in range(n))
    return (
        "---\n"
        "days:\n"
        "- date: '2025-01-01'\n"
        "  photos:\n"
        "  - location: Place A\n"
        "    photos:\n" + photos + "\n"
        "---\nBody text\n"
    )


def _site(tmp_path, monkeypatch, photo_count: int, *, routing: bool = False):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(_journey(photo_count))
    if routing:
        (site / "data").mkdir(parents=True, exist_ok=True)
        (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    monkeypatch.setenv("GMS_SITE_PATH", str(site))
    return site


def test_the_page_never_renders_more_than_one_page_of_rows(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, 250)

    context = library.build_context()

    assert len(context["entries"]) == library.PER_PAGE
    assert library.PER_PAGE < 250


def test_the_count_describes_the_whole_collection_not_the_page(tmp_path, monkeypatch):
    """A page that says "100 items" when there are 250 is worse than one that
    says nothing: it reads as a complete catalog."""
    _site(tmp_path, monkeypatch, 250)

    context = library.build_context()

    assert context["total"] == 250


def test_the_second_page_is_the_next_rows_and_repeats_none_of_the_first(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, 250)

    first = {e.catalog_id for e in library.build_context(page=1)["entries"]}
    second = {e.catalog_id for e in library.build_context(page=2)["entries"]}

    assert len(first) == len(second) == library.PER_PAGE
    assert first.isdisjoint(second)


def test_the_last_page_holds_the_remainder(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, 250)

    context = library.build_context(page=3)

    assert len(context["entries"]) == 250 - 2 * library.PER_PAGE
    assert context["has_next"] is False
    assert context["has_prev"] is True


@pytest.mark.parametrize("page", [0, -3, 99, "abc", None, ""])
def test_a_page_number_that_is_not_a_page_lands_somewhere_real(tmp_path, monkeypatch, page):
    """`page` arrives from a query string, so every value is possible. None of
    them may 500, and none may render an empty table that reads like an empty
    catalog -- the operator gets a real page of real rows."""
    _site(tmp_path, monkeypatch, 250)

    context = library.build_context(page=page)

    assert context["entries"]
    assert 1 <= context["page"] <= context["pages"]


def test_pagination_applies_after_the_filters(tmp_path, monkeypatch):
    """Slicing before filtering would silently drop matches that live past the
    first page -- a filter that finds nothing when the match is row 400."""
    _site(tmp_path, monkeypatch, 250, routing=True)

    context = library.build_context(media_type="video")

    assert context["total"] == 1
    assert len(context["entries"]) == 1
    assert context["pages"] == 1


def test_a_collection_smaller_than_one_page_is_shown_whole(tmp_path, monkeypatch):
    """The accept-case. Most brands are small, and a pager on a five-row table
    is noise -- the page must not grow controls it does not need."""
    _site(tmp_path, monkeypatch, 3)

    context = library.build_context()

    assert len(context["entries"]) == 3
    assert context["total"] == 3
    assert context["pages"] == 1
    assert context["has_next"] is False
    assert context["has_prev"] is False


def test_the_first_page_is_what_you_get_without_asking(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, 250)

    default = [e.catalog_id for e in library.build_context()["entries"]]
    explicit = [e.catalog_id for e in library.build_context(page=1)["entries"]]

    assert default == explicit


def test_an_unreadable_site_still_answers_with_a_whole_context(tmp_path, monkeypatch):
    """`build_context` promises never to raise. The pager keys have to be
    present on the degraded paths too, or the template raises instead."""
    monkeypatch.setenv("GMS_SITE_PATH", str(tmp_path / "does-not-exist"))

    context = library.build_context()

    for key in ("total", "page", "pages", "has_next", "has_prev"):
        assert key in context


def test_the_route_says_which_slice_of_what_it_is_showing(tmp_path, monkeypatch):
    _site(tmp_path, monkeypatch, 250)

    text = TestClient(create_app()).get("/library").text

    assert f"1-{library.PER_PAGE} of 250" in text


def test_the_pager_links_keep_the_filters_that_are_active(tmp_path, monkeypatch):
    """A "next" link that drops the filter takes the operator somewhere else
    entirely, and the rows they were reading are gone."""
    _site(tmp_path, monkeypatch, 250)

    text = TestClient(create_app()).get("/library", params={"media_type": "photo"}).text

    assert "media_type=photo" in text
    assert "page=2" in text


def test_a_small_collection_renders_no_pager_links(tmp_path, monkeypatch):
    """The accept-case on the page itself."""
    _site(tmp_path, monkeypatch, 3)

    text = TestClient(create_app()).get("/library").text

    assert "page=2" not in text
