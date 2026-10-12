"""The library list opens the asset it names.

Everything either side of selection already shipped. `/library` lists a mixed
photo/video collection and `/library/{catalog_id}` renders a preview
and the asset's provenance. Between them there was nothing: the
list printed each `catalog_id` as inert text, and `grep -c 'href="/library/'`
over `library.html` returned 0. The only hrefs on the page were the pager and
"Back", so the asset page was reachable only by constructing a URL by hand --
which is how every existing test reaches it, and why the gap was invisible.

So the acceptance clause "a selected asset opens preview and history" failed
at the FIRST word, with both halves built.

The link is URL-quoted, and that is the load-bearing part rather than a
nicety. A `catalog_id` is `photo:<journey>:<date>:<filename>` built from real
filenames, so one `#` in a filename truncates the href at the fragment and
the link silently opens the wrong asset -- or a 404 -- while looking correct
in the page source. Quoting makes that unrepresentable instead of relying on
filenames staying tidy.
"""

from fastapi.testclient import TestClient

from gomoveshift.console.app import create_app

JOURNEY = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: [photo-001.jpg]\n"
    "    photo_hashes: {photo-001.jpg: abc123}\nphoto_provenance: {photo-001.jpg: raw}\n---\nBody\n"
)

# A filename carrying the two characters that break an unquoted href: `#`
# starts a fragment, so everything after it never reaches the server, and a
# space is not legal in a URL at all.
JOURNEY_AWKWARD_NAME = (
    "---\ndays:\n- date: '2025-01-01'\n  photos:\n  - photos: ['a #2 shot.jpg']\n"
    "    photo_hashes: {}\n---\nBody\n"
)

ROUTING_YAML = "routes: {vid-1: {journey: journey-a, date: '2025-01-01'}}\n"


def _site(tmp_path, journey_body):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "journey-a.md").write_text(journey_body)
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "data" / "video_routing.yaml").write_text(ROUTING_YAML)
    return site


def _client():
    return TestClient(create_app())


def test_each_listed_asset_links_to_its_own_page(monkeypatch, tmp_path):
    """The defect: the id was printed, not linked."""
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path, JOURNEY)))

    body = _client().get("/library").text

    assert 'href="/library/photo%3Ajourney-a%3A2025-01-01%3Aphoto-001.jpg"' in body


def test_the_video_row_links_too(monkeypatch, tmp_path):
    """Both media types, since the collection is mixed and the asset page
    serves both."""
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path, JOURNEY)))

    body = _client().get("/library").text

    assert 'href="/library/video%3Avid-1"' in body


def test_the_link_is_followable_and_lands_on_that_asset(monkeypatch, tmp_path):
    """End to end, and the test the hand-built URLs elsewhere cannot give:
    the href the PAGE produced is fetched, rather than one written here."""
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path, JOURNEY)))
    client = _client()
    body = client.get("/library").text
    href = body.split('href="/library/photo')[1].split('"')[0]

    response = client.get(f"/library/photo{href}")

    assert response.status_code == 200
    assert "abc123" in response.text


def test_an_awkward_filename_still_produces_a_link_that_works(monkeypatch, tmp_path):
    """`#` and a space in a filename. Unquoted, the href truncates at the
    fragment and the page looks fine while the link is wrong."""
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path, JOURNEY_AWKWARD_NAME)))
    client = _client()
    body = client.get("/library").text

    assert '#2 shot.jpg"' not in body, "the raw `#` reached the href unquoted"
    href = body.split('href="/library/photo')[1].split('"')[0]
    assert client.get(f"/library/photo{href}").status_code == 200


def test_the_id_is_still_readable_as_text(monkeypatch, tmp_path):
    """The id is the thing an operator quotes into a bug report, so the link
    must not replace it with a friendlier word."""
    monkeypatch.setenv("GMS_SITE_PATH", str(_site(tmp_path, JOURNEY)))

    body = _client().get("/library").text

    assert ">photo:journey-a:2025-01-01:photo-001.jpg<" in body
