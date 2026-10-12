"""The operator-supplied forward-geocode query must be sanitized.

A hostile/odd location string (control characters, excessive length) must be
stripped/capped before it is interpolated into the Nominatim search URL, and
URL-encoded so query separators can't break out of the query string.
"""

from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.chapters import _nominatim_search


def _capturing_urlopen(captured):
    """Return a urlopen stub that records the requested URL and returns []."""

    def _stub(req, *args, **kwargs):
        captured["url"] = req.full_url
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        resp.read.return_value = b"[]"
        return resp

    return _stub


def test_control_chars_stripped_from_query():
    """Newlines/control chars in the location string must not reach the URL."""
    captured: dict[str, str] = {}
    with patch("urllib.request.urlopen", _capturing_urlopen(captured)):
        with patch("time.sleep"):
            _nominatim_search("Kenmare\n\r\x00, Kerry")
    url = captured["url"]
    # No raw control characters anywhere in the built URL.
    assert "\n" not in url and "\r" not in url and "\x00" not in url
    # And not smuggled in as their percent-encodings either.
    assert "%0A" not in url.upper() and "%0D" not in url.upper() and "%00" not in url.upper()


def test_overlong_query_is_capped():
    """A pathologically long location string must be capped before the request."""
    captured: dict[str, str] = {}
    with patch("urllib.request.urlopen", _capturing_urlopen(captured)):
        with patch("time.sleep"):
            _nominatim_search("A" * 5000)
    url = captured["url"]
    # The query value must be bounded, not a 5000-char URL.
    assert len(url) < 1000


def test_query_separators_are_encoded():
    """`&`/`#`/`=` in the location must be percent-encoded, never literal in the query."""
    captured: dict[str, str] = {}
    with patch("urllib.request.urlopen", _capturing_urlopen(captured)):
        with patch("time.sleep"):
            _nominatim_search("Foo&limit=999#frag")
    url = captured["url"]
    query = url.split("?", 1)[1]
    # The injected separators must be encoded inside the q value, so the only
    # literal '&' are our own param separators and there is no extra param.
    assert "limit=999" not in query
    assert "#" not in url
    assert query.count("&") == 2  # format and limit, our own params only
