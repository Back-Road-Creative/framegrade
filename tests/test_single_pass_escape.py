"""Tests for SinglePassProcessor._escape_text drawtext escaping.

The day-master long-form overlay (_build_location_overlay) escapes the burned
location label via _escape_text before handing it to FFmpeg drawtext. A label
containing brace characters ({ or }) would otherwise be interpreted as a
drawtext format specifier (e.g. %{pts}, %{n}) rather than rendered literally.

_escape_text must match the canonical escape_ffmpeg_text helper: it escapes
the FIVE characters drawtext treats specially — \\ : ' % { } — so that
none of them leak through as render-time directives.
"""

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import SinglePassProcessor


class TestEscapeTextBraces:
    def test_braces_escaped_and_prior_escapes_preserved(self):
        proc = SinglePassProcessor(Config())
        out = proc._escape_text("Route {A}: 50%")
        # New behaviour: braces must be escaped so they are not treated as
        # drawtext format-specifier framing.
        assert "\\{" in out
        assert "\\}" in out
        # Regression guard: the pre-existing escapes must still hold.
        assert "\\:" in out  # colon (option separator)
        assert "\\%" in out  # percent (format-specifier prefix)

    def test_backslash_escaped(self):
        proc = SinglePassProcessor(Config())
        out = proc._escape_text("a\\b")
        assert "\\\\" in out

    def test_single_quote_escaped(self):
        proc = SinglePassProcessor(Config())
        out = proc._escape_text("O'Brien")
        # Matches the canonical escaper's single-quote form.
        assert "'\\''" in out
