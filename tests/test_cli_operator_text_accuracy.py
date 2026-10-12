"""Help text for the ``drive`` command must not mislead an operator into an
irreversible action.

Two defects: ``--youtube-upload``'s help claimed uploads were "disabled by
default", while ``--publish`` silently enables them (``resolve_youtube_upload``
in ``cli/handlers/drive.py``) — an operator staging a run with ``--publish``
alone would ship real videos to YouTube believing they had not opted in.
``--shorts-per-day``'s help claimed a fixed "default: 3" and implied the flag
raises the count; both consumers (``schedule_constants.required_shorts_per_lf``,
``phase_d_scheduling``) only ever truncate the region's configured windows.
"""

from __future__ import annotations

from gomoveshift.cli.parser import _build_parser


def _drive_parser():
    top = _build_parser()
    sub = top._subparsers._group_actions[0]
    return sub.choices["drive"]


def _help_for(parser, option: str) -> str:
    for action in parser._actions:
        if option in action.option_strings:
            assert action.help is not None
            return action.help
    raise AssertionError(f"{option} is not registered on this parser")


class TestPublishYoutubeUploadCoupling:
    def test_publish_help_documents_the_upload_coupling(self):
        help_text = _help_for(_drive_parser(), "--publish")
        assert "--youtube-upload" in help_text
        assert "--no-youtube-upload" in help_text

    def test_youtube_upload_help_does_not_claim_off_by_default(self):
        help_text = _help_for(_drive_parser(), "--youtube-upload")
        assert "disabled by default" not in help_text.lower()
        assert "--publish" in help_text
        assert "--no-youtube-upload" in help_text


class TestShortsPerDayHelpTruncatesOnly:
    def test_help_does_not_claim_a_fixed_numeric_default(self):
        help_text = _help_for(_drive_parser(), "--shorts-per-day")
        assert "default: 3" not in help_text

    def test_help_documents_truncation_only(self):
        help_text = _help_for(_drive_parser(), "--shorts-per-day")
        assert "truncat" in help_text.lower()
