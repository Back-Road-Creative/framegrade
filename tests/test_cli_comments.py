"""``gomoveshift comments pull`` — parser wiring + pull idempotency.
Mocks the YouTube client; never touches live creds.
"""

from __future__ import annotations

from gomoveshift.cli.handlers.comments import run_comments_pull


def test_subparsers_registered():
    from gomoveshift.cli import _COMMAND_MAP, _build_parser

    assert "comments" in _COMMAND_MAP
    parser = _build_parser()
    args = parser.parse_args(["comments", "pull", "--channel-id", "UC123"])
    assert args.command == "comments"
    assert args.comments_command == "pull"
    assert args.channel_id == "UC123"
    assert args.since == "90d"


class _FakeService:
    def __init__(self, items):
        self._items = items

    def commentThreads(self):
        return self

    def list(self, **kwargs):
        return self

    def execute(self):
        return {"items": self._items}


def _thread(comment_id, published_at, video_id="vid1"):
    return {
        "snippet": {
            "topLevelComment": {
                "id": comment_id,
                "snippet": {
                    "videoId": video_id,
                    "authorDisplayName": "A Viewer",
                    "textDisplay": "actually that's Kerry",
                    "publishedAt": published_at,
                    "likeCount": 2,
                },
            }
        }
    }


def test_pull_writes_lines_and_is_idempotent(tmp_path):
    items = [
        _thread("c1", "2026-08-20T00:00:00Z"),
        _thread("c2", "2026-08-21T00:00:00Z"),
        _thread("c3", "2026-08-22T00:00:00Z"),
    ]
    out_path = tmp_path / "comments.jsonl"
    service = _FakeService(items)

    rc = run_comments_pull(channel_id="UC123", since="9000d", out_path=out_path, service=service)
    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    # Re-pull with the same 3 threads adds 0 new lines.
    rc = run_comments_pull(channel_id="UC123", since="9000d", out_path=out_path, service=service)
    assert rc == 0
    lines_again = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_again) == 3


def test_pull_requires_channel_id(tmp_path):
    rc = run_comments_pull(channel_id=None, since="90d", out_path=tmp_path / "c.jsonl")
    assert rc == 2


def test_triage_subparser_registered():
    from gomoveshift.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["comments", "triage", "--in", "in.jsonl", "--out", "out.md"])
    assert args.command == "comments"
    assert args.comments_command == "triage"
    assert str(args.in_path) == "in.jsonl"
    assert str(args.out_path) == "out.md"
