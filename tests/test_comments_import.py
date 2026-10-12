"""``gomoveshift comments import --platform tiktok`` — CSV/txt ingest into the
shared comments.jsonl ledger, with no TikTok API involved."""

from __future__ import annotations

from gomoveshift.cli.handlers.comments import run_comments_import
from gomoveshift.pipeline.comments import build_triage_report, import_tiktok_comments


def test_subparser_registered():
    from gomoveshift.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(
        ["comments", "import", "--platform", "tiktok", "in.csv", "--video-id", "v1"]
    )
    assert args.command == "comments"
    assert args.comments_command == "import"
    assert args.platform == "tiktok"
    assert str(args.file) == "in.csv"
    assert args.video_id == "v1"


def test_import_csv_writes_lines_and_is_idempotent(tmp_path):
    csv_path = tmp_path / "tiktok.csv"
    csv_path.write_text(
        "url,author,text,date\n"
        "https://www.tiktok.com/@example_account/video/7000000000011,Alice,nice drive,2026-08-20\n"
        "https://www.tiktok.com/@example_account/video/7000000000012,Bob,actually that's Kerry,2026-08-21\n"
        "https://www.tiktok.com/@example_account/video/7000000000013,Carol,love this,2026-08-22\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "comments.jsonl"

    rc = run_comments_import(platform="tiktok", in_path=csv_path, out_path=out_path)
    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    rc = run_comments_import(platform="tiktok", in_path=csv_path, out_path=out_path)
    assert rc == 0
    lines_again = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines_again) == 3


def test_csv_url_extracts_trailing_numeric_video_id(tmp_path):
    csv_path = tmp_path / "tiktok.csv"
    csv_path.write_text(
        "url,author,text\nhttps://www.tiktok.com/@example_account/video/7000000000011,Alice,hi\n",
        encoding="utf-8",
    )
    rows = import_tiktok_comments(csv_path)
    assert len(rows) == 1
    assert rows[0]["video_id"] == "7000000000011"
    assert rows[0]["platform"] == "tiktok"
    assert rows[0]["like_count"] is None


def test_txt_requires_video_id(tmp_path):
    txt_path = tmp_path / "tiktok.txt"
    txt_path.write_text("Alice: this was in the wrong spot\n", encoding="utf-8")

    rc = run_comments_import(platform="tiktok", in_path=txt_path, out_path=tmp_path / "c.jsonl")
    assert rc == 2


def test_txt_with_video_id_parses_author_text_and_bare_lines(tmp_path):
    txt_path = tmp_path / "tiktok.txt"
    txt_path.write_text(
        "Alice: actually that's Kerry\nlove this drive\n\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "comments.jsonl"

    rc = run_comments_import(platform="tiktok", in_path=txt_path, video_id="v99", out_path=out_path)
    assert rc == 0
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_triage_flags_a_tiktok_row(tmp_path):
    csv_path = tmp_path / "tiktok.csv"
    csv_path.write_text(
        "video_id,author,text,date\nv1,Alice,actually that's Kerry not Cork,2026-08-20\n",
        encoding="utf-8",
    )
    rows = import_tiktok_comments(csv_path)
    report = build_triage_report(rows, {})
    assert "cue:actually" in report
    assert "tiktok" in report
