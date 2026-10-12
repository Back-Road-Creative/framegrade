"""One call embeds a video AND records it in the site's long-form library.

``data/videos.yaml`` used to be maintained by ``sync-site`` alone, so the other
journey writers (``publish-journal`` and its ``--replay-pending`` cron flush)
embedded published long-forms the library never learned about. The site's
journey-embed deploy check only WARNS about that, so it ships silently.

Both directions matter, because the library is a mirror of *public long-forms
only*: a real LF embed MUST gain a library row, and a Short / stream / held
embed (which journeys legitimately carry) MUST NOT.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from gomoveshift.cli.handlers.publish_journal import run_replay_pending
from gomoveshift.pipeline.journal_publisher import MergePlan, VideoEntry, apply_merge
from gomoveshift.pipeline.pending_journal import PENDING_FILENAME

FIXTURE = Path(__file__).parent / "fixtures" / "journals" / "sample_trip.md"

# A human-authored row that already lives in the library and must survive.
_HUMAN_ROW = {
    "id": "EXISTING_VID_2",
    "title": "Human Shortened Title",
    "date": "2025-09-04",
    "duration_min": 11,
    "summary": "hand written",
}


def _make_site(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    (site / "content" / "journeys").mkdir(parents=True)
    (site / "data").mkdir(parents=True)
    shutil.copy(FIXTURE, site / "content" / "journeys" / "sample-trip.md")
    (site / "data" / "videos.yaml").write_text(yaml.safe_dump({"videos": [dict(_HUMAN_ROW)]}))
    return site


def _seed_pending(tmp_path: Path, site: Path) -> Path:
    """A ready long-form and a ready Short, both routed at the same journey day."""
    out = tmp_path / "out"
    out.mkdir()
    journey = str(site / "content" / "journeys" / "sample-trip.md")
    pending = out / PENDING_FILENAME
    pending.write_text(
        json.dumps(
            [
                {
                    "youtube_id": "REPLAY_LF",
                    "publish_at": "2025-01-01T00:00:00Z",
                    "title": "Replayed Long Form",
                    "duration": "12:34",  # 754s > SHORT_MAX_SECONDS -> real_lf
                    "day_date": "2025-09-03",
                    "journey_path": journey,
                    "lf_index": 1,
                },
                {
                    "youtube_id": "REPLAY_SHORT",
                    "publish_at": "2025-01-01T00:00:00Z",
                    "title": "Replayed Short",
                    "duration": "01:12",  # 72s <= SHORT_MAX_SECONDS -> short
                    "day_date": "2025-09-03",
                    "journey_path": journey,
                    "lf_index": 2,
                },
            ]
        )
    )
    return pending


def _library_rows(site: Path) -> dict[str, dict]:
    data = yaml.safe_load((site / "data" / "videos.yaml").read_text()) or {}
    return {row["id"]: row for row in data.get("videos") or []}


def test_replay_flush_records_the_long_form_and_leaves_the_short_out(tmp_path):
    """THE defect: ``--replay-pending`` embedded a published long-form that the
    library never learned about. The Short embedded alongside it proves the
    library stays a long-form mirror rather than gaining every embed."""
    site = _make_site(tmp_path)
    rc = run_replay_pending(pending_files=[_seed_pending(tmp_path, site)], apply=True)
    assert rc == 0

    journey_text = (site / "content" / "journeys" / "sample-trip.md").read_text()
    assert "REPLAY_LF" in journey_text and "REPLAY_SHORT" in journey_text  # both embedded

    rows = _library_rows(site)
    assert "REPLAY_LF" in rows, "a published long-form embed with no library row is an orphan"
    assert rows["REPLAY_LF"]["title"] == "Replayed Long Form"
    assert rows["REPLAY_LF"]["date"] == "2025-09-03"
    assert rows["REPLAY_LF"]["duration_min"] == 13  # round(754/60)
    assert "REPLAY_SHORT" not in rows, "the library mirrors long-forms only"
    assert rows["EXISTING_VID_2"] == _HUMAN_ROW  # human-authored row untouched


def test_replay_dry_run_writes_neither_the_journey_nor_the_library(tmp_path):
    site = _make_site(tmp_path)
    before_journey = (site / "content" / "journeys" / "sample-trip.md").read_text()
    before_library = (site / "data" / "videos.yaml").read_text()

    rc = run_replay_pending(pending_files=[_seed_pending(tmp_path, site)], apply=False)

    assert rc == 0
    assert (site / "content" / "journeys" / "sample-trip.md").read_text() == before_journey
    assert (site / "data" / "videos.yaml").read_text() == before_library


def test_apply_merge_without_a_library_is_a_type_error():
    """The orphan is unconstructable, not merely gated: a writer that has no
    library to maintain cannot embed at all."""
    doc = {"video_count": 0, "days": [{"day_number": 1, "date": "2025-09-03", "videos": []}]}
    plan = MergePlan(
        inserts=[VideoEntry(1, "NO_LIB", "No Library", "12:34", "2025-09-03")],
        skips=[],
        errors=[],
    )
    with pytest.raises(TypeError):
        apply_merge(doc, plan)
