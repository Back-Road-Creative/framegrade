"""``gomoveshift queue sync-bracket-short-text`` rewrites a queued bracket-route Short.

2026-11-16, nctopa-2026-03-05-rec0063: a plan written before the bracket label fix held 21
queued Shorts titled "Mar 5, 1st stretch: Pennsylvania ...", with no route record and no
``bracket_route``, each refused by the Short review (``description_no_clip_place``). The
sync rebuilds their text with the generator and the parent LF sidecar's bracket record,
and leaves anything it cannot back untouched. All data is synthetic, in the real shapes."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.publishing.plan_bracket_sync import sync_bracket_short_text
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.youtube_adapter import short_review_refusal

LABEL = "Virginia to Pennsylvania"
PROJECT = "nctopa-2026-03-05-rec0063"
RECORD = {
    "label": LABEL,
    "level": "states",
    "gap_h": 17.535,
    "before": {"lat": 37.3, "lon": -80.0, "ts": 1772666885.6, "source": "a.MP4"},
    "after": {"lat": 39.7, "lon": -77.4, "ts": 1772730012.1, "source": "b.MP4"},
}
ORDINALS = ("1st", "2nd", "3rd", "4th")


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _short(n, lf=1, **over):
    short = {
        "additional_media": [],
        "consent_hash": "",
        "experiment_arms": {"fb_hashtags": "A"},
        "format": "short",
        "lf_index": lf,
        "media": {
            "codec": "hevc",
            "duration_seconds": 37.97,
            "height": 1920,
            "path": f"/nonexistent/short_{n:02d}.mp4",
            "sha256": f"{n:064x}",
            "size_bytes": 1000,
            "width": 1080,
        },
        "parent_lf_publish_at": "2026-11-20T20:00:00+00:00",
        "parent_lf_video_id": "t32KI85p5yU",
        "payloads": {
            "facebook": {"description": "Another mile through Pennsylvania. Filmed March 2026."},
            "instagram": {
                "caption": "Driving the stretch through Pennsylvania. Filmed March 2026."
            },
            "tiktok": {
                "caption": "Taking you through Pennsylvania. Filmed March 2026.",
                "disclose_branded_content": False,
                "disclose_your_brand": False,
                "privacy_level": "SELF_ONLY",
                "title": f"Mar 5, {ORDINALS[n - 1]} stretch: 38 seconds, Pennsylvania",
            },
            "youtube": {
                "category": "19",
                "description": "Front-seat view driving through Virginia to Pennsylvania Pt.1 "
                "#Shorts #POVDriving #Pennsylvania\n\nhttps://youtu.be/t32KI85p5yU",
                "tags": ["POV driving", "drive", "road trip", "Virginia to Pennsylvania"],
                "title": f"Mar 5, {ORDINALS[n - 1]} stretch: Pennsylvania #Pennsylvania #Shorts",
            },
        },
        "project_name": PROJECT,
        "recording_date": "2026-03-05",
        "reviewed_plan_hash": "350925c910b8d45e02976f1a2ae353163d77d75e4df2f421ac6fd3b425c59b9d",
        "route_record": None,
        "routes": {"selected": ["youtube", "instagram", "facebook", "tiktok"]},
        "sampled_regions": [LABEL],
        "sampled_roads": [],
        "schedule": {
            "display_timezone": "America/New_York",
            "scheduled_at_utc": "2026-11-20T21:00:00+00:00",
            "scheduled_at_utc_by_platform": {},
        },
        "short_index": n,
    }
    short.update(over)
    return short


def _write(tmp_path, shorts, record=RECORD, lfs=(1,)):
    for lf in lfs:
        side = {"location": LABEL, "drive_text_inputs": {"bracket_route": record}}
        (tmp_path / f"{PROJECT}_lf{lf:02d}_youtube.json").write_text(json.dumps(side))
    plan = tmp_path / f"{PROJECT}_publish_plan.json"
    plan.write_text(json.dumps({"shorts": shorts}, sort_keys=True, indent=2) + "\n")
    return plan


def _records(plan):
    return load_publish_plan(plan).shorts


def test_the_fixture_plan_is_refused_before_the_sync(tmp_path):
    plan = _write(tmp_path, [_short(1)])
    assert short_review_refusal(_records(plan)[0]) is not None


def test_a_bracket_short_is_rewritten_and_passes_the_review(tmp_path):
    plan = _write(tmp_path, [_short(1), _short(2), _short(3)])
    changes = sync_bracket_short_text(plan, apply=True)
    assert [c.status for c in changes] == ["changed"] * 3
    for c, rec in zip(changes, _records(plan)):
        assert rec.payloads.youtube.title == c.new_title
        assert LABEL in rec.payloads.youtube.title
        assert rec.bracket_route == RECORD
        assert short_review_refusal(rec) is None
        assert rec.payloads.youtube.title.startswith(
            f"Mar 5, {ORDINALS[rec.short_index - 1]} stretch:"
        )
        # a platform caption that named only the destination now names the label
        assert LABEL in rec.payloads.instagram.caption
    kept = _records(plan)[0]
    assert (
        kept.reviewed_plan_hash
        == "350925c910b8d45e02976f1a2ae353163d77d75e4df2f421ac6fd3b425c59b9d"
    )
    assert kept.payloads.tiktok.privacy_level == "SELF_ONLY"


def test_a_short_with_a_route_record_is_untouched(tmp_path):
    bins = {"bins": [{"county": "Franklin County", "state": "Pennsylvania"}]}
    plan = _write(tmp_path, [_short(1, route_record=bins), _short(2)])
    before = json.loads(plan.read_text())["shorts"][0]
    changes = sync_bracket_short_text(plan, apply=True)
    assert [c.status for c in changes] == ["unchanged", "changed"]
    assert json.loads(plan.read_text())["shorts"][0] == before


@pytest.mark.parametrize("record", [None, {"label": LABEL, "level": "states", "gap_h": 99.0}])
def test_an_invalid_or_missing_bracket_record_is_refused_untouched(tmp_path, record):
    plan = _write(tmp_path, [_short(1)], record=record)
    before = plan.read_text()
    changes = sync_bracket_short_text(plan, apply=True)
    assert changes[0].status == "refused"
    assert "bracket_route" in changes[0].reason
    assert plan.read_text() == before


def test_a_region_other_than_the_label_is_refused_untouched(tmp_path):
    plan = _write(tmp_path, [_short(1, sampled_regions=["Franklin County, Pennsylvania"])])
    before = plan.read_text()
    changes = sync_bracket_short_text(plan, apply=True)
    assert changes[0].status == "refused"
    assert "sampled_regions" in changes[0].reason
    assert plan.read_text() == before


def test_a_title_without_the_plans_stamp_is_refused(tmp_path):
    odd = _short(1)
    odd["payloads"]["youtube"]["title"] = "Pennsylvania by road"
    plan = _write(tmp_path, [odd])
    assert sync_bracket_short_text(plan, apply=True)[0].status == "refused"


def test_a_missing_lf_sidecar_is_refused(tmp_path):
    plan = _write(tmp_path, [_short(1, lf=2)], lfs=(1,))
    assert sync_bracket_short_text(plan, apply=True)[0].status == "refused"


def test_dry_run_writes_nothing_and_leaves_no_temp_file(tmp_path):
    plan = _write(tmp_path, [_short(1), _short(2)])
    before = plan.read_text()
    changes = sync_bracket_short_text(plan, apply=False)
    assert [c.status for c in changes] == ["changed", "changed"]
    assert not any(c.written for c in changes)
    assert plan.read_text() == before
    assert not list(tmp_path.glob("*.sync-tmp"))


def test_apply_is_idempotent_and_the_plan_still_loads(tmp_path):
    plan = _write(tmp_path, [_short(1), _short(2)])
    sync_bracket_short_text(plan, apply=True)
    assert len(_records(plan)) == 2
    again = sync_bracket_short_text(plan, apply=True)
    assert [c.status for c in again] == ["unchanged", "unchanged"]


def test_the_cli_action_is_dry_run_by_default(tmp_path, capsys):
    plan = _write(tmp_path, [_short(1)])
    before = plan.read_text()
    config = SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path)))
    args = SimpleNamespace(queue_action="sync-bracket-short-text", sidecar=str(plan), apply=False)
    assert cmd_queue(args, config) == 0
    out = capsys.readouterr().out
    assert "dry run: would change" in out
    assert plan.read_text() == before
    args.apply = True
    assert cmd_queue(args, config) == 0
    assert plan.read_text() != before
