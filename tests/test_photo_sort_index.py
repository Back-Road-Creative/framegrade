"""photos sort: index.html sequence cards, generic outputs, top picks, summary."""

from datetime import datetime, timedelta
from pathlib import Path

from gomoveshift.pipeline.photos import sort as ps
from gomoveshift.pipeline.photos.sort_groups import Output

T0 = datetime(2026, 11, 8, 22, 0, 0)


def row(rank, category="other", gid="", at=None, score=0.5):
    return {
        "category": category,
        "rank": str(rank),
        "output": f"/run/{category}/{rank:03d}-f.jpg",
        "quality_score": str(score),
        "capture_time": at.strftime("%Y:%m:%d %H:%M:%S") if at else "",
        "group_id": gid,
        "group_best_rank": "",
    }


def frames(gid, start_rank, n, start, category="night-sky"):
    return [row(start_rank + i, category, gid, start + timedelta(minutes=i)) for i in range(n)]


def page(tmp_path, ranked, outputs=()):
    return ps._index_html(tmp_path, Path("/card"), ranked, list(outputs))


def outside_details(html):
    """The page with every <details>...</details> block removed."""
    out, rest = "", html
    while "<details" in rest:
        head, _, tail = rest.partition("<details")
        out += head
        rest = tail.split("</details>", 1)[1]
    return out + rest


def test_contiguous_night_groups_make_one_card(tmp_path):
    ranked = frames("night-1", 1, 6, T0) + frames("night-2", 7, 6, T0 + timedelta(minutes=10))
    outs = [
        Output("stacks", "night-1", "stacks/night-1.jpg"),
        Output("stacks", "night-2", "stacks/night-2.jpg"),
    ]
    html = page(tmp_path, ranked, outs)
    assert html.count('class="card"') == 1
    assert "stacks/night-1.jpg" in html and "stacks/night-2.jpg" in html
    assert "12 frames" in html and "show 12 frames" in html
    assert "2026:11:08 22:00:00" in html and "2026:11:08 22:15:00" in html
    assert "night-sky/001-f.jpg" in html
    assert "night-sky/001-f.jpg" not in outside_details(html)
    assert html.count('alt="001-f.jpg"') == 1
    assert html.index("stacks/night-1.jpg") < html.index("<details")


def test_night_groups_far_apart_stay_separate_cards(tmp_path):
    ranked = frames("night-1", 1, 6, T0) + frames("night-2", 7, 6, T0 + timedelta(hours=2))
    assert page(tmp_path, ranked).count('class="card"') == 2


def test_small_burst_stays_in_grid(tmp_path):
    ranked = frames("burst-1", 1, 2, T0, "other")
    html = page(tmp_path, ranked)
    assert 'class="card"' not in html
    assert "other/001-f.jpg" in outside_details(html)
    assert "same scene as" not in html
    assert "burst of 2" in html


def test_bracket_with_hdr_output_is_one_card_no_second_heading(tmp_path):
    ranked = frames("bracket-1", 1, 3, T0, "hdr")
    html = page(tmp_path, ranked, [Output("hdr", "bracket-1", "hdr/bracket-1-f.jpg")])
    assert html.count('class="card"') == 1
    assert html.count("<h2>hdr") == 1 and html.count("<section") == 1
    assert "hdr/bracket-1-f.jpg" in outside_details(html)
    assert "hdr/001-f.jpg" not in outside_details(html) and "hdr/001-f.jpg" in html
    assert html.index("hdr/bracket-1-f.jpg") < html.index("<details")
    assert "same scene as" not in html


def test_pano_with_output_is_one_card_in_its_frames_category(tmp_path):
    ranked = frames("pano-1", 1, 3, T0, "wildlife")
    html = page(tmp_path, ranked, [Output("pano", "pano-1", "pano/pano-1-f.jpg")])
    assert html.count('class="card"') == 1
    assert "<h2>pano" not in html and html.count("<h2>wildlife (3)</h2>") == 1
    assert html.index("<h2>wildlife") < html.index("pano/pano-1-f.jpg") < html.index("<details")


def test_small_group_without_output_shows_best_frame_as_burst(tmp_path):
    ranked = frames("bracket-1", 1, 3, T0, "hdr")
    html = page(tmp_path, ranked)
    shown = outside_details(html)
    assert "same scene as" not in html and "burst of 3" in shown
    assert "hdr/001-f.jpg" in shown
    assert "hdr/002-f.jpg" not in shown and "hdr/003-f.jpg" not in shown
    assert "hdr/002-f.jpg" in html and 'class="card"' not in html


def test_big_burst_of_any_kind_is_a_card_in_its_category(tmp_path):
    ranked = frames("burst-3", 1, 7, T0, "other")
    html = page(tmp_path, ranked, [Output("timelapse", "burst-3", "timelapse/burst-3.mp4")])
    assert html.count('class="card"') == 1
    assert "<video" in html and "timelapse/burst-3.mp4" in html
    assert "show 7 frames" in html


def test_mp4_output_renders_video_with_poster_only_if_on_disk(tmp_path):
    (tmp_path / "timelapse").mkdir()
    (tmp_path / "timelapse" / "a.jpg").write_bytes(b"x")
    outs = [
        Output("timelapse", "a", "timelapse/a.mp4"),
        Output("timelapse", "b", "timelapse/b.mp4"),
    ]
    ranked = frames("a", 1, 6, T0, "other") + frames("b", 7, 6, T0 + timedelta(days=1), "other")
    html = page(tmp_path, ranked, outs)
    assert '<video controls preload="none"' in html
    assert 'poster="timelapse/a.jpg"' in html
    assert "timelapse/b.jpg" not in html and html.count("poster=") == 1


def test_section_shows_top_six_singles_rest_in_details(tmp_path):
    ranked = [row(i, "other") for i in range(1, 11)]
    html = page(tmp_path, ranked)
    shown = outside_details(html)
    assert all(f"other/{i:03d}-f.jpg" in shown for i in range(1, 7))
    assert all(f"other/{i:03d}-f.jpg" not in shown for i in range(7, 11))
    assert "show all 4" in html and "other/010-f.jpg" in html


def test_summary_strip_counts_shots_and_sequences(tmp_path):
    ranked = frames("night-1", 1, 6, T0) + [row(7, "other"), row(8, "other")]
    html = page(tmp_path, ranked)
    strip = html.split('class="summary"')[1].split("</p>")[0]
    assert "night-sky 6" in strip and "other 2" in strip and "1 sequence" in strip


def test_page_is_dark_grid_and_self_contained(tmp_path):
    html = page(tmp_path, [row(1)])
    assert "<style>" in html and "background:#" in html
    assert "grid-template-columns:repeat(auto-fill,minmax(240px,1fr))" in html
    assert "<script" not in html and "http" not in html and 'loading="lazy"' in html
