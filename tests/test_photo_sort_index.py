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
    assert "Night sky · Nov 8 · 10:00–10:15 PM · 12 frames" in html
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
    assert html.count("<h3>HDR (3 shots)</h3>") == 1 and html.count("<section") == 1
    assert "hdr/bracket-1-f.jpg" in outside_details(html)
    assert "hdr/001-f.jpg" not in outside_details(html) and "hdr/001-f.jpg" in html
    assert html.index("hdr/bracket-1-f.jpg") < html.index("<details")
    assert "same scene as" not in html


def test_pano_with_output_is_one_card_in_its_frames_category(tmp_path):
    ranked = frames("pano-1", 1, 3, T0, "wildlife")
    html = page(tmp_path, ranked, [Output("pano", "pano-1", "pano/pano-1-f.jpg")])
    assert html.count('class="card"') == 1
    assert "<h3>Panorama" not in html and html.count("<h3>Wildlife (3 shots)</h3>") == 1
    assert html.index("<h3>Wildlife") < html.index("pano/pano-1-f.jpg") < html.index("<details")


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
    assert "show 4 more" in html and "other/010-f.jpg" in html


def test_summary_strip_counts_shots_and_sequences(tmp_path):
    ranked = frames("night-1", 1, 6, T0) + [row(7, "other"), row(8, "other")]
    html = page(tmp_path, ranked)
    strip = html.split('class="summary"')[1].split("</p>")[0]
    assert "Night sky 6" in strip and "Unsorted 2" in strip


def test_page_is_dark_grid_and_self_contained(tmp_path):
    html = page(tmp_path, [row(1)])
    assert "<style>" in html and "background:#" in html
    assert "grid-template-columns:repeat(auto-fill,minmax(240px,1fr))" in html
    assert "<script" not in html and "http" not in html and 'loading="lazy"' in html


# --- day layout, readable dates and labels (U3) ---------------------------------------

import re  # noqa: E402

D1 = datetime(2026, 9, 25, 20, 42, 0)
D2 = datetime(2026, 9, 19, 13, 17, 0)
D3 = datetime(2026, 11, 6, 9, 5, 0)


def headings(html, tags="h2h3"):
    found = []
    for tag in ("h2", "h3", "h4"):
        if tag in tags or tag == "h4" and "h4" in tags:
            found += re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", html)
    return found


def multi_day():
    ranked = [
        row(1, "landscape", "", D1.replace(hour=13, minute=17), 0.27),
        row(2, "wildlife", "", D1.replace(hour=9), 0.6),
        row(3, "other", "", D1.replace(hour=10), 0.5),
        row(4, "landscape", "", D2, 0.4),
        row(5, "hdr", "", D3, 0.4),
        row(6, "close-up", "", D1.replace(hour=11), 0.3),
        row(7, "night-sky", "", D1.replace(hour=22), 0.3),
        row(8, "zebra", "", D1.replace(hour=12), 0.3),
        row(9, "landscape", "", None, 0.3),
        row(10, "lens-cap-darks", "", D1.replace(hour=20), 0.0),
        row(11, "lens-cap-darks", "", D1.replace(hour=20, minute=1), 0.0),
    ]
    return ranked


def test_days_are_sections_in_date_order_with_a_day_strip(tmp_path):
    html = page(tmp_path, multi_day())
    h2s = headings(html, "h2")
    assert h2s == ["Sat Sep 19 2026", "Fri Sep 25 2026", "Fri Nov 6 2026", "Undated"]
    assert 'href="#day-2026-09-19"' in html and 'id="day-2026-09-25"' in html
    assert html.index('href="#day-2026-09-25"') < html.index('id="day-2026-09-19"')


def test_categories_inside_a_day_follow_the_fixed_order(tmp_path):
    html = page(tmp_path, multi_day())
    day = html.split('id="day-2026-09-25"')[1].split("</section>")[0]
    names = [re.sub(r" \(.*", "", h) for h in headings(day, "h3")]
    assert names == ["Night sky", "Landscape", "Wildlife", "Close-up", "Unsorted", "Zebra"]


def test_no_raw_exif_dates_group_ids_or_best_rank_anywhere(tmp_path):
    ranked = frames("night-1", 1, 6, T0) + frames("bracket-1", 7, 3, T0, "hdr")
    ranked += frames("burst-1", 10, 2, T0, "other")
    outs = [
        Output("stacks", "night-1", "stacks/night-1.jpg"),
        Output("hdr", "bracket-1", "hdr/bracket-1-f.jpg"),
    ]
    html = page(tmp_path, ranked, outs)
    assert not re.search(r"\d{4}:\d{2}:\d{2}", html)
    assert "best #" not in html
    for text in headings(html, "h2h3h4"):
        assert not re.search(r"(night|bracket|burst|pano|sequence)-\d", text)


def test_card_heading_and_output_labels_are_readable(tmp_path):
    ranked = frames("night-1", 1, 6, T0) + frames("night-2", 7, 6, T0 + timedelta(minutes=10))
    outs = [
        Output("stacks", "night-2", "stacks/night-2.jpg"),
        Output("stacks", "night-1", "stacks/night-1.jpg"),
        Output("trails", "night-1", "trails/night-1.jpg"),
    ]
    html = page(tmp_path, ranked, outs)
    assert "Night sky · Nov 8 · 10:00–10:15 PM · 12 frames" in html
    assert "Stack · 10:00–10:05 PM · 6 frames" in html
    assert "Stack · 10:10–10:15 PM · 6 frames" in html
    assert html.index("Stack · 10:00") < html.index("Stack · 10:10")
    assert html.index("stacks/night-1.jpg") < html.index("stacks/night-2.jpg")
    assert "Star trail" in html


def test_stacks_are_ordered_by_start_time_not_group_number(tmp_path):
    late = frames("night-1", 1, 6, T0 + timedelta(minutes=10))
    early = frames("night-2", 7, 6, T0)
    outs = [
        Output("stacks", "night-1", "stacks/night-1.jpg"),
        Output("stacks", "night-2", "stacks/night-2.jpg"),
    ]
    html = page(tmp_path, late + early, outs)
    assert html.index("stacks/night-2.jpg") < html.index("stacks/night-1.jpg")


def test_unknown_or_session_output_ids_never_crash(tmp_path):
    ranked = frames("night-1", 1, 6, T0)
    outs = [
        Output("stacks", "night-1-session", "stacks/night-1-session.jpg"),
        Output("stacks", "mystery", "stacks/mystery.jpg"),
    ]
    html = page(tmp_path, ranked, outs)
    assert "stacks/night-1-session.jpg" in html and "stacks/mystery.jpg" in html


def test_single_shot_caption_has_day_time_and_labelled_quality(tmp_path):
    html = page(tmp_path, [row(1, "landscape", "", D1.replace(hour=13, minute=17), 0.27)])
    assert "Sep 25, 1:17 PM · quality 27" in html
    assert "<small>0.27</small>" not in html


def test_calibration_frames_are_one_line_per_day_never_thumbnails(tmp_path):
    html = page(tmp_path, multi_day())
    assert html.count("2 calibration frames (lens cap on), used for the stacks, not shown") == 1
    assert "lens-cap-darks" not in html


def test_fold_only_when_more_than_two_would_be_hidden(tmp_path):
    for total, folded in ((7, 0), (8, 0), (9, 3)):
        html = page(tmp_path, [row(i, "other", "", D1) for i in range(1, total + 1)])
        assert "show all" not in html
        assert (f"show {folded} more" in html) == (folded > 0)
        assert all(f"other/{i:03d}-f.jpg" in html for i in range(1, total + 1))
    html = page(tmp_path, [row(1, "other", "", D1)])
    assert "<details" not in html


def test_title_names_folder_count_without_darks_and_date_range(tmp_path):
    html = page(tmp_path, multi_day())
    title = "card — 9 photos, Sep 19 – Nov 6 2026"
    assert f"<title>{title}</title>" in html and f"<h1>{title}</h1>" in html
    assert "<title>card — 1 photos" not in page(tmp_path, [row(1)])
    undated = page(tmp_path, [row(1), row(2)])
    assert "<h1>card — 2 photos</h1>" in undated


def test_display_names_and_timelapse_card_heading(tmp_path):
    ranked = [row(1, "long-exposure", "", D1), row(2, "close-up", "", D1)]
    ranked += frames("burst-3", 3, 7, T0, "other")
    html = page(tmp_path, ranked, [Output("timelapse", "burst-3", "timelapse/burst-3.mp4")])
    assert "Long exposure" in html and "Close-up" in html
    day = html.split('id="day-2026-11-08"')[1]
    assert "Time-lapse · Nov 8 · 10:00–10:06 PM · 7 frames" in day
    assert "Time-lapse" in headings(day, "h3")[0]


def test_summary_counts_categories_and_built_outputs(tmp_path):
    ranked = frames("night-1", 1, 6, T0) + [row(7, "other", "", D1), row(8, "lens-cap-darks")]
    outs = [Output("stacks", "night-1", "stacks/night-1.jpg")]
    html = page(tmp_path, ranked, outs)
    strip = html.split('class="summary"')[1].split("</p>")[0]
    assert "Night sky 6" in strip and "Unsorted 1" in strip and "lens" not in strip
    assert "1 night session" in strip and "HDR" not in strip
