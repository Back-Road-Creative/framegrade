"""Route Story film job log -> progress summary parsing."""

from gomoveshift.console.route_story_progress import parse_progress


def test_parses_map_leg_progress_line():
    lines = [
        "2026-09-24 12:00:00 gomoveshift.pipeline.route_story.assembly map leg 3/10 "
        "frames, 0.50 s/frame, ~45 s left in this leg (film frame 30)",
    ]

    result = parse_progress(lines)

    assert result["leg"] == 3
    assert result["legs"] == 10
    assert result["seconds_left"] == 45
    assert result["faults"] == []
    assert result["finished"] is False


def test_parses_qa_fault_and_evidence_lines():
    lines = [
        "2026-09-24 12:01:00 gomoveshift.pipeline.route_story.film route-story frame "
        "QA refused a frame; evidence: /out/film.qa-evidence/frame-0042.png",
        "2026-09-24 12:01:01 gomoveshift.pipeline.route_story.film route-story frame "
        "QA evidence: /out/film.qa-evidence/frame-0042.png",
        "2026-09-24 12:01:02 gomoveshift.pipeline.route_story.film route-story frame "
        "QA could not write /out/film.qa-faults.json: [Errno 28] disk full",
    ]

    result = parse_progress(lines)

    assert len(result["faults"]) == 2
    assert result["evidence"] == ["/out/film.qa-evidence/frame-0042.png"]
    assert result["finished"] is False


def test_finished_marker_is_the_bare_output_path():
    lines = [
        "2026-09-24 12:05:00 gomoveshift.pipeline.route_story.assembly map leg 10/10 "
        "frames, 0.50 s/frame, ~0 s left in this leg (film frame 300)",
        "/out/film.mp4",
    ]

    result = parse_progress(lines)

    assert result["finished"] is True
    assert result["output"] == "/out/film.mp4"


def test_no_matching_lines_returns_defaults():
    result = parse_progress(["plain unrelated log text"])

    assert result == {
        "leg": None,
        "legs": None,
        "seconds_left": None,
        "faults": [],
        "evidence": [],
        "finished": False,
        "output": None,
    }
