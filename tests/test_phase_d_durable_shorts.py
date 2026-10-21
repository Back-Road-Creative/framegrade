"""Phase D copies rendered shorts into the run's OUTPUT dir before the
publish plan records their paths.

Why this exists: shorts render to ``<stages_base>/day_<date>/shorts/
short_NN.mp4`` (``phase_d`` disk layout) and ``stages_base`` defaults to
``~/.cache/gomoveshift`` (``paths.py``). A host cleanup job can delete any
per-drive stage cache idle more than 14 days.
A shorts ladder schedules out to ~66 days, so a publish plan pointing at the
cache loses its media long before the last delivery fires.
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.pipeline.drive_phases.phase_d import _persist_shorts_to_output


def _rendered(cache_dir: Path, lf: int, n: int, data=b"video"):
    """Shorts where the pipeline actually renders them: the stage cache."""
    day = cache_dir / f"day_2026-03-1{lf}" / "shorts"
    day.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(1, n + 1):
        p = day / f"short_{i:02d}.mp4"
        p.write_bytes(data)
        paths.append(str(p))
    return paths


class TestPersistShortsToOutput:
    def test_copies_every_short_into_the_output_dir(self, tmp_path):
        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = {1: _rendered(cache, 1, 2), 2: _rendered(cache, 2, 1)}

        result = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)

        for lf, paths in result.items():
            for path in paths:
                assert Path(path).exists()
                # The whole point: durable location, not the stage cache.
                assert out in Path(path).parents
                assert cache not in Path(path).parents
        assert sorted(result) == [1, 2]
        assert [len(result[1]), len(result[2])] == [2, 1]

    def test_returned_paths_are_what_the_plan_should_record(self, tmp_path):
        """The mapping keeps LF keys and per-LF ORDER, because the plan
        position-zips these against social posts -- a reordered list would
        attach the wrong caption to the wrong clip."""
        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = {1: _rendered(cache, 1, 3)}

        result = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)

        assert [Path(p).name for p in result[1]] == [
            "short_01.mp4",
            "short_02.mp4",
            "short_03.mp4",
        ]

    def test_content_is_copied_not_just_the_name(self, tmp_path):
        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = {1: _rendered(cache, 1, 1, data=b"the-real-bytes")}

        result = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)

        assert Path(result[1][0]).read_bytes() == b"the-real-bytes"

    def test_survives_the_stage_cache_being_wiped(self, tmp_path):
        """The actual failure this prevents: a host cleanup job removes the cache
        14 days in, and a delivery scheduled for day 40 still publishes."""
        import shutil

        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = {1: _rendered(cache, 1, 1)}

        result = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)
        shutil.rmtree(cache)

        assert Path(result[1][0]).exists()

    def test_is_idempotent_so_a_resumed_run_does_not_recopy(self, tmp_path):
        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = {1: _rendered(cache, 1, 1)}

        first = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)
        stamped = Path(first[1][0])
        # SAME byte length as the source, so the size check says "already
        # copied" and the copy is skipped. The marker surviving is the proof.
        stamped.write_bytes(b"VIDEO")
        second = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)

        assert second == first
        assert stamped.read_bytes() == b"VIDEO"

    def test_a_truncated_earlier_copy_is_replaced_not_trusted(self, tmp_path):
        """The other half of the resume check: a copy interrupted mid-write
        differs in size and must be redone, or the ladder publishes a corrupt
        clip weeks later."""
        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = {1: _rendered(cache, 1, 1, data=b"the-full-clip")}

        first = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)
        Path(first[1][0]).write_bytes(b"trunc")
        _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)

        assert Path(first[1][0]).read_bytes() == b"the-full-clip"

    def test_empty_and_none_inputs_are_not_errors(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        assert _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=None) == {}
        assert _persist_shorts_to_output(output_base=out, shorts_paths_by_lf={}) == {}

    def test_a_missing_rendered_short_is_skipped_not_fatal(self, tmp_path):
        """Phase D must not hard-fail here: the CSV path does not need these
        copies, and a shorts shortfall already has its own upstream gate."""
        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = _rendered(cache, 1, 2)
        Path(rendered[0]).unlink()

        result = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf={1: rendered})

        assert len(result[1]) == 1
        assert Path(result[1][0]).name == "short_02.mp4"


class TestPhaseDRecordsTheDurablePath:
    def test_the_sidecar_records_the_output_copy_not_the_stage_cache(self, tmp_path):
        """The wiring, not just the helper: whatever Phase D hands the plan
        builder must be a path that survives a host cache cleanup. A helper that
        copies correctly but is never called leaves the defect intact."""
        import json
        from types import SimpleNamespace
        from unittest.mock import patch

        from gomoveshift.config import DriveConfig
        from gomoveshift.pipeline.drive_phases.phase_d import _emit_publish_plan_sidecar
        from gomoveshift.pipeline.title_decision import decide_title
        from gomoveshift.publishing.models import MediaFingerprint

        cache, out = tmp_path / "cache", tmp_path / "out"
        out.mkdir()
        rendered = {1: _rendered(cache, 1, 1)}
        durable = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf=rendered)

        def _fp(path, **_kw):
            return MediaFingerprint(str(path), 5, 9.0, 1080, 1920, "h264", "a" * 64)

        with patch("gomoveshift.publishing.media_fingerprint.fingerprint_media", side_effect=_fp):
            plan_path = _emit_publish_plan_sidecar(
                output_base=out,
                project_base="p",
                drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
                social_posts_per_lf={
                    1: [
                        SimpleNamespace(
                            short_index=10,
                            title_decision=decide_title("short", "", None, suffix=""),
                        )
                    ]
                },
                drive_schedule=SimpleNamespace(
                    shorts_slots=[
                        SimpleNamespace(short_index=10, date="2026-03-15", time="08:00:00")
                    ]
                ),
                youtube_uploads={1: {"video_id": "v", "publish_at": "2026-03-14T08:00:00-05:00"}},
                per_lf_recording_date={1: "2026-03-14"},
                shorts_paths_by_lf=durable,
                publish_pass=True,
                config=SimpleNamespace(drive=DriveConfig(publishing_route_youtube="first_party")),
            )

        recorded = json.loads(plan_path.read_text())["shorts"][0]["media"]["path"]
        assert out in Path(recorded).parents, recorded
        assert cache not in Path(recorded).parents, recorded


class TestCopyFailures:
    def test_metadata_copy_refused_by_the_filesystem_still_keeps_the_copy(
        self, tmp_path, monkeypatch
    ):
        """drvfs writes the bytes but refuses the mode/utime update
        with EPERM. The copy is complete, so record it."""
        import shutil

        src = tmp_path / "cache" / "short_01.mp4"
        src.parent.mkdir()
        src.write_bytes(b"x" * 100)
        out = tmp_path / "out"

        def _refuse(*a, **k):
            raise PermissionError(1, "Operation not permitted")

        monkeypatch.setattr(shutil, "copystat", _refuse)
        got = _persist_shorts_to_output(output_base=out, shorts_paths_by_lf={1: [str(src)]})
        dest = out / "shorts" / "lf_01" / "short_01.mp4"
        assert got == {1: [str(dest)]}
        assert dest.read_bytes() == b"x" * 100

    def test_a_failed_copy_records_the_rendered_path_instead_of_a_hole(self, tmp_path, monkeypatch):
        import shutil

        src = tmp_path / "cache" / "short_01.mp4"
        src.parent.mkdir()
        src.write_bytes(b"x" * 100)

        def _fail(*a, **k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(shutil, "copyfile", _fail)
        got = _persist_shorts_to_output(
            output_base=tmp_path / "out", shorts_paths_by_lf={1: [str(src)]}
        )
        assert got == {1: [str(src)]}
