"""Long-form work files are released as each long-form finishes, and
``gomoveshift cleanup <project>`` reclaims a published drive that also has an
output directory.

Measured 2026-10-10 on a 133-minute, two-part drive: the day master lands near
77 GB and each long-form keeps three artifacts on the cache volume while it
burns (``lf_NN_segments.mp4``, ``lf_NN_with_montage.mp4``, the final), about
3.1x its share of the master (2026-10-08: a 16.78 GB share peaked at 51.78 GB).
The purge ran once, at day end, so part 2 needed room for both parts' work
files: about 233 GB against 163 GB free. Nothing checked the volume before a
part started; a failed montage prepend only logs "skipping".

The same week ``gomoveshift cleanup <drive> --confirm`` freed nothing and
exited 0 for a published drive: the named-project lookup took the output
directory (which has no ``_stages``) over the stage cache, so the drive read as
"pipeline not complete".
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from _state_helpers import make_state
from gomoveshift.cli.handlers import cleanup as cleanup_handler
from test_cleanup_drive_cache import _drive
from test_day_pipeline import _make_day_batch, _make_lf_plan, _make_scores, _make_seg

_GB = 1_073_741_824


# --- gomoveshift cleanup <project> ------------------------------------


def _run_cleanup(tmp_path, project_name, *, confirm=False, min_age=0):
    out, stg = tmp_path / "out", tmp_path / "stg"
    out.mkdir(exist_ok=True)
    stg.mkdir(exist_ok=True)
    args = SimpleNamespace(
        project=project_name, confirm=confirm, min_age=min_age, archive_base=None
    )
    with (
        patch("gomoveshift.paths.gms_output_dir", return_value=out),
        patch("gomoveshift.paths.gms_staging_dir", return_value=stg),
        patch("gomoveshift.paths.gms_intermediate_base", return_value=tmp_path / "cache"),
    ):
        return cleanup_handler.cmd_cleanup(args, None)


def test_named_drive_with_an_output_dir_reclaims_its_stage_cache(tmp_path, capsys):
    """A published drive writes its publish records under the output dir, so both
    candidates exist. The one that holds ``_stages`` is the project."""
    project = _drive(tmp_path / "cache")
    (tmp_path / "out" / project.name).mkdir(parents=True)
    (tmp_path / "out" / project.name / "youtube_uploads.json").write_text("{}")

    rc = _run_cleanup(tmp_path, project.name, confirm=True)

    assert rc == 0
    lf = project / "_stages" / "day_2026-03-05" / "regular_lfs" / "lf_01.mp4"
    assert not lf.exists(), capsys.readouterr().out
    assert not (project / "_stages" / "episode_01_2026-03-05.mp4").exists()
    assert (project / "_stages" / "pipeline_state.json").exists()


def test_named_project_that_is_skipped_exits_nonzero_and_says_why(tmp_path, capsys):
    """A caller chaining ``cleanup <project> --confirm && echo reclaimed``
    must not see success when the project was skipped."""
    project = _drive(tmp_path / "cache", phases=("phase_a", "phase_bc_day_first"))

    rc = _run_cleanup(tmp_path, project.name, confirm=True)

    captured = capsys.readouterr()
    assert rc == 2
    assert "Pipeline not complete" in captured.out + captured.err
    assert (project / "_stages" / "episode_01_2026-03-05.mp4").exists()


def test_named_project_with_nothing_left_still_exits_zero(tmp_path):
    """Idempotent: a second run over an already-reclaimed drive is not an error."""
    project = _drive(tmp_path / "cache")
    assert _run_cleanup(tmp_path, project.name, confirm=True) == 0
    assert _run_cleanup(tmp_path, project.name, confirm=True) == 0


# --- purge_lf_intermediates -------------------------------------------


def _lf_work(day_dir: Path, idx: int, *, final: bytes | None = b"final") -> dict:
    lf_dir = day_dir / "regular_lfs"
    work = lf_dir / f".lf_{idx:02d}_work"
    work.mkdir(parents=True)
    paths = {
        "segments": lf_dir / f"lf_{idx:02d}_segments.mp4",
        "montage": lf_dir / f"lf_{idx:02d}_with_montage.mp4",
        "work": work,
        "final": lf_dir / f"lf_{idx:02d}.mp4",
        "fingerprint": lf_dir / f"lf_{idx:02d}.ost.sha256",
    }
    paths["segments"].write_bytes(b"s" * 100)
    paths["montage"].write_bytes(b"m" * 100)
    (work / "part.ts").write_bytes(b"w" * 50)
    paths["fingerprint"].write_text("abc")
    if final is not None:
        paths["final"].write_bytes(final)
    return paths


def test_purge_lf_intermediates_removes_one_lfs_work_files_only(tmp_path):
    from gomoveshift.pipeline.cleanup import purge_lf_intermediates

    one, two = _lf_work(tmp_path, 1), _lf_work(tmp_path, 2)

    summary = purge_lf_intermediates(tmp_path, 1)

    assert summary == {"files_deleted": 2, "dirs_deleted": 1, "bytes_freed": 250}
    assert not one["segments"].exists() and not one["montage"].exists()
    assert not one["work"].exists()
    assert one["final"].exists() and one["fingerprint"].exists()
    assert two["segments"].exists() and two["montage"].exists() and two["work"].exists()


@pytest.mark.parametrize("final", [None, b""])
def test_purge_lf_intermediates_keeps_everything_without_a_final(tmp_path, final):
    """The work files are the only way to re-burn an LF whose final is absent
    or empty."""
    from gomoveshift.pipeline.cleanup import purge_lf_intermediates

    one = _lf_work(tmp_path, 1, final=final)

    assert purge_lf_intermediates(tmp_path, 1)["files_deleted"] == 0
    assert one["segments"].exists() and one["montage"].exists() and one["work"].exists()


# --- check_lf_work_space ----------------------------------------------


class TestLfWorkSpace:
    def _check(
        self, tmp_path, *, free_gb, lf_sec=3600.0, master_gb=60, master_sec=7200.0, later=()
    ):
        from gomoveshift.pipeline.preflight import check_lf_work_space

        master = tmp_path / "day_master.mp4"
        master.write_bytes(b"x")
        with (
            patch.object(Path, "stat", return_value=SimpleNamespace(st_size=master_gb * _GB)),
            patch("shutil.disk_usage", return_value=SimpleNamespace(free=free_gb * _GB)),
        ):
            check_lf_work_space(
                master, tmp_path, lf_sec, master_sec, lf_index=2, later_lf_secs=later
            )

    def test_a_later_lf_is_priced_on_top_of_this_lfs_final(self, tmp_path):
        """Each final stays on the cache volume, so the next LF burns beside it:
        30 GB final + 30 GB x3.1 = 123 GB, x1.1 margin = ~135 GB. 110 GB holds
        this LF alone but would stop the run at the next LF's gate hours later
        (Ace Basin 2026-10-27: LF2 needed 145.8 GiB, ~125 GiB left)."""
        with pytest.raises(RuntimeError, match=r"(?i)work space for LF 2.*1 later long-form"):
            self._check(tmp_path, free_gb=110, later=[3600.0])

    def test_room_for_every_remaining_lf_passes(self, tmp_path):
        self._check(tmp_path, free_gb=140, later=[3600.0])

    def test_a_short_later_lf_does_not_raise_the_need(self, tmp_path):
        """The peak is the worst single LF plus the finals before it, not a sum
        of every LF's work: a 6-minute later LF (3 GB) never exceeds this one."""
        self._check(tmp_path, free_gb=110, later=[360.0])

    def test_three_coexisting_artifacts_are_priced(self, tmp_path):
        """A 30 GB share needs about 102 GB (x3.1, x1.1 margin). 70 GB holds
        the stream copy and the montage copy but not the final."""
        with pytest.raises(RuntimeError, match=r"(?i)work space for LF 2.*70\.0 GB free"):
            self._check(tmp_path, free_gb=70)

    def test_enough_room_passes(self, tmp_path):
        self._check(tmp_path, free_gb=110)

    @pytest.mark.parametrize("master_sec", [0.0, -1.0])
    def test_unmeasurable_master_skips_rather_than_raises(self, tmp_path, master_sec):
        self._check(tmp_path, free_gb=1, master_sec=master_sec)

    def test_a_part_with_no_length_prices_at_zero(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            self._check(tmp_path, free_gb=1, lf_sec=0.0)
        assert not caplog.records

    def test_a_master_that_cannot_be_read_skips(self, tmp_path, caplog):
        """No master on disk: the concat fails on its own a moment later and
        writes nothing, so the gate has nothing to protect and must not be
        the thing that crashes."""
        from gomoveshift.pipeline.preflight import check_lf_work_space

        with caplog.at_level("WARNING"):
            check_lf_work_space(tmp_path / "absent.mp4", tmp_path, 3600.0, 7200.0, lf_index=1)
        assert "skipping the space estimate" in caplog.text


# --- run_day_pipeline wiring ------------------------------------------


def _two_lf_day(tmp_path, *, state, on_assemble=None, space=None):
    """Run a two-LF day with every heavy stage faked; returns (artifacts,
    lf_dir, trace)."""
    from gomoveshift.pipeline import day_pipeline

    out = tmp_path / "out"
    master = out / "day_2026-02-20" / "day_2026-02-20_master.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"m")
    lf_dir = master.parent / "regular_lfs"
    trace: list = []

    def _asm(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        idx = int(output_path.name[3:5])
        trace.append(("assemble", idx))
        if on_assemble is not None:
            on_assemble(idx, lf_dir)
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        (Path(work_dir) / "part.ts").write_bytes(b"w")
        output_path.write_bytes(b"x")
        return output_path

    def _write(output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"x")
        return output_path

    def _space(master_path, output_dir, lf_sec, master_sec, **kw):
        trace.append(("space", kw.get("lf_index"), lf_sec, master_sec))
        trace.append(("later", kw.get("lf_index"), tuple(kw.get("later_lf_secs", ()))))
        if space is not None:
            space(kw.get("lf_index"))

    with (
        patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master),
        patch("gomoveshift.pipeline.day_pipeline.extract_and_concat_segments", side_effect=_asm),
        patch(
            "gomoveshift.pipeline.day_pipeline.prepend_montage",
            side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: _write(output_path),
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline.apply_ost",
            side_effect=lambda input_path, output_path, location, layout, **kw: _write(output_path),
        ),
        patch(
            "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
            side_effect=lambda source_video, output_path, *a, **kw: bool(_write(output_path)),
        ),
        patch("gomoveshift.pipeline.day_pipeline.check_lf_work_space", side_effect=_space),
    ):
        artifacts = day_pipeline.run_day_pipeline(
            day_batch=_make_day_batch("2026-02-20", 2, per_ep_duration=1200.0),
            day_scores=_make_scores(2400),
            output_dir=out,
            location="L",
            config=None,
            lf_plans=[
                _make_lf_plan(1, [_make_seg(0, 100, 200), _make_seg(1, 50, 150)], []),
                _make_lf_plan(2, [_make_seg(1, 200, 500)], []),
            ],
            state=state,
        )
    return artifacts, lf_dir, trace


def _lf1_work_present(lf_dir: Path) -> list[str]:
    names = ("lf_01_segments.mp4", "lf_01_with_montage.mp4", ".lf_01_work")
    return [n for n in names if (lf_dir / n).exists()]


def test_lf_work_files_are_gone_before_the_next_lf_starts(tmp_path):
    seen: dict = {}

    def _on_assemble(idx, lf_dir):
        if idx == 2:
            seen["lf1_work"] = _lf1_work_present(lf_dir)
            seen["lf1_final"] = (lf_dir / "lf_01.mp4").exists()

    artifacts, _, _ = _two_lf_day(
        tmp_path, state=make_state(ost_audit_confirmed=True), on_assemble=_on_assemble
    )

    assert artifacts is not None and len(artifacts.regular_lf_paths) == 2
    assert seen == {"lf1_work": [], "lf1_final": True}


def test_keep_intermediates_retains_each_lfs_work_files(tmp_path):
    seen: dict = {}

    def _on_assemble(idx, lf_dir):
        if idx == 2:
            seen["lf1_work"] = _lf1_work_present(lf_dir)

    _two_lf_day(
        tmp_path,
        state=make_state(ost_audit_confirmed=True, keep_intermediates=True),
        on_assemble=_on_assemble,
    )

    assert len(seen["lf1_work"]) == 3


def test_each_lf_is_space_checked_before_its_concat(tmp_path):
    _, _, trace = _two_lf_day(tmp_path, state=make_state(ost_audit_confirmed=True))

    lf_events = [e for e in trace if e[0] in ("space", "assemble")]
    assert [e[:2] for e in lf_events] == [
        ("space", 1),
        ("assemble", 1),
        ("space", 2),
        ("assemble", 2),
    ]
    # LF duration is the sum of its kept segments; the master is the whole day.
    assert trace[0][2:] == (200.0, 2400.0)
    assert [e for e in trace if e[:2] == ("space", 2)][0][2:] == (300.0, 2400.0)


def test_the_first_lf_gate_prices_every_lf_still_to_burn(tmp_path):
    """LF 1's gate sees LF 2's length, so a day whose second LF cannot fit
    beside the first's final stops before any concat, not hours later."""
    _, _, trace = _two_lf_day(tmp_path, state=make_state(ost_audit_confirmed=True))

    later = {e[1]: e[2] for e in trace if e[0] == "later"}
    assert later == {1: (300.0,), 2: ()}


def test_a_later_lf_whose_final_exists_is_not_priced_again(tmp_path):
    """On a resume LF 2's final is already on disk (and already out of the
    free figure), so LF 1's re-burn must not reserve room for it twice."""
    lf_dir = tmp_path / "out" / "day_2026-02-20" / "regular_lfs"
    lf_dir.mkdir(parents=True)
    (lf_dir / "lf_02.mp4").write_bytes(b"done")
    _, _, trace = _two_lf_day(tmp_path, state=make_state(ost_audit_confirmed=True))

    later = {e[1]: e[2] for e in trace if e[0] == "later"}
    assert later[1] == ()


def test_a_space_shortfall_stops_the_day_before_the_concat(tmp_path):
    """The shortfall must surface as the failure, not as a half-written
    concat or a silently skipped long-form."""

    def _space(lf_index):
        if lf_index == 2:
            raise RuntimeError("Insufficient work space for LF 2")

    with pytest.raises(RuntimeError, match="work space for LF 2"):
        _two_lf_day(tmp_path, state=make_state(ost_audit_confirmed=True), space=_space)

    lf_dir = tmp_path / "out" / "day_2026-02-20" / "regular_lfs"
    assert (lf_dir / "lf_01.mp4").exists()
    assert not (lf_dir / "lf_02_segments.mp4").exists()
