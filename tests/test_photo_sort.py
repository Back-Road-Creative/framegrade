"""photos sort: dedupe, classify and develop a folder into a fixed output home."""

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.photos import sort as ps
from gomoveshift.pipeline.photos.capture_meta import CaptureRecord
from gomoveshift.pipeline.photos.subject import (
    CaptureOptics,
    SubjectCategory,
    SubjectReading,
    recognise_photos,
)

TODAY = "2026-11-09"


def jpeg(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (16, 24, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


class Spy:
    """Fake develop task: writes a stub JPEG, or raises for chosen stems."""

    def __init__(self, fail=()):
        self.fail, self.tasks = set(fail), []

    def __call__(self, task):
        self.tasks.append(task)
        if task.source.stem in self.fail:
            raise RuntimeError("decode failed")
        task.out.write_bytes(b"jpg")


def hooks(spy, *, readings=None, records=None, warnings=(), **over):
    def read_records(paths):
        recs = {p: CaptureRecord(path=p) for p in paths}
        recs.update(records or {})
        return recs, list(warnings)

    def recognise(paths, optics=None):
        out = {p: ps.UNKNOWN_SUBJECT for p in paths}
        out.update(readings or {})
        return out

    return ps.SortHooks(
        **{
            "read_records": read_records,
            "recognise": recognise,
            "corrections": lambda paths: [(None, 0)] * len(paths),
            "develop": spy,
            "executor_factory": lambda workers: ThreadPoolExecutor(max_workers=workers),
            **over,
        }
    )


def opts(**kw):
    return ps.SortOptions(**kw)


def hash_tree(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def folder(tmp_path):
    src = tmp_path / "card" / "trip"
    jpeg(src / "a.jpg", 1)
    jpeg(src / "b.jpg", 2)
    jpeg(src / "c.jpg", 3)
    return src


@pytest.fixture
def root(tmp_path):
    return tmp_path / "out"


def run(folder, root, spy, options=None, **kw):
    return ps.run_sort(
        folder, root, options or opts(), hooks(spy, **kw), today=TODAY, say=lambda *_: None
    )


@pytest.mark.parametrize("kind", ["missing", "no_parent", "root_in_folder", "folder_in_root"])
def test_refusals_decode_and_write_nothing(tmp_path, folder, kind):
    spy = Spy()
    target, root = folder, tmp_path / "out"
    if kind == "missing":
        target = tmp_path / "nope"
    elif kind == "no_parent":
        root = tmp_path / "unmounted" / "camPics_output"
    elif kind == "root_in_folder":
        root = folder / "sorted"
    else:
        root = tmp_path / "card"
    before = hash_tree(tmp_path)
    with pytest.raises(ps.SortRefusalError):
        run(target, root, spy)
    assert spy.tasks == []
    assert hash_tree(tmp_path) == before
    assert not (tmp_path / "out").exists()


def test_run_folder_naming_and_collision(folder, root):
    first = run(folder, root, Spy())
    assert first.run_dir == root / "runs" / f"{TODAY}_trip"
    second = run(folder, root, Spy(), opts(reprocess=True))
    assert second.run_dir == root / "runs" / f"{TODAY}_trip_2"
    third = run(folder, root, Spy(), opts(reprocess=True))
    assert third.run_dir.name == f"{TODAY}_trip_3"


def test_every_unique_photo_in_one_category_and_one_row(folder, root):
    readings = {
        folder / "a.jpg": SubjectReading(SubjectCategory.ASTRO, 0.9),
        folder / "b.jpg": SubjectReading(SubjectCategory.CLOSE_UP, 0.65, evidence=("near",)),
        folder / "c.jpg": SubjectReading(SubjectCategory.EXPOSURE_BRACKET, 0.9),
    }
    spy = Spy()
    result = run(folder, root, spy, readings=readings)
    run_dir = result.run_dir
    assert (run_dir / "night-sky" / "001-a.jpg").is_file()
    assert (run_dir / "close-up" / "001-b.jpg").is_file()
    assert (run_dir / "hdr-source" / "001-c.jpg").is_file()
    ranked = rows(run_dir / "ranked.csv")
    assert sorted(r["category"] for r in ranked) == ["close-up", "hdr-source", "night-sky"]
    close = next(r for r in ranked if r["category"] == "close-up")
    assert close["rank"] == "1" and close["evidence"] == "near"
    assert close["source"] == str(folder / "b.jpg") and close["output"].endswith("001-b.jpg")
    assert close["quality_score"] == "" and close["aesthetic_score"] == ""
    by_stem = {t.source.stem: t for t in spy.tasks}
    assert by_stem["a"].style == "moody"
    assert by_stem["c"].style == "matte_shoulder"
    assert by_stem["b"].style is None
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["counts"] == {"night-sky": 1, "close-up": 1, "hdr-source": 1}
    assert meta["framegrade_version"] and meta["source"] == str(folder)


def test_capture_order_numbering_and_unknown_to_other(folder, root):
    records = {
        folder / "a.jpg": CaptureRecord(
            path=folder / "a.jpg", datetime_original="2026:01:01 10:00:02"
        ),
        folder / "b.jpg": CaptureRecord(
            path=folder / "b.jpg", datetime_original="2026:01:01 10:00:01"
        ),
        folder / "c.jpg": CaptureRecord(
            path=folder / "c.jpg", datetime_original="2026:01:01 10:00:03"
        ),
    }
    result = run(folder, root, Spy(), records=records)
    names = sorted(p.name for p in (result.run_dir / "other").iterdir())
    assert names == ["001-b.jpg", "002-a.jpg", "003-c.jpg"]
    ranked = rows(result.run_dir / "ranked.csv")
    assert [r["capture_time"] for r in ranked] == [
        "2026:01:01 10:00:01",
        "2026:01:01 10:00:02",
        "2026:01:01 10:00:03",
    ]


def test_renamed_duplicate_goes_to_duplicates_csv(folder, root):
    (folder / "a_copy.jpg").write_bytes((folder / "a.jpg").read_bytes())
    spy = Spy()
    result = run(folder, root, spy)
    assert {t.source.stem for t in spy.tasks} == {"a", "b", "c"}
    dups = rows(result.run_dir / "duplicates.csv")
    assert len(dups) == 1
    assert dups[0]["duplicate"].endswith("a_copy.jpg") and dups[0]["tier"] == "exact_copy"
    assert dups[0]["kept"].endswith("a.jpg") and dups[0]["seen_run"] == ""
    assert result.counts["duplicates"] == 1


def test_second_run_all_seen_and_reprocess_overrides(folder, root):
    run(folder, root, Spy())
    spy = Spy()
    again = run(folder, root, spy)
    assert spy.tasks == []
    dups = rows(again.run_dir / "duplicates.csv")
    assert [d["tier"] for d in dups] == ["seen_before"] * 3
    assert dups[0]["seen_run"] == f"{TODAY}_trip"
    spy = Spy()
    run(folder, root, spy, opts(reprocess=True))
    assert len(spy.tasks) == 3


def test_index_only_then_run_sees_everything(folder, root):
    spy = Spy()
    indexed = run(folder, root, spy, opts(index_only=True))
    assert spy.tasks == []
    assert sorted(p.name for p in indexed.run_dir.iterdir()) == ["run.json"]
    again = run(folder, root, spy)
    assert spy.tasks == []
    assert {d["seen_run"] for d in rows(again.run_dir / "duplicates.csv")} == {
        f"index:trip:{TODAY}"
    }


def test_non_photo_files_are_skipped_with_reason(folder, root):
    (folder / "clip.MOV").write_bytes(b"mov")
    (folder / "a.xmp").write_text("x")
    (folder / "a.jpg.subject.json").write_text("{}")
    result = run(folder, root, Spy())
    skipped = rows(result.run_dir / "skipped.csv")
    assert [Path(r["path"]).name for r in skipped] == ["clip.MOV"]
    assert skipped[0]["reason"]


def test_develop_failure_skips_and_continues_and_is_not_recorded(folder, root):
    result = run(folder, root, Spy(fail={"b"}))
    skipped = rows(result.run_dir / "skipped.csv")
    assert [Path(r["path"]).name for r in skipped] == ["b.jpg"]
    assert "decode failed" in skipped[0]["reason"]
    assert len(rows(result.run_dir / "ranked.csv")) == 2
    spy = Spy()
    run(folder, root, spy)
    assert [t.source.stem for t in spy.tasks] == ["b"]


def test_corrections_failure_falls_back_per_file(folder, root):
    def corrections(paths):
        if len(paths) > 1:
            raise RuntimeError("cluster failed")
        if paths[0].stem == "b":
            raise RuntimeError("measure failed")
        return [(None, 0)]

    result = run(folder, root, Spy(), corrections=corrections)
    skipped = rows(result.run_dir / "skipped.csv")
    assert [Path(r["path"]).name for r in skipped] == ["b.jpg"]
    assert len(rows(result.run_dir / "ranked.csv")) == 2


def test_corrections_chunked_at_limit(tmp_path, root, monkeypatch):
    src = tmp_path / "many"
    for i in range(5):
        jpeg(src / f"p{i}.jpg", i)
    monkeypatch.setattr(ps, "CORRECTION_CHUNK", 2)
    sizes = []
    run(src, root, Spy(), corrections=lambda p: sizes.append(len(p)) or [(None, 0)] * len(p))
    assert sizes == [2, 2, 1]


def test_close_up_routes_from_capture_records(folder, root):
    records = {
        folder / "a.jpg": CaptureRecord(
            path=folder / "a.jpg", focus_distance_m=0.3, focal_length_mm=100.0
        ),
    }
    seen = {}

    def recognise(paths, optics=None):
        seen.update(optics)
        return recognise_photos(paths, optics=optics)

    result = run(folder, root, Spy(), records=records, recognise=recognise)
    assert seen[folder / "a.jpg"] == CaptureOptics(
        focal_length_mm=100.0, f_number=None, focus_distance_m=0.3, lens_model=None
    )
    assert (result.run_dir / "close-up" / "001-a.jpg").is_file()
    assert (result.run_dir / "other" / "001-b.jpg").is_file()


def test_dry_run_writes_nothing_and_prints_estimate(folder, root):
    said = []
    spy = Spy()
    before = hash_tree(folder.parent)
    result = ps.run_sort(
        folder, root, opts(dry_run=True, workers=3), hooks(spy), today=TODAY, say=said.append
    )
    assert spy.tasks == [] and result.run_dir is None
    assert not root.exists() and hash_tree(folder.parent) == before
    text = "\n".join(said)
    assert "3" in text and "estimate" in text.lower()
    assert result.estimate_s == pytest.approx(3 * ps.DEVELOP_S_PER_PHOTO_ESTIMATE / 3)


def test_dry_run_reads_existing_ledger_without_creating_one(folder, root):
    run(folder, root, Spy())
    result = ps.run_sort(
        folder, root, opts(dry_run=True), hooks(Spy()), today=TODAY, say=lambda *_: None
    )
    assert result.counts["duplicates"] == 3


def test_index_html_lists_every_output_escaped(tmp_path, root):
    src = tmp_path / "card" / "a&b"
    jpeg(src / "x<y>&z.jpg", 4)
    jpeg(src / "plain.jpg", 5)
    result = run(src, root, Spy())
    page = (result.run_dir / "index.html").read_text()
    assert "002-x&lt;y&gt;&amp;z.jpg" in page and "<y>" not in page
    assert "001-plain.jpg" in page
    assert 'loading="lazy"' in page and "<script" not in page
    assert "other/001-plain.jpg" in page and "http" not in page
    assert page.count("<section") == 1


def test_originals_unchanged_and_flags_recorded(folder, root):
    (folder / "clip.MOV").write_bytes(b"mov")
    before = hash_tree(folder.parent)
    result = run(folder, root, Spy(), opts(no_stacks=True, workers=9))
    assert hash_tree(folder.parent) == before
    args = json.loads((result.run_dir / "run.json").read_text())["args"]
    assert args["no_stacks"] is True and args["workers"] == 4
    assert "gpu" not in args


def test_warnings_and_summary(folder, root):
    said = []
    result = ps.run_sort(
        folder,
        root,
        opts(),
        hooks(Spy(), warnings=["exiftool missing"]),
        today=TODAY,
        say=said.append,
    )
    assert json.loads((result.run_dir / "run.json").read_text())["warnings"] == ["exiftool missing"]
    assert any("other" in line and "3" in line for line in said)


def test_raw_jpeg_pair_collapses_onto_raw(tmp_path, root):
    src = tmp_path / "card" / "pair"
    jpeg(src / "DSC_1.jpg", 1)
    (src / "DSC_1.NEF").write_bytes(b"raw")
    spy = Spy()
    result = run(src, root, spy)
    assert [t.source.name for t in spy.tasks] == ["DSC_1.NEF"]
    assert "RAW+JPEG" in rows(result.run_dir / "skipped.csv")[0]["reason"]


def test_failed_run_records_nothing(folder, root):
    with pytest.raises(ZeroDivisionError):
        run(folder, root, Spy(), recognise=lambda p, optics=None: 1 / 0)
    spy = Spy()
    run(folder, root, spy)
    assert len(spy.tasks) == 3


def test_clamp_workers():
    assert [ps.clamp_workers(n) for n in (-1, 0, 1, 3, 4, 99)] == [1, 1, 1, 3, 4, 4]


def test_develop_one_real_grade_writes_jpeg(tmp_path):
    from gomoveshift.pipeline.photos.develop import compute_batch_corrections

    src = jpeg(tmp_path / "x.jpg", 7)
    params, index = compute_batch_corrections([src])[0]
    for style in (None, "moody"):
        out = tmp_path / f"{style}.jpg"
        ps.develop_one(ps.DevelopTask(src, out, params, index, style))
        assert Image.open(out).size == (24, 16)


def test_real_default_hooks_end_to_end(folder, root):
    result = ps.run_sort(
        folder,
        root,
        opts(workers=1),
        ps.SortHooks(read_records=lambda p: ({x: CaptureRecord(path=x) for x in p}, [])),
        today=TODAY,
        say=lambda *_: None,
    )
    assert len(rows(result.run_dir / "ranked.csv")) == 3
    assert len(list((result.run_dir / "other").glob("*.jpg"))) == 3


def test_lower_priority_ignores_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(ps.os, "nice", lambda n: calls.append(n))
    ps.lower_priority()
    assert calls == [19]

    def boom(n):
        raise OSError("no")

    monkeypatch.setattr(ps.os, "nice", boom)
    ps.lower_priority()


def test_default_develop_pool_workers_inherit_no_parent_state(monkeypatch):
    """The parent measures corrections with rawpy (OpenMP) before the pool starts.

    A forked worker inherits that OpenMP lock state and hangs on its first decode
    (seen 2026-11-09: 452 photos measured, then two workers idle 26 min, 0 written).
    A spawned worker is a fresh interpreter: it sees none of the parent's runtime
    state, which a marker set here (and only here) shows.
    """
    import pydoc

    monkeypatch.setattr(ps, "_PARENT_ONLY_MARKER", "inherited", raising=False)
    with ps.SortHooks().executor_factory(1) as pool:
        seen = pool.submit(pydoc.locate, f"{ps.__name__}._PARENT_ONLY_MARKER")
        assert seen.result(timeout=120) is None


# --- --resort: re-classify an existing run folder, never develop or score again ---------


class CopySpy(Spy):
    """Develop that writes a real JPEG (a copy of the source) so scoring can read it."""

    def __call__(self, task):
        self.tasks.append(task)
        task.out.write_bytes(task.source.read_bytes())


class FuseSpy:
    """Fake group ops: brackets are fixed, fuse writes a stub and is counted."""

    def __init__(self, brackets=()):
        self.brackets, self.fused = [tuple(b) for b in brackets], []

    def ops(self):
        def fuse(paths, out_path):
            self.fused.append(out_path.name)
            out_path.write_bytes(b"hdr")

        return ps.sort_groups.GroupOps(
            detect_sequences=lambda paths: [],
            chunk_sequence=lambda run: [tuple(run)],
            detect_brackets=lambda paths: [b for b in self.brackets if all(p in paths for p in b)],
            phash=lambda rgb: 0,
            fuse=fuse,
            classify_night=lambda paths: (list(paths), []),
        )


def _score(rgb, category, *, aesthetic, path, exposure_s, focal_mm):
    t = {"a": 0.9, "b": 0.5, "c": 0.7, "d": 0.8}.get(path.stem, 0.1)
    return ps.ShotScore(t, t, aesthetic, t, {}, {})


def grouped(spy, fuse, **kw):
    return hooks(spy, **{"score": _score, "groups": fuse.ops(), **kw})


def resort(folder, root, run_dir, hk, options=None):
    return ps.run_sort(
        folder,
        root,
        options or opts(),
        hk,
        today=TODAY,
        say=lambda *_: None,
        resort=run_dir,
    )


@pytest.fixture
def first_run(folder, root):
    fuse = FuseSpy(brackets=[(folder / "a.jpg", folder / "b.jpg", folder / "c.jpg")])
    jpeg(folder / "d.jpg", 4)
    result = ps.run_sort(
        folder, root, opts(), grouped(CopySpy(), fuse), today=TODAY, say=lambda *_: None
    )
    return result.run_dir, fuse


def test_resort_moves_a_changed_shot_and_rewrites_csv_and_page(folder, root, first_run):
    run_dir, fuse = first_run
    assert (run_dir / "other" / "001-a.jpg").is_file()
    readings = {folder / "d.jpg": SubjectReading(SubjectCategory.LANDSCAPE, 0.8)}
    result = resort(folder, root, run_dir, grouped(Spy(), fuse, readings=readings))
    assert (run_dir / "landscape" / "001-d.jpg").is_file()
    assert not list(run_dir.glob("other/*-d.jpg"))
    by_stem = {Path(r["source"]).stem: r for r in rows(run_dir / "ranked.csv")}
    assert by_stem["d"]["category"] == "landscape"
    assert by_stem["d"]["output"] == str(run_dir / "landscape" / "001-d.jpg")
    assert by_stem["d"]["quality_score"] == "0.8000"  # reused, not re-scored
    assert [by_stem[s]["rank"] for s in "acb"] == ["1", "2", "3"]  # quality order kept
    assert "landscape/001-d.jpg" in (run_dir / "index.html").read_text(encoding="utf-8")
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["resort_of"] == str(run_dir) and meta["args"]["resort_of"] == str(run_dir)
    assert result.counts["landscape"] == 1 and result.counts["other"] == 3


def test_resort_never_develops_or_scores(folder, root, first_run):
    run_dir, fuse = first_run
    spy = Spy()

    def no_score(*a, **k):
        raise AssertionError("scored")

    resort(folder, root, run_dir, grouped(spy, fuse, score=no_score))
    assert spy.tasks == []


def test_resort_reuses_a_finished_group_output(folder, root, first_run):
    run_dir, fuse = first_run
    assert len(fuse.fused) == 1
    resort(folder, root, run_dir, grouped(Spy(), fuse))
    assert len(fuse.fused) == 1  # STACK_MARKER in the ledger: not rebuilt
    assert json.loads((run_dir / "run.json").read_text())["groups"]["resumed"] == 1


def test_resort_refuses_a_foreign_dir_and_a_dir_without_ranked_csv(
    tmp_path, folder, root, first_run
):
    run_dir, fuse = first_run
    hk = grouped(Spy(), fuse)
    bare = root / "bare"
    bare.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "ranked.csv").write_text("category\n")
    for bad in (bare, elsewhere):
        with pytest.raises(ps.SortRefusalError):
            resort(folder, root, bad, hk)
    with pytest.raises(ps.SortRefusalError):
        resort(folder, root, run_dir, ps.SortHooks())  # no group hooks
    (run_dir / "other" / "001-a.jpg").unlink()
    with pytest.raises(ps.SortRefusalError):  # a JPEG named in ranked.csv is gone
        resort(folder, root, run_dir, hk)


def test_resort_refuses_with_resume_index_only_or_dry_run(folder, root, first_run):
    run_dir, fuse = first_run
    for options in (opts(index_only=True), opts(dry_run=True)):
        with pytest.raises(ps.SortRefusalError):
            resort(folder, root, run_dir, grouped(Spy(), fuse), options)
    with pytest.raises(ps.SortRefusalError):
        ps.run_sort(folder, root, opts(), grouped(Spy(), fuse), resume=run_dir, resort=run_dir)
