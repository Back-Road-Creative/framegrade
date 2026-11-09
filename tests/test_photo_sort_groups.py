"""photos sort: scene-group tags, HDR / panorama / night-stack outputs, resume."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.photos import sort as ps
from gomoveshift.pipeline.photos import sort_groups as sg
from gomoveshift.pipeline.photos.capture_meta import CaptureRecord
from gomoveshift.pipeline.photos.dedupe import Ledger
from gomoveshift.pipeline.photos.subject import SubjectCategory, SubjectReading

TODAY = "2026-11-09"


def jpeg(path, seed=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (16, 24, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def develop_copy(task):
    task.out.write_bytes(task.source.read_bytes())


class Totals:
    def __init__(self, totals):
        self.totals = totals

    def __call__(self, rgb, category, *, aesthetic, path, exposure_s, focal_mm):
        t = self.totals.get(path.stem, 0.5)
        return ps.ShotScore(t, t, aesthetic, t, {}, {})


class Ops:
    """Fake GroupOps pieces that count calls and write tiny outputs."""

    def __init__(self, sequences=(), brackets=(), fail=(), hashes=None):
        self.sequences, self.brackets, self.fail = list(sequences), list(brackets), set(fail)
        self.hashes = hashes or {}
        self.calls = []

    def detect_sequences(self, paths):
        return [tuple(s) for s in self.sequences if all(p in paths for p in s)]

    def chunk_sequence(self, run):
        return [tuple(run)]

    def detect_brackets(self, paths):
        return [tuple(b) for b in self.brackets if all(p in paths for p in b)]

    def phash(self, rgb):
        return self.hashes.get(int(rgb[0, 0, 0]), int(rgb[0, 0, 0]) << 40)

    def fuse(self, paths, out_path):
        self.calls.append(("fuse", out_path.name))
        if "fuse" in self.fail:
            raise RuntimeError("fuse boom")
        out_path.write_bytes(b"hdr")

    def stitch(self, paths, out_path):
        self.calls.append(("stitch", out_path.name))
        if "stitch" in self.fail:
            raise sg_error("seam gate")
        out_path.write_bytes(b"pano")

    def prepare(self, paths, *, decode_workers, output_dir):
        self.calls.append(("prepare", decode_workers))
        return SimpleNamespace(paths=tuple(paths))

    def stack(self, prepared, out_path):
        self.calls.append(("stack", out_path.name))
        out_path.write_bytes(b"tif")
        preview = out_path.with_suffix(".preview.jpg")
        preview.write_bytes(b"jpg")
        return SimpleNamespace(preview_path=preview)

    def ops(self):
        return sg.GroupOps(
            detect_sequences=self.detect_sequences,
            chunk_sequence=self.chunk_sequence,
            detect_brackets=self.detect_brackets,
            phash=self.phash,
            fuse=self.fuse,
            stitch=self.stitch,
            prepare=self.prepare,
            stack=self.stack,
        )


def sg_error(msg):
    return ValueError(msg)


def rec(path, second=0, focal=24.0, f=8.0, exp=0.01):
    return CaptureRecord(
        path=path,
        datetime_original=f"2026:11:09 10:00:{second:02d}",
        focal_length_mm=focal,
        f_number=f,
        exposure_s=exp,
    )


def make_hooks(ops, score=None, records=None, readings=None):
    def read_records(paths):
        recs = {p: CaptureRecord(path=p) for p in paths}
        recs.update(records or {})
        return recs, []

    def recognise(paths, optics=None):
        out = {p: ps.UNKNOWN_SUBJECT for p in paths}
        out.update(readings or {})
        return out

    return ps.SortHooks(
        read_records=read_records,
        recognise=recognise,
        corrections=lambda paths: [(None, 0)] * len(paths),
        develop=develop_copy,
        executor_factory=lambda workers: ThreadPoolExecutor(max_workers=workers),
        score=score or Totals({}),
        groups=ops.ops(),
    )


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(folder, root, hooks, options=None, **kw):
    return ps.run_sort(
        folder, root, options or ps.SortOptions(), hooks, today=TODAY, say=lambda *_: None, **kw
    )


def stems(folder, names):
    return [folder / f"{n}.jpg" for n in names]


@pytest.fixture
def root(tmp_path):
    return tmp_path / "out"


def card(tmp_path, names):
    src = tmp_path / "card"
    for i, n in enumerate(names, 1):
        jpeg(src / f"{n}.jpg", i)
    return src


def night_readings(paths):
    return {p: SubjectReading(SubjectCategory.ASTRO, 0.9) for p in paths}


def test_night_session_tags_best_rank_and_builds_stack(tmp_path, root):
    src = card(tmp_path, list("abcde"))
    seq = stems(src, "abcde")
    ops = Ops(sequences=[seq])
    totals = Totals({"a": 0.2, "b": 0.9, "c": 0.4, "d": 0.3, "e": 0.1})
    result = run(src, root, make_hooks(ops, totals, readings=night_readings(seq)))
    ranked = rows(result.run_dir / "ranked.csv")
    assert {r["group_id"] for r in ranked} == {"night-1"}
    assert {r["group_best_rank"] for r in ranked} == {"1"}
    page = (result.run_dir / "index.html").read_text()
    assert "same scene as #001" in page
    assert (result.run_dir / "stacks" / "night-1.jpg").is_file()
    assert (result.run_dir / "stacks" / "night-1.tif").is_file()
    assert all(Path(r["output"]).is_file() for r in ranked)
    assert ("prepare", 2) in ops.calls


def test_index_shows_stacks_first_in_night_sky(tmp_path, root):
    src = card(tmp_path, list("abcde"))
    seq = stems(src, "abcde")
    result = run(src, root, make_hooks(Ops(sequences=[seq]), readings=night_readings(seq)))
    page = (result.run_dir / "index.html").read_text()
    section = page.split("<h2>night-sky")[1]
    assert section.index("stacks/night-1.jpg") < section.index("001-")


def test_bracket_fused_into_hdr_and_members_stay(tmp_path, root):
    src = card(tmp_path, list("abc"))
    ops = Ops(brackets=[stems(src, "abc")])
    result = run(src, root, make_hooks(ops))
    assert [c[0] for c in ops.calls] == ["fuse"]
    assert (result.run_dir / "hdr" / "bracket-1-a.jpg").is_file()
    ranked = rows(result.run_dir / "ranked.csv")
    assert len(ranked) == 3 and {r["category"] for r in ranked} == {"other"}
    assert {r["group_id"] for r in ranked} == {"bracket-1"}
    assert "<h2>hdr (1)" in (result.run_dir / "index.html").read_text()


def pano_records(src, names, seconds, **kw):
    return {p: rec(p, s, **kw) for p, s in zip(stems(src, names), seconds)}


def test_pano_detector_run_gap_and_settings():
    p = [Path(f"/x/{i}.jpg") for i in range(6)]
    one = [rec(p[0], 0), rec(p[1], 2), rec(p[2], 4)]
    assert sg.detect_panorama_runs(one) == [tuple(p[:3])]
    gap = [rec(p[0], 0), rec(p[1], 2), rec(p[2], 11)]
    assert sg.detect_panorama_runs(gap) == []
    split = [rec(p[i], s) for i, s in enumerate((0, 2, 4, 20, 22, 24))]
    assert sg.detect_panorama_runs(split) == [tuple(p[:3]), tuple(p[3:])]
    focal = [rec(p[0], 0), rec(p[1], 2), rec(p[2], 4, focal=35.0)]
    assert sg.detect_panorama_runs(focal) == []
    missing = [rec(p[0], 0), rec(p[1], 2), CaptureRecord(path=p[2])]
    assert sg.detect_panorama_runs(missing) == []
    bad = [rec(p[0], 0), rec(p[1], 2), CaptureRecord(path=p[2], datetime_original="junk")]
    assert sg.detect_panorama_runs(bad) == []


def test_capture_datetime_formats_and_subsec():
    r = CaptureRecord(path=Path("a"), datetime_original="2026-11-09T10:00:01", subsec="50")
    assert sg.capture_datetime(r).microsecond == 500000
    assert sg.capture_datetime(CaptureRecord(path=Path("a"))) is None


def test_pano_built_and_stitch_failure_goes_to_skipped(tmp_path, root):
    src = card(tmp_path, list("abc"))
    recs = pano_records(src, "abc", (0, 2, 4))
    ops = Ops()
    result = run(src, root, make_hooks(ops, records=recs))
    assert (result.run_dir / "pano" / "pano-1-a.jpg").is_file()
    assert "<h2>pano (1)" in (result.run_dir / "index.html").read_text()
    root2 = tmp_path / "out2"
    bad = Ops(fail=["stitch"])
    result = run(src, root2, make_hooks(bad, records=recs))
    skipped = rows(result.run_dir / "skipped.csv")
    assert [(r["path"], r["reason"]) for r in skipped] == [
        ("group:pano-1", "pano failed: seam gate")
    ]
    assert not list((result.run_dir / "pano").glob("*.jpg"))
    assert len(rows(result.run_dir / "ranked.csv")) == 3


def test_fuse_failure_never_drops_members(tmp_path, root):
    src = card(tmp_path, list("abc"))
    result = run(src, root, make_hooks(Ops(brackets=[stems(src, "abc")], fail=["fuse"])))
    assert [r["path"] for r in rows(result.run_dir / "skipped.csv")] == ["group:bracket-1"]
    assert len(list((result.run_dir / "other").iterdir())) == 3


def test_stacks_run_highest_scoring_session_first(tmp_path, root):
    src = card(tmp_path, list("abcdefghij"))
    one, two = stems(src, "abcde"), stems(src, "fghij")
    ops = Ops(sequences=[one, two])
    totals = Totals({**dict.fromkeys("abcde", 0.2), **dict.fromkeys("fghij", 0.8)})
    run(src, root, make_hooks(ops, totals, readings=night_readings(one + two)))
    assert [c[1] for c in ops.calls if c[0] == "stack"] == ["night-2.tif", "night-1.tif"]


def test_no_stacks_skips_night_only(tmp_path, root):
    src = card(tmp_path, list("abcdefgh"))
    seq = stems(src, "abcde")
    ops = Ops(sequences=[seq], brackets=[stems(src, "fgh")])
    result = run(
        src,
        root,
        make_hooks(ops, readings=night_readings(seq)),
        ps.SortOptions(no_stacks=True),
    )
    assert [c[0] for c in ops.calls] == ["fuse"]
    ranked = rows(result.run_dir / "ranked.csv")
    assert {r["group_id"] for r in ranked if r["category"] == "night-sky"} == {"night-1"}
    assert not (result.run_dir / "stacks").exists()


def test_burst_groups_within_a_category_only(tmp_path, root):
    src = card(tmp_path, list("abcd"))
    # a and b look alike but sit in different categories; c and d look alike in one.
    readings = {src / "a.jpg": SubjectReading(SubjectCategory.LANDSCAPE, 0.9)}
    ops = Ops()
    ops.phash = lambda rgb: 0
    result = run(src, root, make_hooks(ops, readings=readings))
    ranked = {Path(r["source"]).stem: r for r in rows(result.run_dir / "ranked.csv")}
    assert ranked["a"]["group_id"] == ""
    assert {ranked[s]["group_id"] for s in "bcd"} == {"burst-1"}


def test_resume_does_only_missing_outputs(tmp_path, root):
    src = card(tmp_path, list("abcdefgh"))
    seq = stems(src, "abcde")
    ops = Ops(sequences=[seq], brackets=[stems(src, "fgh")], fail=["fuse"])
    first = run(src, root, make_hooks(ops, readings=night_readings(seq)))
    assert [c[0] for c in ops.calls].count("stack") == 1
    ops.fail.clear()
    ops.calls.clear()
    again = run(src, root, make_hooks(ops), resume=first.run_dir)
    assert again.run_dir == first.run_dir
    assert [c[0] for c in ops.calls] == ["fuse"]
    assert (first.run_dir / "hdr" / "bracket-1-f.jpg").is_file()
    assert rows(first.run_dir / "skipped.csv") == []
    assert json.loads((first.run_dir / "run.json").read_text())["groups"]["resumed"] == 1


def test_resume_redoes_group_without_marker_or_file(tmp_path, root):
    src = card(tmp_path, list("abc"))
    ops = Ops(brackets=[stems(src, "abc")])
    first = run(src, root, make_hooks(ops))
    out = first.run_dir / "hdr" / "bracket-1-a.jpg"
    out.unlink()
    ops.calls.clear()
    run(src, root, make_hooks(ops), resume=first.run_dir)
    assert len(ops.calls) == 1
    with Ledger(root / ps.LEDGER_NAME) as ledger:
        key = sg.group_key("bracket", [str(p) for p in stems(src, "abc")])
        assert ledger.get_score(key, sg.STACK_MARKER) == 1


def test_resume_refusals(tmp_path, root):
    src = card(tmp_path, list("abc"))
    hooks = make_hooks(Ops())
    with pytest.raises(ps.SortRefusalError):
        run(src, root, hooks, resume=root / "runs" / "nope")
    first = run(src, root, hooks)
    with pytest.raises(ps.SortRefusalError):
        run(src, root, ps.SortHooks(), resume=first.run_dir)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "ranked.csv").write_text("category\n")
    with pytest.raises(ps.SortRefusalError):
        run(src, root, hooks, resume=elsewhere)


def test_resume_keeps_old_non_group_skips(tmp_path, root):
    src = card(tmp_path, list("abc"))
    (src / "notes.txt").write_text("x")
    first = run(src, root, make_hooks(Ops()))
    again = run(src, root, make_hooks(Ops()), resume=first.run_dir)
    assert again.counts["skipped"] == 1
    assert [Path(r["path"]).name for r in rows(first.run_dir / "skipped.csv")] == ["notes.txt"]


def test_group_columns_in_ranked_csv_blank_without_groups(tmp_path, root):
    src = card(tmp_path, list("abc"))
    hooks = make_hooks(Ops())
    hooks = ps.SortHooks(**{**hooks.__dict__, "groups": None})
    result = run(src, root, hooks)
    ranked = rows(result.run_dir / "ranked.csv")
    assert all(r["group_id"] == "" and r["group_best_rank"] == "" for r in ranked)


def test_night_folder_constant_matches_sort():
    assert sg.NIGHT_FOLDER == ps.CATEGORY_FOLDERS[SubjectCategory.ASTRO]


def test_real_ops_call_the_pipeline_seams(tmp_path, monkeypatch):
    import gomoveshift.pipeline.photos.astro_orchestrator as ao
    import gomoveshift.pipeline.photos.develop as dv
    import gomoveshift.pipeline.photos.panorama as pn
    import gomoveshift.pipeline.photos.preflight as pf

    seen = []
    monkeypatch.setattr(pn, "stitch_panorama", lambda p, o: seen.append(("stitch", len(p))))
    monkeypatch.setattr(dv, "fuse_bracket", lambda p, align=True: "decoded")
    monkeypatch.setattr(dv, "compute_batch_corrections", lambda p: [("params", 3)])
    monkeypatch.setattr(
        dv,
        "develop",
        lambda d, params, idx: (
            seen.append(("develop", d, params, idx)) or SimpleNamespace(rgb="rgb")
        ),
    )
    monkeypatch.setattr(dv, "export_jpeg", lambda rgb, out: seen.append(("export", rgb)))
    monkeypatch.setattr(pf, "preflight_astro_stack", lambda *a, **k: 1)
    monkeypatch.setattr(
        ao, "prepare_frames", lambda paths, **k: seen.append(("prep", k["decode_workers"])) or "P"
    )
    monkeypatch.setattr(ao, "stack_run", lambda prepared, out: seen.append(("stack", prepared)))
    monkeypatch.setattr(sg, "_megapixels", lambda path: 20.0)
    ops = sg.real_ops()
    paths = [Path("a"), Path("b"), Path("c")]
    ops.stitch(paths, tmp_path / "p.jpg")
    ops.fuse(paths, tmp_path / "h.jpg")
    prepared = ops.prepare(paths, decode_workers=4, output_dir=tmp_path)
    ops.stack(prepared, tmp_path / "s.tif")
    assert ("stitch", 3) in seen and ("develop", "decoded", "params", 3) in seen
    assert ("prep", 1) in seen and ("stack", "P") in seen


def test_default_wrappers_run_on_trivial_input(monkeypatch):
    assert sg._detect_sequences([]) == []
    assert sg._chunk_sequence(()) == []
    assert sg._detect_brackets([]) == []
    assert isinstance(sg._phash(np.zeros((32, 32, 3), dtype=np.uint8)), int)

    class Raw:
        sizes = SimpleNamespace(width=4000, height=3000)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import rawpy

    monkeypatch.setattr(rawpy, "imread", lambda p: Raw())
    assert sg._megapixels("x.nef") == 12.0
