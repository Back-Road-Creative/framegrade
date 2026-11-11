"""photos sort: scene-group tags, HDR / panorama / night-stack outputs, resume."""

import csv
import json
import re
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

    def __init__(self, sequences=(), brackets=(), fail=(), hashes=None, chunk=None, darks=()):
        self.sequences, self.brackets, self.fail = list(sequences), list(brackets), set(fail)
        self.chunk, self.darks = chunk, {Path(d) for d in darks}
        self.lapse, self.trails = [], []
        self.hashes = hashes or {}
        self.calls = []

    def detect_sequences(self, paths):
        runs = [tuple(p for p in s if p in paths) for s in self.sequences]
        return [r for r in runs if len(r) >= 2]  # the sequencer sees only the frames it is given

    def chunk_sequence(self, run):
        step = self.chunk or len(run)
        return [tuple(run[i : i + step]) for i in range(0, len(run), step)]

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
        if "stack" in self.fail:
            raise RuntimeError("stack boom")
        out_path.write_bytes(b"tif")
        out_path.with_suffix(".linear").mkdir(exist_ok=True)  # the real stack_run writes one
        preview = out_path.with_suffix(".preview.jpg")
        preview.write_bytes(b"jpg")
        return SimpleNamespace(preview_path=preview)

    def combine_session(self, linears, out_tif, out_jpg):
        self.calls.append(("combine_session", out_tif.name))
        self.combined = [Path(p).name for p in linears]
        if "combine" in self.fail:
            raise RuntimeError("combine boom")
        out_tif.write_bytes(b"deep")
        out_jpg.write_bytes(b"jpg")

    def classify_night(self, paths):
        self.calls.append(("classify_night", len(paths)))
        return (
            [p for p in paths if Path(p) not in self.darks],
            [p for p in paths if Path(p) in self.darks],
        )

    def night_timelapse(self, lights, darks, out_mp4, out_jpg):
        self.calls.append(("night_timelapse", out_mp4.name))
        self.lapse.append((list(lights), list(darks)))
        out_mp4.write_bytes(b"mp4")
        if "night_timelapse" in self.fail:
            raise RuntimeError("lapse boom")
        out_jpg.write_bytes(b"jpg")

    def star_trails(self, lights, darks, out_tif, out_jpg):
        self.calls.append(("star_trails", out_tif.name))
        self.trails.append((list(lights), list(darks)))
        out_tif.write_bytes(b"tif")
        if "star_trails" in self.fail:
            raise RuntimeError("trail boom")
        out_jpg.write_bytes(b"jpg")

    def still_timelapse(self, jpegs, out_mp4, out_jpg):
        self.calls.append(("still_timelapse", out_mp4.name))
        self.lapse.append((list(jpegs), []))
        out_mp4.write_bytes(b"mp4")
        out_jpg.write_bytes(b"jpg")

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
            combine_session=self.combine_session,
            classify_night=self.classify_night,
            night_timelapse=self.night_timelapse,
            still_timelapse=self.still_timelapse,
            star_trails=self.star_trails,
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
    assert "same scene as" not in page
    assert (result.run_dir / "stacks" / "night-1.jpg").is_file()
    assert (result.run_dir / "stacks" / "night-1.tif").is_file()
    assert all(Path(r["output"]).is_file() for r in ranked)
    assert ("prepare", 2) in ops.calls


def test_index_shows_stacks_first_in_night_sky(tmp_path, root):
    src = card(tmp_path, list("abcde"))
    seq = stems(src, "abcde")
    result = run(src, root, make_hooks(Ops(sequences=[seq]), readings=night_readings(seq)))
    page = (result.run_dir / "index.html").read_text()
    section = page.split("<h3>Night sky")[1]
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
    page = (result.run_dir / "index.html").read_text()
    assert "<h3>Unsorted (3 shots)" in page and "<h3>HDR" not in page
    assert page.count('class="card"') == 1 and "same scene as" not in page


def test_bracket_frames_and_fuse_share_the_hdr_folder(tmp_path, root):
    src = card(tmp_path, list("abc"))
    seq = stems(src, "abc")
    readings = {p: SubjectReading(SubjectCategory.EXPOSURE_BRACKET, 0.9) for p in seq}
    result = run(src, root, make_hooks(Ops(brackets=[seq]), readings=readings))
    names = sorted(p.name for p in (result.run_dir / "hdr").iterdir())
    assert len(names) == 4 and "bracket-1-a.jpg" in names
    assert sum(n[:3].isdigit() for n in names) == 3
    assert not (result.run_dir / "hdr-source").exists()
    page = (result.run_dir / "index.html").read_text()
    assert page.count("HDR ·") == 1 and page.count('class="card"') == 1


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
    page = (result.run_dir / "index.html").read_text()
    assert "<h2>pano" not in page and page.count('class="card"') == 1
    assert page.index("pano/pano-1-a.jpg") < page.index("<details")
    root2 = tmp_path / "out2"
    bad = Ops(fail=["stitch"])
    result = run(src, root2, make_hooks(bad, records=recs))
    skipped = rows(result.run_dir / "skipped.csv")
    assert [(r["path"], r["reason"]) for r in skipped] == [
        ("group:pano-1", "pano failed: seam gate")
    ]
    assert not list((result.run_dir / "pano").glob("*.jpg"))
    page = (result.run_dir / "index.html").read_text()
    assert "burst of 3" in page and "same scene as" not in page
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


def _singles(n, category="landscape"):
    paths = [Path(f"/card/s{i}.nef") for i in range(n)]
    ranked = [
        {"source": str(p), "category": category, "rank": i + 1, "quality_score": "0.5"}
        for i, p in enumerate(paths)
    ]
    return paths, ranked


@pytest.mark.parametrize(
    "gap_s, bits, grouped",
    [
        (9, 14, True),  # hand-held look-alikes seconds apart (DSC_1900/1901: 9 s, 14 bits)
        (9, 16, True),  # BURST_HAMMING_THRESHOLD
        (9, 17, False),
        (61, 14, False),  # alike but minutes apart: two shots
        (7200, 3, True),  # a true near-duplicate groups at any distance in time
    ],
)
def test_lookalike_singles_seconds_apart_form_a_burst(gap_s, bits, grouped):
    paths, ranked = _singles(2)
    records = interval_records(paths, [gap_s])
    hashes = {str(paths[0]): 0, str(paths[1]): (1 << bits) - 1}
    sg.tag_groups(ranked, records, Ops().ops(), hashes)
    ids = {r.get("group_id", "") for r in ranked}
    assert ids == ({"burst-1"} if grouped else {""})


def test_loose_burst_match_needs_both_capture_times():
    paths, ranked = _singles(2)
    records = {paths[0]: rec(paths[0]), paths[1]: CaptureRecord(path=paths[1])}
    sg.tag_groups(ranked, records, Ops().ops(), {str(paths[0]): 0, str(paths[1]): (1 << 14) - 1})
    assert {r.get("group_id", "") for r in ranked} == {""}


def test_burst_of_two_lookalikes_shows_once_on_the_page(tmp_path):
    paths, ranked = _singles(2)
    records = interval_records(paths, [9])
    sg.tag_groups(ranked, records, Ops().ops(), {str(paths[0]): 0, str(paths[1]): (1 << 14) - 1})
    for i, row in enumerate(ranked):
        row.update(output=str(tmp_path / f"landscape/00{i + 1}-s{i}.jpg"), capture_time="")
    page = ps._index_html(tmp_path, Path("/card"), ranked)
    assert "burst of 2" in page
    shown = re.sub(r"<details>.*?</details>", "", page, flags=re.S)
    assert shown.count("<img") == 1  # the other look-alike is folded behind it


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
    monkeypatch.setattr(
        ao, "stack_run", lambda prepared, out, **k: seen.append(("stack", prepared, k))
    )
    monkeypatch.setattr(sg, "_megapixels", lambda path: 20.0)
    ops = sg.real_ops()
    paths = [Path("a"), Path("b"), Path("c")]
    ops.stitch(paths, tmp_path / "p.jpg")
    ops.fuse(paths, tmp_path / "h.jpg")
    prepared = ops.prepare(paths, decode_workers=4, output_dir=tmp_path)
    ops.stack(prepared, tmp_path / "s.tif")
    assert ("stitch", 3) in seen and ("develop", "decoded", "params", 3) in seen
    assert ("prep", 1) in seen
    # each chunk stack keeps its linear artifact: the session combine's input
    assert ("stack", "P", {"linear_artifact_path": tmp_path / "s.linear"}) in seen


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


def test_pano_stitches_the_developed_jpegs_not_the_raw_sources(tmp_path, root):
    """The stitcher reads with ``cv2.imread``, which cannot open a camera RAW: on a real
    card (2026-11-09) every panorama failed "DSC_1876.NEF: could not be read as an image".
    It must be handed the developed JPEGs this run wrote, in capture order."""
    src = card(tmp_path, list("abc"))
    recs = pano_records(src, "abc", (0, 2, 4))
    seen = []
    ops = Ops()
    real_stitch = ops.stitch

    def stitch(paths, out_path):
        seen.append([Path(p) for p in paths])
        real_stitch(paths, out_path)

    ops.stitch = stitch
    result = run(src, root, make_hooks(ops, records=recs))
    by_source = {Path(r["source"]): Path(r["output"]) for r in rows(result.run_dir / "ranked.csv")}
    assert seen == [[by_source[p] for p in stems(src, "abc")]]
    assert all(p.is_relative_to(result.run_dir) for p in seen[0])


def night_records(paths, start_minute=0):
    """One frame a second from ``start_minute``: a run the sequencer would call one session."""
    return {
        p: CaptureRecord(
            path=p, datetime_original=f"2026:11:09 22:{start_minute + i // 60:02d}:{i % 60:02d}"
        )
        for i, p in enumerate(paths)
    }


def lapse_calls(ops, name="night_timelapse"):
    return [c for c in ops.calls if c[0] == name]


def test_night_session_of_two_chunks_renders_one_timelapse(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], chunk=15)
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    result = run(src, root, hooks)
    assert {r["group_id"] for r in rows(result.run_dir / "ranked.csv")} == {"night-1", "night-2"}
    assert lapse_calls(ops) == [("night_timelapse", "night-1.mp4")]
    assert ops.lapse == [(seq, [])]  # every light, capture order, one render
    assert (result.run_dir / "timelapse" / "night-1.mp4").is_file()
    assert (result.run_dir / "timelapse" / "night-1.jpg").is_file()
    page = (result.run_dir / "index.html").read_text()
    assert page  # the page still renders with the new output kind
    with Ledger(root / ps.LEDGER_NAME) as ledger:
        key = sg.group_key("timelapse", [str(p) for p in seq])
        assert ledger.get_score(key, sg.STACK_MARKER) == 1


def test_separate_runs_are_separate_sessions(tmp_path, root):
    names = [f"n{i:02d}" for i in range(50)]
    src = card(tmp_path, names)
    one, two = stems(src, names[:25]), stems(src, names[25:])
    records = {**night_records(one), **night_records(two, start_minute=10)}
    ops = Ops(sequences=[one, two])
    run(src, root, make_hooks(ops, records=records, readings=night_readings(one + two)))
    assert sorted(c[1] for c in lapse_calls(ops)) == ["night-1.mp4", "night-2.mp4"]


def test_lens_cap_darks_are_never_rendered_as_lights(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    darks = seq[-4:]
    ops = Ops(sequences=[seq], chunk=15, darks=darks)
    run(src, root, make_hooks(ops, records=night_records(seq), readings=night_readings(seq)))
    lights, passed = ops.lapse[0]
    assert lights == seq[:-4] and passed == darks
    assert not set(lights) & set(passed)


def dark_card(tmp_path, n_lights=26, n_darks=4):
    """A night run whose last frames are lens-cap darks, one frame a second from 22:00."""
    names = [f"n{i:02d}" for i in range(n_lights + n_darks)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    return src, seq, seq[:n_lights], seq[n_lights:]


def test_lens_cap_darks_get_their_own_folder_and_stay_off_the_page(tmp_path, root):
    src, seq, lights, darks = dark_card(tmp_path)
    ops = Ops(sequences=[seq], darks=darks)
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    result = run(src, root, hooks)
    ranked = rows(result.run_dir / "ranked.csv")
    by_stem = {Path(r["source"]).stem: r for r in ranked}
    assert {by_stem[p.stem]["category"] for p in darks} == {sg.DARKS_FOLDER}
    assert {by_stem[p.stem]["group_id"] for p in darks} == {""}
    assert {by_stem[p.stem]["category"] for p in lights} == {sg.NIGHT_FOLDER}
    page = (result.run_dir / "index.html").read_text()
    assert f'src="{sg.DARKS_FOLDER}/' not in page and f'href="{sg.DARKS_FOLDER}/' not in page
    assert "4 calibration frames (lens cap on), used for the stacks, not shown" in page
    assert not any(sg.DARKS_FOLDER in line for line in ps._top_lines(ranked))


def test_darks_never_form_a_burst_bracket_or_pano():
    srcs = [f"/card/d{i}.nef" for i in range(4)]
    ranked = [
        {"source": s, "category": sg.DARKS_FOLDER, "rank": i + 1, "quality_score": "0.5"}
        for i, s in enumerate(srcs)
    ]
    records = {Path(s): rec(Path(s), second=i) for i, s in enumerate(srcs)}
    ops = Ops(brackets=[[Path(s) for s in srcs[:3]]])
    sg.tag_groups(ranked, records, ops.ops(), {s: 0 for s in srcs})
    assert [r.get("group_id", "") for r in ranked] == [""] * 4


def test_session_darks_calibrate_its_trail_and_timelapse(tmp_path, root):
    src, seq, lights, darks = dark_card(tmp_path)
    far = jpeg(src / "far.jpg", 99)  # a dark shot an hour later belongs to no session here
    records = {
        **night_records(seq),
        far: CaptureRecord(far, datetime_original="2026:11:09 23:30:00"),
    }
    ops = Ops(sequences=[seq], darks=[*darks, far])
    run(src, root, make_hooks(ops, records=records, readings=night_readings([*seq, far])))
    assert ops.lapse == [(lights, darks)]
    assert ops.trails == [(lights, darks)]


def test_resort_reuses_the_first_runs_lens_cap_verdict(tmp_path, root):
    src, seq, lights, darks = dark_card(tmp_path)
    ops = Ops(sequences=[seq], darks=darks)
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    result = run(src, root, hooks)
    ops.calls.clear()
    run(src, root, hooks, resort=result.run_dir)
    assert lapse_calls(ops, "classify_night") == []
    by_stem = {Path(r["source"]).stem: r for r in rows(result.run_dir / "ranked.csv")}
    assert {by_stem[p.stem]["category"] for p in darks} == {sg.DARKS_FOLDER}
    assert {by_stem[p.stem]["category"] for p in lights} == {sg.NIGHT_FOLDER}


def test_resort_checks_only_frames_that_newly_read_as_night(tmp_path, root):
    src, seq, _lights, darks = dark_card(tmp_path)
    ops = Ops(sequences=[seq], darks=darks)
    first = make_hooks(ops, records=night_records(seq), readings=night_readings(seq[:-1]))
    result = run(src, root, first)
    ops.calls.clear()
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    run(src, root, hooks, resort=result.run_dir)
    assert lapse_calls(ops, "classify_night") == [("classify_night", 1)]
    by_stem = {Path(r["source"]).stem: r for r in rows(result.run_dir / "ranked.csv")}
    assert by_stem[seq[-1].stem]["category"] == sg.DARKS_FOLDER


def test_resort_does_not_retry_a_stack_that_already_failed(tmp_path, root):
    names = [f"n{i:02d}" for i in range(10)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], fail=["stack"])
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    result = run(src, root, hooks)
    ops.calls.clear()
    run(src, root, hooks, resort=result.run_dir)
    assert [c for c in ops.calls if c[0] in ("prepare", "stack")] == []
    reasons = {r["path"]: r["reason"] for r in rows(result.run_dir / "skipped.csv")}
    assert reasons["group:night-1"] == "night failed in an earlier run; --resume retries it"
    ops.calls.clear()
    run(src, root, hooks, resume=result.run_dir)
    assert [c[1] for c in ops.calls if c[0] == "stack"] == ["night-1.tif"]


def test_a_failed_dark_triage_keeps_every_frame_in_the_night_sky(tmp_path, root):
    src, seq, _lights, darks = dark_card(tmp_path)

    class Boom(Ops):
        def classify_night(self, paths):
            raise RuntimeError("decode boom")

    ops = Boom(sequences=[seq], darks=darks)
    warnings = []
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    result = ps.run_sort(src, root, ps.SortOptions(), hooks, today=TODAY, say=warnings.append)
    assert {r["category"] for r in rows(result.run_dir / "ranked.csv")} == {sg.NIGHT_FOLDER}
    assert any("lens-cap dark check failed" in w for w in warnings)


def test_session_with_too_few_usable_lights_is_skipped_quietly(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], darks=seq[sg.TIMELAPSE_MIN_FRAMES - 6 :])  # 18 lights left
    result = run(
        src, root, make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    )
    assert lapse_calls(ops) == [] and rows(result.run_dir / "skipped.csv") == []
    assert lapse_calls(ops, "star_trails") == []
    assert not (result.run_dir / "timelapse").exists()
    assert not (result.run_dir / "trails").exists()
    ops.calls.clear()
    run(src, root, make_hooks(ops), resume=result.run_dir)  # the verdict is remembered
    assert lapse_calls(ops, "classify_night") == []


def test_burst_timelapse_needs_the_minimum_member_count(tmp_path, root):
    names = [f"b{i:02d}" for i in range(30)] + [f"c{i}" for i in range(5)]
    src = card(tmp_path, names)
    big, small = stems(src, names[:30]), stems(src, names[30:])
    readings = {p: SubjectReading(SubjectCategory.LANDSCAPE, 0.9) for p in big}
    ops = Ops()
    ops.phash = lambda rgb: 0  # every frame looks alike within its category
    result = run(src, root, make_hooks(ops, readings=readings))
    ranked = rows(result.run_dir / "ranked.csv")
    assert {r["group_id"] for r in ranked if r["category"] == "landscape"} == {"burst-1"}
    assert {r["group_id"] for r in ranked if r["category"] != "landscape"} == {"burst-2"}
    assert lapse_calls(ops, "still_timelapse") == [("still_timelapse", "burst-1.mp4")]
    by_source = {Path(r["source"]): Path(r["output"]) for r in ranked}
    assert ops.lapse == [([by_source[p] for p in big], [])]
    assert (result.run_dir / "timelapse" / "burst-1.jpg").is_file()
    assert not (result.run_dir / "timelapse" / "burst-2.mp4").exists()


def interval_records(paths, gaps):
    """Capture times from 06:30:00, each frame ``gaps[i]`` seconds after the one before."""
    from datetime import datetime, timedelta

    when, out = datetime(2026, 9, 22, 6, 30), {}
    for i, path in enumerate(paths):
        when += timedelta(seconds=gaps[i - 1] if i else 0)
        out[path] = CaptureRecord(
            path=path, datetime_original=when.strftime("%Y:%m:%d %H:%M:%S"), exposure_s=15.0
        )
    return out


def test_an_intervalometer_run_is_one_sequence_with_a_timelapse(tmp_path, root):
    names = [f"d{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    readings = {p: SubjectReading(SubjectCategory.LONG_EXPOSURE, 0.9) for p in seq}
    ops = Ops()
    ops.phash = lambda rgb: int(rgb[0, 0, 0]) % 2  # look-alikes would split it into two bursts
    hooks = make_hooks(ops, records=interval_records(seq, [18] * 29), readings=readings)
    result = run(src, root, hooks)
    ranked = rows(result.run_dir / "ranked.csv")
    assert {r["group_id"] for r in ranked} == {"sequence-1"}
    assert lapse_calls(ops, "still_timelapse") == [("still_timelapse", "sequence-1.mp4")]
    by_source = {Path(r["source"]): Path(r["output"]) for r in ranked}
    assert ops.lapse == [([by_source[p] for p in seq], [])]  # capture order
    page = (result.run_dir / "index.html").read_text()
    assert "<h4>Time-lapse ·" in page and "timelapse/sequence-1.mp4" in page


@pytest.mark.parametrize(
    "gaps",
    [
        [18] * 22,  # 23 frames: too short for a time-lapse
        [2, 50] * 15,  # hand-held: no steady cadence
        [18] * 14 + [90] + [18] * 14,  # a long pause splits it into two short runs
    ],
)
def test_no_sequence_without_a_long_steady_run(gaps):
    paths = [Path(f"/card/d{i:02d}.nef") for i in range(len(gaps) + 1)]
    ranked = [
        {"source": str(p), "category": "long-exposure", "rank": i + 1, "quality_score": "0.5"}
        for i, p in enumerate(paths)
    ]
    sg.tag_groups(ranked, interval_records(paths, gaps), Ops().ops(), {})
    assert not any(r.get("group_id", "").startswith("sequence") for r in ranked)


def test_detect_interval_runs_drops_a_lone_frame_before_the_run():
    paths = [Path(f"/card/d{i:02d}.nef") for i in range(31)]
    records = interval_records(paths, [170] + [18] * 29)  # JP's dawn run: one test shot first
    assert sg.detect_interval_runs(records.values()) == [paths[1:]]


def test_resume_skips_finished_timelapse_and_rebuilds_a_missing_one(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq])
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    first = run(src, root, hooks)
    ops.calls.clear()
    run(src, root, hooks, resume=first.run_dir)
    assert ops.calls == []
    (first.run_dir / "timelapse" / "night-1.mp4").unlink()
    run(src, root, hooks, resume=first.run_dir)
    assert [c[0] for c in ops.calls] == ["classify_night", "night_timelapse"]


def test_failed_timelapse_is_a_skipped_row_and_the_run_goes_on(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], fail=["night_timelapse"])
    result = run(
        src, root, make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    )
    skipped = rows(result.run_dir / "skipped.csv")
    assert [(r["path"], r["reason"]) for r in skipped] == [
        ("group:night-1", "timelapse failed: lapse boom")
    ]
    assert not list((result.run_dir / "timelapse").glob("night-1.*"))
    assert (result.run_dir / "stacks" / "night-1.tif").is_file()
    assert json.loads((result.run_dir / "run.json").read_text())["groups"]["failed"] == 1


def test_no_stacks_skips_night_timelapses(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq])
    run(
        src,
        root,
        make_hooks(ops, records=night_records(seq), readings=night_readings(seq)),
        ps.SortOptions(no_stacks=True),
    )
    assert ops.calls == []


def test_night_session_renders_one_star_trail_beside_its_timelapse(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], chunk=15, darks=seq[-4:])
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    result = run(src, root, hooks)
    assert lapse_calls(ops, "star_trails") == [("star_trails", "night-1.tif")]
    assert ops.trails == [(seq[:-4], seq[-4:])]  # lights in capture order; lens-cap darks apart
    assert lapse_calls(ops, "classify_night") == [("classify_night", 30)]  # one triage, shared
    assert (result.run_dir / "trails" / "night-1.tif").is_file()
    assert (result.run_dir / "trails" / "night-1.jpg").is_file()
    page = (result.run_dir / "index.html").read_text()
    stack, trail = page.index("stacks/night-1-session.jpg"), page.index("trails/night-1.jpg")
    assert stack < trail < page.index("timelapse/night-1.mp4")
    with Ledger(root / ps.LEDGER_NAME) as ledger:
        key = sg.group_key("trails", [str(p) for p in seq[:-4]])  # darks sit in their own folder
        assert ledger.get_score(key, sg.STACK_MARKER) == 1


def test_resume_skips_a_finished_trail_and_rebuilds_a_missing_one(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq])
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    first = run(src, root, hooks)
    ops.calls.clear()
    run(src, root, hooks, resume=first.run_dir)
    assert ops.calls == []
    (first.run_dir / "trails" / "night-1.jpg").unlink()
    run(src, root, hooks, resume=first.run_dir)
    assert [c[0] for c in ops.calls] == ["classify_night", "star_trails"]


def test_failed_trail_is_a_skipped_row_and_the_timelapse_still_lands(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], fail=["star_trails"])
    result = run(
        src, root, make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    )
    skipped = rows(result.run_dir / "skipped.csv")
    assert [(r["path"], r["reason"]) for r in skipped] == [
        ("group:night-1", "trails failed: trail boom")
    ]
    assert not list((result.run_dir / "trails").glob("night-1.*"))
    assert (result.run_dir / "timelapse" / "night-1.mp4").is_file()
    assert "trails/night-1.jpg" not in (result.run_dir / "index.html").read_text()


def test_default_trail_wrapper_uses_the_astro_trail_engine(tmp_path, monkeypatch):
    import gomoveshift.pipeline.photos.astro_calibrate as ac
    import gomoveshift.pipeline.photos.astro_orchestrator as ao
    import gomoveshift.pipeline.photos.astro_trails as at
    import gomoveshift.pipeline.photos.night_products as npx

    seen = {}
    monkeypatch.setattr(sg, "_megapixels", lambda p: 20.0)
    monkeypatch.setattr(at, "preflight_trail", lambda d, **k: seen.update(pre=(d, k)))

    def bad_darks(lights, darks):
        raise ac.AstroCalibrationError("mismatch")

    monkeypatch.setattr(ao, "build_validated_master_dark", bad_darks)
    monkeypatch.setattr(
        at,
        "render_trail",
        lambda lights, out, darks: (
            seen.update(trail=(list(lights), list(darks), out)) or (out, 2, ())
        ),
    )
    monkeypatch.setattr(
        npx, "render_master_preview", lambda master, out: seen.update(preview=(master, out)) or out
    )
    lights = [Path("l1"), Path("l2")]
    tif, jpg = tmp_path / "t.tif", tmp_path / "t.jpg"
    sg._star_trails(lights, [Path("d1")], tif, jpg)
    assert seen["pre"] == (tmp_path, {"megapixels": 20.0, "dark_count": 0})
    assert seen["trail"] == (lights, [], tif)  # unmatched darks dropped, never subtracted
    assert seen["preview"] == (tif, jpg)


def test_default_classify_night_splits_cap_on_frames_from_lights(monkeypatch):
    import gomoveshift.pipeline.photos.astro_orchestrator as ao
    import gomoveshift.pipeline.photos.astro_triage as at

    frame = SimpleNamespace(rgb=np.zeros((2, 2, 3), dtype=np.float32))
    monkeypatch.setattr(
        ao,
        "decode_with_he_fallback",
        lambda p, d, h: (None, "bad") if p.name == "bad.nef" else (frame, None),
    )
    monkeypatch.setattr(
        at,
        "measure_frame",
        lambda p, luma, **k: SimpleNamespace(
            exposure_class="cap_on" if p.name.startswith("dark") else "light"
        ),
    )
    paths = [Path(n) for n in ("a.nef", "dark1.nef", "bad.nef", "b.nef")]
    lights, darks = sg._classify_night(paths)
    assert [p.name for p in lights] == ["a.nef", "bad.nef", "b.nef"]
    assert [p.name for p in darks] == ["dark1.nef"]


def test_default_timelapse_wrappers_use_the_shared_engines(tmp_path, monkeypatch):
    import gomoveshift.pipeline.photos.astro_calibrate as ac
    import gomoveshift.pipeline.photos.astro_orchestrator as ao
    import gomoveshift.pipeline.photos.astro_timelapse as nt
    import gomoveshift.pipeline.timelapse as st

    seen = {}
    monkeypatch.setattr(sg, "_megapixels", lambda p: 20.0)
    monkeypatch.setattr(sg, "_poster", lambda v, j, used, fps: seen.setdefault("poster", (v, used)))
    monkeypatch.setattr(nt, "preflight_timelapse", lambda d, **k: seen.update(pre=k))

    def bad_darks(lights, darks):
        raise ac.AstroCalibrationError("mismatch")

    monkeypatch.setattr(ao, "build_validated_master_dark", bad_darks)
    monkeypatch.setattr(
        nt,
        "render_timelapse",
        lambda lights, out, darks, **k: (
            seen.update(night=(list(lights), list(darks), k)) or (out, 7, ())
        ),
    )
    lights = [Path("l1"), Path("l2")]
    sg._night_timelapse(lights, [Path("d1")], tmp_path / "n.mp4", tmp_path / "n.jpg")
    assert seen["night"][1] == [] and seen["pre"]["dark_count"] == 0  # unmatched darks dropped
    assert seen["night"][2] == {
        "fps": nt.TIMELAPSE_FPS,
        "scale_height": nt.TIMELAPSE_HEIGHT_DEFAULT,
    }
    assert seen["poster"] == (tmp_path / "n.mp4", 7)

    def stills(paths, out, fps, scale_height):
        seen["stills"] = [Path(p).name for p in paths]
        seen["targets"] = [Path(p).resolve().name for p in paths]
        return out, 3, ()

    monkeypatch.setattr(st, "render_stills_timelapse", stills)
    jpegs = [jpeg(tmp_path / name, i) for i, name in enumerate(["z.jpg", "a.jpg", "m.jpg"])]
    sg._still_timelapse(jpegs, tmp_path / "s.mp4", tmp_path / "s.jpg")
    assert seen["stills"] == ["000000.jpg", "000001.jpg", "000002.jpg"]  # capture order kept


def test_poster_extracts_the_middle_frame_and_names_a_failure(tmp_path, monkeypatch):
    import subprocess

    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        if cmd[-1].endswith("ok.jpg"):
            Path(cmd[-1]).write_bytes(b"jpg")
        return SimpleNamespace(returncode=0 if cmd[-1].endswith("ok.jpg") else 1, stderr=b"nope")

    monkeypatch.setattr(subprocess, "run", fake_run)
    sg._poster(tmp_path / "v.mp4", tmp_path / "ok.jpg", 48, 24)
    assert calls[0][calls[0].index("-ss") + 1] == "1.000"
    with pytest.raises(RuntimeError, match="poster frame failed"):
        sg._poster(tmp_path / "v.mp4", tmp_path / "bad.jpg", 48, 24)


def three_chunk_night(tmp_path, root, **ops_kw):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], chunk=10, **ops_kw)
    hooks = make_hooks(ops, records=night_records(seq), readings=night_readings(seq))
    return ops, run(src, root, hooks)


def test_session_of_three_chunks_shows_one_stack_and_keeps_the_chunks(tmp_path, root):
    ops, result = three_chunk_night(tmp_path, root)
    stacks = result.run_dir / "stacks"
    assert [c[1] for c in ops.calls if c[0] == "combine_session"] == ["night-1-session.tif"]
    assert ops.combined == ["night-1.linear", "night-2.linear", "night-3.linear"]
    assert (stacks / "night-1-session.tif").is_file() and (stacks / "night-1-session.jpg").is_file()
    for n in (1, 2, 3):  # intermediates stay on disk; their linear inputs are spent
        assert (stacks / f"night-{n}.tif").is_file()
        assert not (stacks / f"night-{n}.linear").exists()
    page = (result.run_dir / "index.html").read_text()
    assert "stacks/night-1-session.jpg" in page
    assert "stacks/night-2.jpg" not in page and "stacks/night-3.jpg" not in page
    assert "stacks/night-1.jpg" not in page


class Usable(Totals):
    """Totals whose ``technical.usable`` part is 0 for the named frames, 1 otherwise."""

    def __init__(self, starless):
        super().__init__({})
        self.starless = set(starless)

    def __call__(self, rgb, category, *, aesthetic, path, exposure_s, focal_mm):
        u = 0.0 if path.stem in self.starless else 1.0
        return ps.ShotScore(0.5, 0.5, aesthetic, 0.5, {"usable": u}, {})


def starless_night(tmp_path, root, starless):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], chunk=10)
    hooks = make_hooks(
        ops, Usable(starless), records=night_records(seq), readings=night_readings(seq)
    )
    return ops, run(src, root, hooks)


def test_chunk_with_no_usable_frame_is_skipped_before_stacking(tmp_path, root):
    ops, result = starless_night(tmp_path, root, [f"n{i:02d}" for i in range(10)])
    assert [c[1] for c in ops.calls if c[0] == "stack"] == ["night-2.tif", "night-3.tif"]
    assert ops.combined == ["night-2.linear", "night-3.linear"]
    reasons = {r["path"]: r["reason"] for r in rows(result.run_dir / "skipped.csv")}
    assert reasons["group:night-1"] == "bright sky: no stars measured in 10 frames"
    assert "stacks/night-1-session.jpg" in (result.run_dir / "index.html").read_text()


def test_chunk_with_one_usable_frame_still_stacks(tmp_path, root):
    ops, result = starless_night(tmp_path, root, [f"n{i:02d}" for i in range(9)])
    assert [c[1] for c in ops.calls if c[0] == "stack"] == [
        "night-1.tif",
        "night-2.tif",
        "night-3.tif",
    ]
    assert not any(r["path"] == "group:night-1" for r in rows(result.run_dir / "skipped.csv"))


def test_failed_session_combine_falls_back_to_the_best_chunk(tmp_path, root):
    names = [f"n{i:02d}" for i in range(30)]
    src = card(tmp_path, names)
    seq = stems(src, names)
    ops = Ops(sequences=[seq], chunk=10, fail=["combine"])
    totals = Totals({n: (0.9 if 10 <= i < 20 else 0.2) for i, n in enumerate(names)})
    hooks = make_hooks(ops, totals, records=night_records(seq), readings=night_readings(seq))
    result = run(src, root, hooks)
    page = (result.run_dir / "index.html").read_text()
    assert "stacks/night-2.jpg" in page
    assert "stacks/night-1.jpg" not in page and "stacks/night-3.jpg" not in page
    assert "session-1" not in page and "-session." not in page
    assert not (result.run_dir / "stacks" / "night-1-session.tif").exists()
    for n in (1, 2, 3):  # kept so a resume can retry the combine
        assert (result.run_dir / "stacks" / f"night-{n}.linear").is_dir()
    skipped = {r["path"]: r["reason"] for r in rows(result.run_dir / "skipped.csv")}
    # the session row is labelled as the session, never as its first chunk
    assert skipped["group:night-1-session"] == (
        "session combine failed (night-1, night-2, night-3): combine boom"
    )
    assert "group:night-1" not in skipped


def test_single_chunk_session_is_unchanged(tmp_path, root):
    src = card(tmp_path, list("abcde"))
    seq = stems(src, "abcde")
    ops = Ops(sequences=[seq])
    result = run(src, root, make_hooks(ops, readings=night_readings(seq)))
    assert [c for c in ops.calls if c[0] == "combine_session"] == []
    assert not list((result.run_dir / "stacks").glob("*-session*"))
    assert "stacks/night-1.jpg" in (result.run_dir / "index.html").read_text()


def test_resume_reuses_a_built_session_result(tmp_path, root):
    ops, first = three_chunk_night(tmp_path, root)
    with Ledger(root / ps.LEDGER_NAME) as ledger:
        members = [str(p) for p in stems(tmp_path / "card", [f"n{i:02d}" for i in range(30)])]
        assert ledger.get_score(sg.group_key("session", members), sg.STACK_MARKER) == 1
    again = Ops(sequences=ops.sequences, chunk=10)
    hooks = make_hooks(
        again,
        records=night_records(ops.sequences[0]),
        readings=night_readings(ops.sequences[0]),
    )
    run(tmp_path / "card", root, hooks, resume=first.run_dir)
    assert [c for c in again.calls if c[0] in ("stack", "combine_session")] == []
    assert "stacks/night-1-session.jpg" in (first.run_dir / "index.html").read_text()
