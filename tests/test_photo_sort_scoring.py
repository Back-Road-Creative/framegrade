"""photos sort: score each developed shot, order every category by quality."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline import orchestrator_lock
from gomoveshift.pipeline.photos import sort as ps
from gomoveshift.pipeline.photos.aesthetic import AestheticModelUnavailable
from gomoveshift.pipeline.photos.capture_meta import CaptureRecord
from gomoveshift.pipeline.photos.dedupe import Ledger, sha256_file
from gomoveshift.pipeline.photos.subject import SubjectCategory, SubjectReading

TODAY = "2026-11-09"


def jpeg(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (16, 24, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def develop_real_jpeg(task):
    """Fake develop worker: a tiny but decodable JPEG so the parent can score it."""
    task.out.write_bytes(task.source.read_bytes())


def shot_score(total, *, aesthetic=None, tparts=None, cparts=None):
    return ps.ShotScore(total, total / 2, aesthetic, total, tparts or {}, cparts or {})


class StubScore:
    """Fixed total per source stem; optionally raises for chosen stems."""

    def __init__(self, totals, fail=(), **kw):
        self.totals, self.fail, self.kw, self.calls = totals, set(fail), kw, []

    def __call__(self, rgb, category, *, aesthetic, path, exposure_s, focal_mm):
        self.calls.append((path.stem, category, aesthetic, exposure_s, focal_mm))
        if path.stem in self.fail:
            raise RuntimeError("score boom")
        return shot_score(self.totals[path.stem], aesthetic=aesthetic, **self.kw)


class StubAesthetic:
    def __init__(self, value=0.7, fail_on=None):
        self.value, self.fail_on, self.calls = value, fail_on, 0

    def score_unit(self, rgb):
        self.calls += 1
        if self.fail_on == self.calls:
            raise RuntimeError("onnx boom")
        return self.value


def make_hooks(score=None, factory=None, records=None, readings=None, **over):
    def read_records(paths):
        recs = {p: CaptureRecord(path=p) for p in paths}
        recs.update(records or {})
        return recs, []

    def recognise(paths, optics=None):
        out = {p: ps.UNKNOWN_SUBJECT for p in paths}
        out.update(readings or {})
        return out

    return ps.SortHooks(
        **{
            "read_records": read_records,
            "recognise": recognise,
            "corrections": lambda paths: [(None, 0)] * len(paths),
            "develop": develop_real_jpeg,
            "executor_factory": lambda workers: ThreadPoolExecutor(max_workers=workers),
            "score": score,
            "aesthetic_factory": factory,
            **over,
        }
    )


def rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(folder, root, hooks, options=None, said=None):
    return ps.run_sort(
        folder,
        root,
        options or ps.SortOptions(),
        hooks,
        today=TODAY,
        say=(said.append if said is not None else (lambda *_: None)),
    )


def meta_of(result):
    return json.loads((result.run_dir / "run.json").read_text())


@pytest.fixture
def folder(tmp_path):
    src = tmp_path / "card" / "trip"
    for i, stem in enumerate("abc", 1):
        jpeg(src / f"{stem}.jpg", i)
    return src


@pytest.fixture
def root(tmp_path):
    return tmp_path / "out"


def test_category_order_follows_total_and_filenames_carry_rank(folder, root):
    score = StubScore({"a": 0.2, "b": 0.9, "c": 0.5})
    result = run(folder, root, make_hooks(score))
    names = sorted(p.name for p in (result.run_dir / "other").iterdir())
    assert names == ["001-b.jpg", "002-c.jpg", "003-a.jpg"]
    ranked = rows(result.run_dir / "ranked.csv")
    assert [(r["rank"], Path(r["source"]).stem, r["quality_score"]) for r in ranked] == [
        ("1", "b", "0.9000"),
        ("2", "c", "0.5000"),
        ("3", "a", "0.2000"),
    ]
    assert all(Path(r["output"]).is_file() for r in ranked)
    assert ranked[0]["technical"] == "0.9000" and ranked[0]["category_score"] == "0.4500"


def test_tie_is_deterministic_by_source_path(folder, root):
    score = StubScore({"a": 0.5, "b": 0.5, "c": 0.5})
    result = run(folder, root, make_hooks(score))
    assert sorted(p.name for p in (result.run_dir / "other").iterdir()) == [
        "001-a.jpg",
        "002-b.jpg",
        "003-c.jpg",
    ]


def test_ranked_csv_prefixes_colliding_astro_parts(folder, root):
    readings = {folder / "a.jpg": SubjectReading(SubjectCategory.ASTRO, 0.9)}
    score = StubScore(
        {"a": 0.4, "b": 0.3, "c": 0.2}, tparts={"stars": 0.1}, cparts={"stars": 0.8, "sky": 0.5}
    )
    result = run(folder, root, make_hooks(score, readings=readings))
    ranked = rows(result.run_dir / "ranked.csv")
    first = ranked[0]
    assert first["technical.stars"] == "0.1000" and first["category.stars"] == "0.8000"
    assert first["category.sky"] == "0.5000"
    assert "stars" not in first


def test_part_columns_are_a_union_blank_where_absent(folder, root):
    class Mixed(StubScore):
        def __call__(self, rgb, category, *, aesthetic, path, exposure_s, focal_mm):
            self.kw = {"cparts": {"only_a": 1.0}} if path.stem == "a" else {}
            return super().__call__(
                rgb,
                category,
                aesthetic=aesthetic,
                path=path,
                exposure_s=exposure_s,
                focal_mm=focal_mm,
            )

    result = run(folder, root, make_hooks(Mixed({"a": 0.9, "b": 0.5, "c": 0.4})))
    ranked = {Path(r["source"]).stem: r for r in rows(result.run_dir / "ranked.csv")}
    assert ranked["a"]["category.only_a"] == "1.0000" and ranked["b"]["category.only_a"] == ""


def test_missing_weights_run_completes_and_warns_with_the_command(
    folder, root, tmp_path, monkeypatch
):
    monkeypatch.setenv("FRAMEGRADE_AESTHETIC_DIR", str(tmp_path / "no-models"))
    score = StubScore({"a": 0.2, "b": 0.9, "c": 0.5})
    said = []
    hooks = make_hooks(score, factory=ps.make_aesthetic_scorer)
    result = run(folder, root, hooks, said=said)
    ranked = rows(result.run_dir / "ranked.csv")
    assert [r["aesthetic_score"] for r in ranked] == ["", "", ""]
    assert {c[2] for c in score.calls} == {None}
    warnings = meta_of(result)["warnings"]
    assert len(warnings) == 1 and "gomoveshift photos fetch-models" in warnings[0]
    assert sum("fetch-models" in line for line in said) == 1


def test_aesthetic_load_error_other_than_missing_weights_is_a_warning(folder, root):
    def factory(*, use_gpu):
        raise RuntimeError("no onnxruntime")

    result = run(folder, root, make_hooks(StubScore({"a": 1, "b": 2, "c": 3}), factory=factory))
    assert "no onnxruntime" in meta_of(result)["warnings"][0]
    assert len(rows(result.run_dir / "ranked.csv")) == 3


def test_one_aesthetic_failure_keeps_the_shot_and_is_counted(folder, root):
    scorer = StubAesthetic(0.7, fail_on=2)
    result = run(
        folder,
        root,
        make_hooks(StubScore({"a": 0.3, "b": 0.2, "c": 0.1}), factory=lambda *, use_gpu: scorer),
    )
    ranked = rows(result.run_dir / "ranked.csv")
    assert len(ranked) == 3 and len(list((result.run_dir / "other").glob("*.jpg"))) == 3
    assert sorted(r["aesthetic_score"] for r in ranked) == ["", "0.7000", "0.7000"]
    meta = meta_of(result)
    assert meta["scoring"]["aesthetic_failed"] == 1 and meta["scoring"]["aesthetic_scored"] == 2
    assert any("aesthetic score failed for 1" in w for w in meta["warnings"])


def test_score_failure_and_unreadable_output_rank_last_in_capture_order(folder, root):
    records = {
        folder / s: CaptureRecord(path=folder / s, datetime_original=f"2026:01:01 10:00:0{i}")
        for i, s in enumerate(("a.jpg", "b.jpg", "c.jpg"), 1)
    }

    def develop(task):
        if task.source.stem == "c":
            task.out.write_bytes(b"not a jpeg")
        else:
            develop_real_jpeg(task)

    score = StubScore({"a": 0.9, "b": 0.8}, fail={"b"})
    result = run(folder, root, make_hooks(score, records=records, develop=develop))
    ranked = rows(result.run_dir / "ranked.csv")
    assert [(Path(r["source"]).stem, r["quality_score"]) for r in ranked] == [
        ("a", "0.9000"),
        ("b", ""),
        ("c", ""),
    ]
    assert meta_of(result)["scoring"]["score_failed"] == 2
    assert sorted(p.name for p in (result.run_dir / "other").iterdir()) == [
        "001-a.jpg",
        "002-b.jpg",
        "003-c.jpg",
    ]


def test_cache_hit_skips_the_model_and_miss_writes_the_ledger(folder, root):
    first = StubAesthetic(0.6)
    run(folder, root, make_hooks(StubScore({"a": 1, "b": 2, "c": 3}), lambda *, use_gpu: first))
    assert first.calls == 3
    with Ledger(root / ps.LEDGER_NAME) as ledger:
        for stem in "abc":
            sha = sha256_file(folder / f"{stem}.jpg")
            assert ledger.get_score(sha, ps.AESTHETIC_SCORE_NAME) == pytest.approx(0.6)
    spy = StubAesthetic(0.1)
    score = StubScore({"a": 1, "b": 2, "c": 3})
    result = run(
        folder,
        root,
        make_hooks(score, lambda *, use_gpu: spy),
        ps.SortOptions(reprocess=True),
    )
    assert spy.calls == 0
    assert [c[2] for c in score.calls] == [pytest.approx(0.6)] * 3
    assert meta_of(result)["scoring"]["aesthetic_cached"] == 3


def test_sort_has_no_gpu_option_or_lock_hooks():
    assert "gpu" not in ps.SortOptions.__dataclass_fields__
    assert not {"gpu_lock", "gpu_release"} & set(ps.SortHooks.__dataclass_fields__)
    assert not hasattr(ps, "acquire_gpu_lock")


def test_sort_never_takes_the_gpu_lock(folder, root, monkeypatch):
    """The GoMoveShift drives own the GPU lock; a photo sort must never contend for it."""
    taken = Mock(side_effect=AssertionError("photos sort took the GPU lock"))
    monkeypatch.setattr(orchestrator_lock, "acquire_gpu_lock", taken)
    seen = []
    hooks = make_hooks(
        StubScore({"a": 1, "b": 2, "c": 3}),
        lambda *, use_gpu: seen.append(use_gpu) or StubAesthetic(),
    )
    run(folder, root, hooks)
    taken.assert_not_called()
    assert seen == [False]


def test_exposure_and_focal_come_from_the_capture_record(folder, root):
    records = {
        folder / "a.jpg": CaptureRecord(
            path=folder / "a.jpg", exposure_s=20.0, focal_length_mm=14.0
        )
    }
    score = StubScore({"a": 1, "b": 2, "c": 3})
    run(folder, root, make_hooks(score, records=records))
    by_stem = {c[0]: c for c in score.calls}
    assert by_stem["a"][3:] == (20.0, 14.0) and by_stem["b"][3:] == (None, None)
    assert by_stem["a"][1] == "unknown"


def test_renumbering_twelve_files_never_loses_or_duplicates(tmp_path):
    files = []
    for i in range(1, 13):
        f = tmp_path / f"{i:03d}-s{i}.jpg"
        f.write_bytes(f"content-{i}".encode())
        files.append(f)
    renames = [(f, tmp_path / f"{13 - i:03d}-s{i}.jpg") for i, f in enumerate(files, 1)]
    ps._renumber(renames)
    got = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert len(got) == 12
    assert got == {f"{13 - i:03d}-s{i}.jpg": f"content-{i}".encode() for i in range(1, 13)}


def test_twelve_shots_end_to_end_reversed_by_score(tmp_path, root):
    src = tmp_path / "card" / "many"
    for i in range(1, 13):
        jpeg(src / f"p{i:02d}.jpg", i)
    score = StubScore({f"p{i:02d}": i / 100 for i in range(1, 13)})
    result = run(src, root, make_hooks(score))
    names = sorted(p.name for p in (result.run_dir / "other").iterdir())
    assert names == [f"{n:03d}-p{13 - n:02d}.jpg" for n in range(1, 13)]
    assert len(rows(result.run_dir / "ranked.csv")) == 12


def test_a_develop_failure_gap_is_closed_by_the_renumber(folder, root):
    def develop(task):
        if task.source.stem == "b":
            raise RuntimeError("decode failed")
        develop_real_jpeg(task)

    result = run(folder, root, make_hooks(StubScore({"a": 0.1, "c": 0.9}), develop=develop))
    assert sorted(p.name for p in (result.run_dir / "other").iterdir()) == [
        "001-c.jpg",
        "002-a.jpg",
    ]


def test_index_html_sections_in_rank_order_with_total(folder, root):
    readings = {folder / "a.jpg": SubjectReading(SubjectCategory.ASTRO, 0.9)}
    score = StubScore({"a": 0.5, "b": 0.2, "c": 0.9})
    result = run(folder, root, make_hooks(score, readings=readings))
    page = (result.run_dir / "index.html").read_text()
    assert page.index("001-c.jpg") < page.index("002-b.jpg")
    assert "quality 90" in page and "quality 20" in page
    assert "<h3>Night sky" in page and "<h3>Unsorted" in page


def test_summary_lists_top_five_per_category(tmp_path, root):
    src = tmp_path / "card" / "seven"
    for i in range(1, 8):
        jpeg(src / f"p{i}.jpg", i)
    said = []
    score = StubScore({f"p{i}": i / 10 for i in range(1, 8)})
    run(src, root, make_hooks(score), said=said)
    top = [line for line in said if line.startswith("    ")]
    assert len(top) == 5 and top[0].strip().startswith("1. 001-p7.jpg  0.70")
    assert any("top other" in line for line in said)


def test_without_a_score_hook_nothing_is_scored_or_renamed(folder, root):
    result = run(folder, root, make_hooks())
    assert sorted(p.name for p in (result.run_dir / "other").iterdir()) == [
        "001-a.jpg",
        "002-b.jpg",
        "003-c.jpg",
    ]
    assert "scoring" not in meta_of(result)
    assert {r["quality_score"] for r in rows(result.run_dir / "ranked.csv")} == {""}


def test_dry_run_estimate_gains_the_aesthetic_term(folder, root):
    said = []
    result = ps.run_sort(
        folder,
        root,
        ps.SortOptions(dry_run=True, workers=3),
        make_hooks(StubScore({}), lambda *, use_gpu: StubAesthetic()),
        today=TODAY,
        say=said.append,
    )
    expected = 3 * ps.DEVELOP_S_PER_PHOTO_ESTIMATE / 3 + 3 * ps.AESTHETIC_S_PER_PHOTO_ESTIMATE
    assert result.estimate_s == pytest.approx(expected)
    assert any("aesthetic" in line for line in said)


def test_real_scoring_on_real_jpegs_orders_and_splits_astro_parts(folder, root):
    readings = {folder / "a.jpg": SubjectReading(SubjectCategory.ASTRO, 0.9)}
    hooks = make_hooks(ps.score_shot, readings=readings)
    result = run(folder, root, hooks)
    ranked = rows(result.run_dir / "ranked.csv")
    astro = next(r for r in ranked if r["category"] == "night-sky")
    assert astro["technical.stars"] != "" and astro["category.stars"] != ""
    other = [r for r in ranked if r["category"] == "other"]
    totals = [float(r["quality_score"]) for r in other]
    assert totals == sorted(totals, reverse=True) and len(other) == 2


def test_score_shot_clamps_aesthetic_and_none_passes_through():
    rgb = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert ps.score_shot(rgb, "landscape", aesthetic=None).aesthetic is None
    assert ps.score_shot(rgb, "landscape", aesthetic=3.0).aesthetic == 1.0


def test_scoring_hooks_wires_the_real_scorers():
    hooks = ps.scoring_hooks()
    assert hooks.score is ps.score_shot and hooks.aesthetic_factory is ps.make_aesthetic_scorer


def test_make_aesthetic_scorer_builds_a_two_thread_cpu_scorer(monkeypatch):
    made = {}

    class Fake:
        def __init__(self, **kw):
            made.update(kw)

    monkeypatch.setattr("gomoveshift.pipeline.photos.aesthetic.AestheticScorer", Fake)
    ps.make_aesthetic_scorer(use_gpu=False)
    assert made == {"use_gpu": False, "threads": 2}


def test_aesthetic_unavailable_error_names_the_cli_command():
    from gomoveshift.pipeline.photos import aesthetic

    with pytest.raises(AestheticModelUnavailable, match="gomoveshift photos fetch-models"):
        aesthetic.ensure_models()
