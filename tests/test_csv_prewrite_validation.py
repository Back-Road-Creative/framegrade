"""Golden-reference validation must run BEFORE the Metricool CSV is written.

Validating the file after ``open(csv_path, "w")`` is a late gate: the malformed
CSV already exists on disk when the ``ValueError`` is raised, so every consumer
that tests for *existence* rather than *validity* (``stages/csv_gen.py`` reports
failure but leaves the file behind) sees a CSV that "passed".

The rows are fully built in memory before the write, so the same rules can run
one step earlier and make the bad artifact impossible by construction.
"""

import csv

import pytest

from gomoveshift.pipeline.csv_generator import GOLDEN_COLUMNS, CSVGenerator, validate_csv
from gomoveshift.pipeline.scheduler import PipelineScheduler
from gomoveshift.pipeline.social_generator import SocialGenerator
from gomoveshift.pipeline.uploader import UploadManifest

NUM_SHORTS = 3


@pytest.fixture
def parts():
    """Posts + schedule + manifest for a small, fully linked shorts run."""
    posts = SocialGenerator(location="County Kerry, Ireland", series_part=1).generate_all(
        num_shorts=NUM_SHORTS
    )
    schedule = PipelineScheduler().generate_schedule(
        publish_date="next_wed",
        publish_day="wed",
        num_shorts=NUM_SHORTS,
        project_name="test",
        location="Kerry",
    )
    manifest = UploadManifest(
        project_name="test",
        files={
            f"short_{i:02d}.mp4": f"https://drive.google.com/uc?export=download&id=fake{i}"
            for i in range(1, NUM_SHORTS + 1)
        },
    )
    return posts, schedule, manifest


def _corrupt_dates(monkeypatch):
    """Make every row carry a malformed (but non-blank) Date.

    Non-blank keeps the upstream schedule gate satisfied so the failure lands on
    the golden-reference date-format rule, not on an earlier assertion.
    """
    original = CSVGenerator._build_row

    def bad_row(self, post, slot, drive_link, *args, **kwargs):
        row = original(self, post, slot, drive_link, *args, **kwargs)
        row["Date"] = "09/10/2025"
        return row

    monkeypatch.setattr(CSVGenerator, "_build_row", bad_row)


def test_invalid_rows_never_reach_disk(tmp_path, parts, monkeypatch):
    """RED before the fix: the CSV is written, THEN validated."""
    posts, schedule, manifest = parts
    _corrupt_dates(monkeypatch)
    csv_path = None
    with pytest.raises(ValueError, match="golden reference validation"):
        csv_path = CSVGenerator().generate(
            project_name="test",
            social_posts=posts,
            schedule=schedule,
            manifest=manifest,
            output_dir=str(tmp_path),
            validate=True,
        )

    assert csv_path is None
    assert list(tmp_path.iterdir()) == [], (
        "a CSV that fails golden validation must never be created on disk"
    )


def test_valid_rows_still_write_identical_bytes(tmp_path, parts):
    """Happy path is byte-for-byte what the unvalidated writer produces."""
    posts, schedule, manifest = parts
    gen = CSVGenerator()

    kwargs = dict(project_name="test", social_posts=posts, schedule=schedule, manifest=manifest)
    unvalidated = gen.generate(output_dir=str(tmp_path / "off"), validate=False, **kwargs)
    validated = gen.generate(output_dir=str(tmp_path / "on"), validate=True, **kwargs)

    assert validated.read_bytes() == unvalidated.read_bytes()
    assert validate_csv(validated).valid


def test_prewrite_and_on_disk_validators_agree(tmp_path, parts, monkeypatch):
    """One rule set, two entry points — no second copy of the rules to drift.

    Whatever ``validate_csv`` reports for the written file, the in-memory
    validator must report for the rows that produced it: same verdict, same
    errors, same counts. Checked for valid rows and for malformed ones.
    """
    from gomoveshift.pipeline.csv_generator import validate_rows

    posts, schedule, manifest = parts
    gen = CSVGenerator()
    slots = {slot.short_index: slot for slot in schedule.slots}

    for corrupt in (False, True):
        if corrupt:
            _corrupt_dates(monkeypatch)
        rows = [
            gen._build_row(
                post, slots[post.short_index], gen._match_short_to_file(post.short_index, manifest)
            )
            for post in posts
        ]

        path = tmp_path / f"corrupt_{corrupt}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=GOLDEN_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        in_memory = validate_rows(rows)
        on_disk = validate_csv(path)

        assert in_memory.valid is on_disk.valid is (not corrupt)
        assert in_memory.errors == on_disk.errors
        assert in_memory.row_count == on_disk.row_count == NUM_SHORTS
        assert in_memory.column_count == on_disk.column_count
