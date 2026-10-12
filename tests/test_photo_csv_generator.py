"""Tests for photo Metricool CSV generator."""

import csv

import pytest

from gomoveshift.pipeline.csv_generator import GOLDEN_COLUMNS, validate_csv
from gomoveshift.pipeline.photos.csv_generator import PhotoCSVGenerator, PhotoPost
from gomoveshift.pipeline.photos.scheduler import PhotoSlot
from gomoveshift.pipeline.uploader import UploadManifest


@pytest.fixture
def sample_posts():
    return [
        PhotoPost(
            photo_index=0,
            caption="Beautiful morning in Connemara, Ireland",
            tiktok_caption="Connemara morning vibes",
            hashtags="#ireland #connemara #travel",
            first_comment="#ireland #connemara #travel #roadtrip #scenicdrives",
            photo_filename="photo-001.jpg",
        ),
        PhotoPost(
            photo_index=1,
            caption="The cliffs were breathtaking today",
            tiktok_caption="Cliffs of Moher views",
            hashtags="#cliffsofmoher #ireland #travel",
            first_comment="#cliffsofmoher #ireland #travel #wildatlanticway",
            photo_filename="photo-002.jpg",
        ),
    ]


@pytest.fixture
def sample_slots():
    return [
        PhotoSlot(date="2026-03-01", time="10:00:00", day_of_week="Sunday"),
        PhotoSlot(date="2026-03-02", time="10:00:00", day_of_week="Monday"),
    ]


@pytest.fixture
def sample_manifest():
    manifest = UploadManifest(project_name="daily_photos")
    manifest.files = {
        "photo-001.jpg": "https://drive.google.com/uc?export=download&id=abc123",
        "photo-002.jpg": "https://drive.google.com/uc?export=download&id=def456",
    }
    return manifest


class TestPhotoCSVGenerator:
    def test_generates_csv(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            week_label="2026-03-01",
            validate=False,
        )
        assert csv_path.exists()
        assert csv_path.name == "photo_metricool_2026-03-01.csv"

    def test_correct_column_count(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert len(header) == 31
            assert header == GOLDEN_COLUMNS

    def test_youtube_disabled(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["YouTube"] == "FALSE"
                assert row["YouTube Video Title"] == ""
                assert row["YouTube Video Type"] == ""

    def test_post_types_are_post(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["Instagram Post Type"] == "POST"
                assert row["Facebook Post Type"] == "POST"

    def test_drive_links_mapped(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert "abc123" in rows[0]["Picture Url 1"]
            assert "def456" in rows[1]["Picture Url 1"]

    def test_validates_against_golden_reference(
        self, sample_posts, sample_slots, sample_manifest, tmp_path
    ):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        result = validate_csv(csv_path)
        assert result.valid, f"Validation errors: {result.errors}"
        assert result.row_count == 2

    def test_tiktok_title_truncated(self, sample_slots, sample_manifest, tmp_path):
        long_caption = "x" * 100
        posts = [
            PhotoPost(
                photo_index=0,
                caption="Test",
                tiktok_caption=long_caption,
                hashtags="",
                first_comment="",
                photo_filename="photo-001.jpg",
            ),
        ]
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=posts,
            slots=sample_slots[:1],
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            assert len(row["TikTok Title"]) <= 90


class TestPhotoCSVHardening:
    """Media-link form, cell escaping, BOM default and blank-slot gate."""

    def test_drive_view_link_converted_to_download_form(self, sample_posts, sample_slots, tmp_path):
        """A raw /view link makes TikTok/Meta pull Drive's HTML page, not the bytes."""
        manifest = UploadManifest(project_name="daily_photos")
        manifest.files = {
            "photo-001.jpg": "https://drive.google.com/file/d/abc123/view?usp=sharing",
            "photo-002.jpg": "https://drive.google.com/file/d/def456/view?usp=sharing",
        }
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["Picture Url 1"] == "https://drive.google.com/uc?export=download&id=abc123"
        assert rows[1]["Picture Url 1"] == "https://drive.google.com/uc?export=download&id=def456"

    def test_text_cells_escaped_against_formula_injection(
        self, sample_slots, sample_manifest, tmp_path
    ):
        posts = [
            PhotoPost(
                photo_index=0,
                caption="=cmd|'/c calc'!A1",
                tiktok_caption="+SUM(A1:A9)",
                hashtags="",
                first_comment="@evil",
                photo_filename="photo-001.jpg",
            ),
        ]
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=posts,
            slots=sample_slots[:1],
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
        assert row["Text"].startswith("'=")
        assert row["TikTok Title"].startswith("'+")
        assert row["First Comment Text"].startswith("'@")

    def test_no_bom_by_default(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        """The BOM breaks Metricool's column mapping — plain UTF-8 by default."""
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )
        assert not csv_path.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_excel_compat_writes_bom(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
            excel_compat=True,
        )
        assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_missing_slot_hard_fails(self, sample_posts, sample_manifest, tmp_path):
        """A blank Date/Time row is imported by Metricool as an immediate post."""
        gen = PhotoCSVGenerator()
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-SCHEDULE"):
            gen.generate(
                photo_posts=sample_posts,
                slots=[PhotoSlot(date="2026-03-01", time="10:00:00", day_of_week="Sunday")],
                manifest=sample_manifest,
                output_dir=str(tmp_path),
                validate=False,
            )
        assert not list(tmp_path.glob("*.csv"))

    def test_hashtags_reach_non_instagram_platforms(
        self, sample_posts, sample_slots, sample_manifest, tmp_path
    ):
        """First Comment Text is Instagram-only; TikTok/Twitter/Threads read Text."""
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
        assert "Beautiful morning in Connemara" in row["Text"]
        for tag in ("#ireland", "#connemara", "#travel"):
            assert tag in row["Text"]

    def test_hashtags_never_break_twitter_limit(self, sample_slots, sample_manifest, tmp_path):
        posts = [
            PhotoPost(
                photo_index=0,
                caption="word " * 60,
                tiktok_caption="t",
                hashtags="#ireland #connemara #travel",
                first_comment="",
                photo_filename="photo-001.jpg",
            ),
        ]
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=posts,
            slots=sample_slots[:1],
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
        assert len(row["Text"]) <= 280

    def test_golden_schema_preserved(self, sample_posts, sample_slots, sample_manifest, tmp_path):
        gen = PhotoCSVGenerator()
        csv_path = gen.generate(
            photo_posts=sample_posts,
            slots=sample_slots,
            manifest=sample_manifest,
            output_dir=str(tmp_path),
            validate=False,
        )

        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == GOLDEN_COLUMNS
            assert len(reader.fieldnames) == 31
            for row in reader:
                assert row["YouTube Video Category"] == ""


class TestPhotoPost:
    def test_to_dict(self, sample_posts):
        d = sample_posts[0].to_dict()
        assert d["photo_index"] == 0
        assert d["caption"] == "Beautiful morning in Connemara, Ireland"
        assert d["photo_filename"] == "photo-001.jpg"
