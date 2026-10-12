"""Tests for photo resizer."""

import io
from pathlib import Path

import pytest
from PIL import Image, ImageCms

from gomoveshift.pipeline.photos.resizer import PhotoResizer, _assert_no_gps

GPS_IFD = 0x8825
ORIENTATION = 0x0112

XMP_GPS = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
    b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/" '
    b'exif:GPSLatitude="40,26.767N" exif:GPSLongitude="74,0.166W"/>'
    b"</rdf:RDF></x:xmpmeta>"
)


def _exif(orientation=None, gps=False):
    exif = Image.Exif()
    if orientation is not None:
        exif[ORIENTATION] = orientation
    if gps:
        exif[GPS_IFD] = {1: "N", 2: (40.0, 0.0, 0.0), 3: "W", 4: (74.0, 0.0, 0.0)}
    return exif


class TestPhotoResizer:
    @pytest.fixture
    def resizer(self, tmp_path):
        return PhotoResizer(output_base=tmp_path / "resized")

    @pytest.fixture
    def sample_photos(self, tmp_path):
        """Create sample JPEG files at 1920x1440."""
        photos_dir = tmp_path / "source"
        photos_dir.mkdir()
        paths = []
        for i in range(3):
            img = Image.new("RGB", (1920, 1440), color=(100 + i * 50, 80, 60))
            path = photos_dir / f"photo-{i:03d}.jpg"
            img.save(path, "JPEG", quality=95)
            paths.append(str(path))
        return paths

    def test_resize_batch_creates_output_dir(self, resizer, sample_photos):
        result = resizer.resize_batch(sample_photos, "2026-02-23")
        assert (resizer.output_base / "2026-02-23").is_dir()
        assert len(result) == 3

    def test_resize_scales_to_target_width(self, resizer, sample_photos):
        result = resizer.resize_batch(sample_photos, "2026-02-23")
        for path in result:
            with Image.open(path) as img:
                assert img.size[0] == 1080
                # 1920x1440 -> 1080x810 (maintains aspect ratio)
                assert img.size[1] == 810

    def test_resize_reduces_file_size(self, resizer, sample_photos):
        result = resizer.resize_batch(sample_photos, "2026-02-23")
        for orig, resized in zip(sample_photos, result):
            orig_size = Path(orig).stat().st_size
            resized_size = resized.stat().st_size
            assert resized_size < orig_size

    def test_resize_preserves_filenames(self, resizer, sample_photos):
        result = resizer.resize_batch(sample_photos, "2026-02-23")
        for orig, resized in zip(sample_photos, result):
            assert resized.name == Path(orig).name

    def test_small_image_not_upscaled(self, resizer, tmp_path):
        """Images already at or below target width are quality-optimized only."""
        photos_dir = tmp_path / "small"
        photos_dir.mkdir()
        img = Image.new("RGB", (800, 600), color=(200, 100, 50))
        path = photos_dir / "small.jpg"
        img.save(path, "JPEG")

        result = resizer.resize_batch([str(path)], "2026-02-23")
        with Image.open(result[0]) as out_img:
            assert out_img.size == (800, 600)

    def test_custom_target_width(self, tmp_path):
        resizer = PhotoResizer(target_width=720, output_base=tmp_path / "resized")
        photos_dir = tmp_path / "source"
        photos_dir.mkdir()
        img = Image.new("RGB", (1920, 1440))
        path = photos_dir / "photo.jpg"
        img.save(path, "JPEG")

        result = resizer.resize_batch([str(path)], "2026-02-23")
        with Image.open(result[0]) as out_img:
            assert out_img.size[0] == 720
            assert out_img.size[1] == 540  # 1440 * 720/1920

    def test_custom_quality(self, tmp_path):
        low_q = PhotoResizer(quality=50, output_base=tmp_path / "low")
        high_q = PhotoResizer(quality=95, output_base=tmp_path / "high")

        photos_dir = tmp_path / "source"
        photos_dir.mkdir()
        img = Image.new("RGB", (1920, 1440), color=(128, 64, 192))
        path = photos_dir / "photo.jpg"
        img.save(path, "JPEG")

        low_result = low_q.resize_batch([str(path)], "2026-02-23")
        high_result = high_q.resize_batch([str(path)], "2026-02-23")

        assert low_result[0].stat().st_size < high_result[0].stat().st_size

    def test_empty_batch(self, resizer):
        result = resizer.resize_batch([], "2026-02-23")
        assert result == []

    def test_missing_file_raises(self, resizer, tmp_path):
        with pytest.raises(Exception):
            resizer.resize_batch([str(tmp_path / "nonexistent.jpg")], "2026-02-23")

    def test_week_start_creates_subdirectory(self, resizer, tmp_path):
        photos_dir = tmp_path / "source"
        photos_dir.mkdir()
        img = Image.new("RGB", (1920, 1440))
        path = photos_dir / "photo.jpg"
        img.save(path, "JPEG")

        resizer.resize_batch([str(path)], "2026-03-02")
        assert (resizer.output_base / "2026-03-02").is_dir()

    def test_resize_with_journey_slugs(self, resizer, tmp_path):
        """Two same-named photos from different journeys get distinct output names."""
        photos_dir = tmp_path / "source"
        photos_dir.mkdir()
        paths = []
        for slug in ["example-2018", "example-2025"]:
            img = Image.new("RGB", (1920, 1440), color=(100, 80, 60))
            path = photos_dir / f"{slug}_photo-005.jpg"
            img.save(path, "JPEG")
            paths.append(str(path))

        # Both source files have different names, but simulate same bare name scenario
        # by using journey_slugs to prefix output
        photo_a = photos_dir / "photo-005a.jpg"
        photo_b = photos_dir / "photo-005b.jpg"
        Image.new("RGB", (1920, 1440)).save(photo_a, "JPEG")
        Image.new("RGB", (1920, 1440)).save(photo_b, "JPEG")

        result = resizer.resize_batch(
            [str(photo_a), str(photo_b)],
            "2026-03-18",
            journey_slugs=["example-2018", "example-2025"],
        )

        assert result[0].name == "example-2018_photo-005a.jpg"
        assert result[1].name == "example-2025_photo-005b.jpg"
        assert result[0] != result[1]

    def test_journey_slugs_length_mismatch_raises(self, resizer, tmp_path):
        """ValueError raised when journey_slugs length != photo_paths length."""
        photos_dir = tmp_path / "source"
        photos_dir.mkdir()
        paths = []
        for i in range(3):
            img = Image.new("RGB", (1920, 1440))
            path = photos_dir / f"photo-{i:03d}.jpg"
            img.save(path, "JPEG")
            paths.append(str(path))

        with pytest.raises(ValueError, match="journey_slugs length"):
            resizer.resize_batch(
                paths,
                "2026-03-18",
                journey_slugs=["slug-a", "slug-b"],  # 2 slugs for 3 photos
            )

    def test_resize_without_journey_slugs_backward_compat(self, resizer, tmp_path):
        """None journey_slugs preserves old behavior (bare filename)."""
        photos_dir = tmp_path / "source"
        photos_dir.mkdir()
        img = Image.new("RGB", (1920, 1440))
        path = photos_dir / "photo-001.jpg"
        img.save(path, "JPEG")

        result = resizer.resize_batch([str(path)], "2026-03-18", journey_slugs=None)
        assert result[0].name == "photo-001.jpg"

    def test_exif_orientation_applied(self, resizer, tmp_path):
        """Orientation-tagged portrait photos are rotated upright, not shipped sideways."""
        path = tmp_path / "portrait.jpg"
        Image.new("RGB", (2000, 1000), (10, 120, 200)).save(path, "JPEG", exif=_exif(orientation=6))

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert out.size == (1000, 2000)
            assert out.getexif().get(ORIENTATION) is None

    def test_rgba_source_does_not_crash_batch(self, resizer, tmp_path):
        """An RGBA PNG must not raise 'cannot write mode RGBA as JPEG'."""
        path = tmp_path / "alpha.png"
        Image.new("RGBA", (1920, 1440), (200, 100, 50, 128)).save(path, "PNG")

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert out.mode == "RGB"
            assert out.size[0] == 1080

    def test_palette_source_does_not_crash_batch(self, resizer, tmp_path):
        """A palette ('P' mode) PNG must convert rather than kill the publish batch."""
        path = tmp_path / "palette.png"
        Image.new("RGB", (1920, 1440), (30, 90, 150)).convert("P").save(path, "PNG")

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert out.mode == "RGB"

    def test_output_extension_forced_to_jpg(self, resizer, tmp_path):
        """Bytes are always JPEG, so the name must not claim .png (MIME mismatch on upload)."""
        path = tmp_path / "shot.png"
        Image.new("RGB", (1920, 1440), (10, 20, 30)).save(path, "PNG")

        result = resizer.resize_batch([str(path)], "2026-04-06")
        assert result[0].name == "shot.jpg"
        with Image.open(result[0]) as out:
            assert out.format == "JPEG"

    def test_gps_exif_stripped(self, resizer, tmp_path):
        """Location metadata must never reach Drive."""
        path = tmp_path / "geo.jpg"
        Image.new("RGB", (1920, 1440), (60, 60, 60)).save(path, "JPEG", exif=_exif(gps=True))

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert GPS_IFD not in out.getexif()

    def test_gps_exif_stripped_on_no_resize_path(self, resizer, tmp_path):
        """The already-small path must strip GPS too — never a blind copy."""
        path = tmp_path / "small-geo.jpg"
        Image.new("RGB", (800, 600), (60, 60, 60)).save(path, "JPEG", exif=_exif(gps=True))

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert GPS_IFD not in out.getexif()
            assert out.size == (800, 600)

    def test_icc_profile_converted_to_srgb(self, resizer, tmp_path):
        """Tagged sources ship a colour-managed sRGB output, not a stripped/shifted one."""
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        path = tmp_path / "tagged.jpg"
        Image.new("RGB", (1920, 1440), (200, 120, 60)).save(
            path, "JPEG", icc_profile=profile, quality=95
        )

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            out_profile = out.info.get("icc_profile")
            assert out_profile, "resized photo dropped its ICC profile"
            desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(out_profile)))
            assert "sRGB" in desc

    def test_unreadable_icc_profile_does_not_crash_batch(self, resizer, tmp_path):
        """A corrupt profile falls back to a plain save instead of failing the batch."""
        path = tmp_path / "bad-icc.jpg"
        Image.new("RGB", (1920, 1440), (10, 20, 30)).save(
            path, "JPEG", icc_profile=b"not-a-real-icc-profile"
        )

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert out.size[0] == 1080

    def test_failed_icc_conversion_is_saved_untagged(self, resizer, tmp_path, caplog):
        """Pixels left in the source space must not be labelled sRGB."""
        path = tmp_path / "bad-icc-tag.jpg"
        Image.new("RGB", (1920, 1440), (10, 20, 30)).save(
            path, "JPEG", icc_profile=b"not-a-real-icc-profile"
        )

        with caplog.at_level("WARNING"):
            result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert "icc_profile" not in out.info
        assert "Unusable ICC profile" in caplog.text

    def test_clean_small_photo_is_copied_not_reencoded(self, resizer, tmp_path):
        """No resize needed and nothing to strip — copy the bytes, don't re-encode at q85."""
        path = tmp_path / "clean.jpg"
        Image.new("RGB", (800, 600), (200, 100, 50)).save(path, "JPEG", quality=95)

        result = resizer.resize_batch([str(path)], "2026-04-06")
        assert result[0].read_bytes() == path.read_bytes()

    def test_xmp_gps_stripped_on_copy_path(self, resizer, tmp_path):
        """An EXIF-clean small JPEG whose XMP packet carries GPS must not ship verbatim."""
        path = tmp_path / "xmp-geo.jpg"
        Image.new("RGB", (800, 600), (60, 60, 60)).save(path, "JPEG", xmp=XMP_GPS)

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert b"GPSLatitude" not in (out.info.get("xmp") or b"")
            assert GPS_IFD not in out.getexif()

    def test_gps_comment_stripped_on_copy_path(self, resizer, tmp_path):
        """A JPEG comment carrying coordinates must not ride the fast copy path."""
        path = tmp_path / "comment-geo.jpg"
        Image.new("RGB", (800, 600), (60, 60, 60)).save(
            path, "JPEG", comment=b"GPSLatitude=40.44 GPSLongitude=-74.02"
        )

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert b"GPSLatitude" not in (out.info.get("comment") or b"")

    def test_untagged_source_output_still_carries_srgb_tag(self, resizer, tmp_path):
        """Every re-encoded output ships the sRGB tag, matching the gallery exports —
        not only the sources that happened to arrive with a profile."""
        path = tmp_path / "untagged.jpg"
        Image.new("RGB", (1920, 1440), (200, 120, 60)).save(path, "JPEG", quality=95)

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            icc = out.info.get("icc_profile")
            assert icc, "untagged source produced an untagged resize"
            desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc)))
            assert "sRGB" in desc

    def test_resized_output_uses_explicit_chroma_subsampling(self, resizer, tmp_path):
        """The published copy must not fall back to PIL's implicit 4:2:0 default."""
        from PIL import JpegImagePlugin

        path = tmp_path / "photo.jpg"
        Image.new("RGB", (1920, 1440), (10, 200, 40)).save(path, "JPEG", quality=95)

        result = resizer.resize_batch([str(path)], "2026-04-06")
        with Image.open(result[0]) as out:
            assert JpegImagePlugin.get_sampling(out) == 1  # 4:2:2, not the 4:2:0 default

    def test_assert_no_gps_rejects_xmp_gps(self, tmp_path):
        """The pre-upload gate must catch GPS riding XMP, not only the EXIF GPS IFD."""
        path = tmp_path / "xmp-geo.jpg"
        Image.new("RGB", (100, 100), (10, 10, 10)).save(path, "JPEG", xmp=XMP_GPS)

        with pytest.raises(ValueError, match="GPS"):
            _assert_no_gps(path)
