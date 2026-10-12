"""Optional brand kit (font/colors/logo/safe_margin) in brand.yaml,
consumed by the OST drawtext pipeline when present.

HARD INVARIANT: a brand with no kit makes ``brand_kit()`` return ``None``, and
``apply_ost``/``_build_drawtext_filter`` with no ``brand_kit`` argument must emit
the BYTE-IDENTICAL drawtext filter string. ``_GOLDEN_FILTER`` below is that
string for the inputs ``_build`` uses. The brands come from the
``shipped_brands`` fixture (``tests/conftest.py``), since the source tree ships
no brand templates.
"""

import pytest
import yaml

from gomoveshift.console import settings as console_settings
from gomoveshift.pipeline.ost_applier import FONT_PATH, LAYOUTS, _build_drawtext_filter

_GOLDEN_FILTER = (
    "drawtext=fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf':"
    "text='GoMoveShift':fontsize=64:fontcolor=white@1.0:borderw=3:bordercolor=black@0.8:"
    "x=(w-text_w)/2:y=h*0.68:shadowcolor=black@0.6:shadowx=5:shadowy=5:"
    "alpha='if(lt(t,14.5),1,if(lt(t,15.0),1-(t-(14.5))/0.5,0))':enable='between(t,0,15.0)',"
    "drawtext=fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf':"
    "text='Frogmore to Port Royal via Overland Trail':fontsize=48:fontcolor=white@1.0:"
    "borderw=3:bordercolor=black@0.8:x=(w-text_w)/2:y=h*0.76+0:shadowcolor=black@0.6:"
    "shadowx=4:shadowy=4:alpha='if(lt(t,14.5),1,if(lt(t,15.0),1-(t-(14.5))/0.5,0))':"
    "enable='between(t,0,15.0)',"
    "drawtext=fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf':"
    "text='gomoveshift.com':fontsize=28:fontcolor=white@0.85:borderw=2:bordercolor=black@0.7:"
    "x=w-text_w-40:y=40:shadowcolor=black@0.6:shadowx=2:shadowy=2,"
    "drawtext=fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf':"
    "text='gomoveshift.com':fontsize=48:fontcolor=white@1.0:borderw=3:bordercolor=black@0.8:"
    "x=(w-text_w)/2:y=h*0.75:shadowcolor=black@0.6:shadowx=4:shadowy=4:"
    "alpha='if(lt(t\\,593.5)\\,0\\,if(lt(t\\,594.5)\\,(t-593.5)/1.0\\,"
    "if(lt(t\\,598.5)\\,1\\,if(lt(t\\,599.5)\\,(599.5-t)/1.0\\,0))))':"
    "enable='between(t,593.5,599.5)'"
)


def _build(brand_kit=None, font_path=FONT_PATH):
    return _build_drawtext_filter(
        location="Frogmore to Port Royal via Overland Trail",
        layout=LAYOUTS["regular_lf"],
        font_path=font_path,
        video_width=1920,
        video_duration=600.0,
        montage_duration=15.0,
        brand_kit=brand_kit,
    )


class TestNoKitIsByteIdentical:
    def test_no_brand_kit_argument_matches_golden(self):
        assert _build() == _GOLDEN_FILTER

    def test_gomoveshift_has_no_brand_kit(self, shipped_brands):
        assert console_settings.brand_kit("gomoveshift") is None

    def test_examplebrand_disabled_tabs_only_has_no_kit(self, shipped_brands):
        # The fixture's brand.yaml is disabled_tabs-only --
        # confirms brand_kit() degrades to None for a kit-less brand.yaml too.
        assert console_settings.brand_kit("examplebrand") is None


class TestBrandKitConsumedByOst:
    def test_kit_font_overrides_when_no_explicit_font_path_passed(self):
        kit = {"font": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "colors": {}}
        out = _build_drawtext_filter(
            location="Loc",
            layout=LAYOUTS["regular_lf"],
            font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            video_width=1920,
            video_duration=600.0,
            montage_duration=15.0,
            brand_kit=kit,
        )
        assert "DejaVuSans.ttf" in out

    def test_kit_text_color_overrides_fontcolor(self):
        kit = {"colors": {"text": "#ff8800"}}
        out = _build(brand_kit=kit)
        assert "fontcolor=#ff8800@1.0" in out
        assert "fontcolor=white@1.0" not in out

    def test_no_kit_text_still_defaults_to_white(self):
        kit = {"colors": {}}
        out = _build(brand_kit=kit)
        assert "fontcolor=white@1.0" in out


class TestBrandKitSchemaValidation:
    def _write_kit(self, tmp_path, monkeypatch, kit_yaml_block):
        brands_dir = tmp_path / "brands" / "testbrand"
        brands_dir.mkdir(parents=True)
        (brands_dir / "brand.yaml").write_text(kit_yaml_block)
        monkeypatch.setattr(console_settings, "_SHIPPED_BRANDS_DIR", tmp_path / "brands")

    def test_valid_kit_parses(self, tmp_path, monkeypatch):
        self._write_kit(
            tmp_path,
            monkeypatch,
            yaml.safe_dump(
                {
                    "kit": {
                        "font": "/fonts/brand.ttf",
                        "colors": {"primary": "#112233", "accent": "#445566", "text": "#ffffff"},
                        "logo": "/assets/logo.png",
                        "safe_margin": 0.08,
                    }
                }
            ),
        )
        kit = console_settings.brand_kit("testbrand")
        assert kit["font"] == "/fonts/brand.ttf"
        assert kit["colors"]["primary"] == "#112233"
        assert kit["logo"] == "/assets/logo.png"
        assert kit["safe_margin"] == pytest.approx(0.08)

    def test_missing_kit_key_returns_none(self, tmp_path, monkeypatch):
        self._write_kit(tmp_path, monkeypatch, "disabled_tabs: []\n")
        assert console_settings.brand_kit("testbrand") is None

    def test_invalid_hex_color_raises_with_field_name(self, tmp_path, monkeypatch):
        self._write_kit(
            tmp_path,
            monkeypatch,
            yaml.safe_dump({"kit": {"colors": {"text": "not-a-hex"}}}),
        )
        with pytest.raises(ValueError, match="kit.colors.text"):
            console_settings.brand_kit("testbrand")

    def test_invalid_safe_margin_out_of_range_raises_with_field_name(self, tmp_path, monkeypatch):
        self._write_kit(
            tmp_path,
            monkeypatch,
            yaml.safe_dump({"kit": {"safe_margin": 1.5}}),
        )
        with pytest.raises(ValueError, match="kit.safe_margin"):
            console_settings.brand_kit("testbrand")

    def test_non_mapping_kit_raises(self, tmp_path, monkeypatch):
        self._write_kit(tmp_path, monkeypatch, yaml.safe_dump({"kit": "nope"}))
        with pytest.raises(ValueError, match="kit"):
            console_settings.brand_kit("testbrand")
