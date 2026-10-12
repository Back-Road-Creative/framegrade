"""Thumbnail finish: design-spec bands and the 2 MB gate.

Image-quality spec: contrast +15 % minimum, saturation +15-20 %, mild
unsharp mask. The finish lives in ``pipeline/quality.py``
and runs between the lanczos grab and the text styles.
"""

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline import quality
from gomoveshift.pipeline import thumbnail_text_overlay as tto
from gomoveshift.pipeline.thumbnail_text_overlay import generate_single_style, ThumbnailStyle


def _scene(w=640, h=360) -> np.ndarray:
    """Mid-tone, mid-saturation gradient scene with headroom for +15 %."""
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:h, 0:w]
    base = 110 + 40 * np.sin(x / 37.0) * np.cos(y / 29.0)
    img = np.stack([base + 18, base - 6, base - 22], axis=-1)
    img += rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_thumbnail_finish_matches_design_spec_bands():
    before = _scene()
    after = quality.thumbnail_finish(before)
    assert after.shape == before.shape and after.dtype == np.uint8
    d = quality.measure_finish_deltas(before, after)
    # Spec values, not module constants: "+15 % minimum" / "+15-20 %".
    assert 0.15 <= d["contrast"] <= 0.25, d
    assert 0.15 <= d["saturation"] <= 0.20, d
    assert quality.THUMB_CONTRAST_BAND[0] == 0.15
    assert quality.THUMB_SATURATION_BAND == (0.15, 0.20)
    assert quality.thumbnail_finish_in_band(before, after)
    # Unsharp is mild: it raises detail (contrast gain included) but stays under 2.5x.
    ratio = quality.measure_detail(after) / quality.measure_detail(before)
    assert 1.0 < ratio <= 2.5, ratio


def test_finish_gate_rejects_off_spec_result():
    before = _scene()
    assert not quality.thumbnail_finish_in_band(before, before)
    blown = np.clip((before.astype(float) - 110) * 1.8 + 110, 0, 255).astype(np.uint8)
    assert not quality.thumbnail_finish_in_band(before, blown)


def _synthetic(before, contrast, saturation):
    """Hand-built 'after' frame with exact luma-std and chroma ratios."""
    f = before.astype(np.float64)
    luma = f @ np.array([0.2126, 0.7152, 0.0722])
    f = f + ((contrast - 1.0) * (luma - luma.mean()))[..., None]
    luma2 = f @ np.array([0.2126, 0.7152, 0.0722])
    f = luma2[..., None] + saturation * (f - luma2[..., None])
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    "contrast,saturation,ok",
    [
        (1.17, 1.175, True),
        (1.11, 1.175, False),  # contrast under +15 %
        (1.30, 1.175, False),  # contrast over the cap
        (1.17, 1.13, False),  # saturation under +15 %
        (1.17, 1.24, False),  # saturation over +20 %
    ],
)
def test_gate_edges_match_spec(contrast, saturation, ok):
    before = _scene()
    after = _synthetic(before, contrast, saturation)
    assert quality.thumbnail_finish_in_band(before, after) is ok, quality.measure_finish_deltas(
        before, after
    )


def _clipped_scene(w=640, h=360) -> np.ndarray:
    """Vivid sunset-like frame with clipped channels, JPEG q92 round-tripped."""
    import io

    y, x = np.mgrid[0:h, 0:w]
    base = 150 + 110 * np.sin(x / 37.0) * np.cos(y / 29.0)
    img = np.clip(np.stack([base + 90, base - 20, base - 110], axis=-1), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=92)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def test_clipped_frame_finish_lands_in_both_bands():
    """Clipping eats contrast and saturation at different rates; the gains are
    stepped independently so the clipped frame measures inside BOTH bands."""
    before = _clipped_scene()
    after = quality.thumbnail_finish(before)
    d = quality.measure_finish_deltas(before, after)
    assert 0.15 <= d["contrast"] <= 0.25, d
    assert 0.15 <= d["saturation"] <= 0.20, d
    assert quality.thumbnail_finish_in_band(before, after)


def test_clipped_frame_ships_finished(tmp_path, caplog):
    src = tmp_path / "scene.png"
    grab = _clipped_scene(1280, 720)
    Image.fromarray(grab).save(src)
    out = tmp_path / "t.jpg"
    with caplog.at_level("WARNING"):
        generate_single_style(str(src), "Title", None, ThumbnailStyle.MINIMAL, str(out))
    assert out.exists() and out.stat().st_size <= quality.THUMB_MAX_BYTES
    assert "unfinished" not in caplog.text


def _black_white_split(w=640, h=360) -> np.ndarray:
    """Pure black | pure white: luma std is high but no gain can raise it."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, w // 2 :] = 255
    return img


def test_unreachable_frame_ships_unfinished_and_logs_why(tmp_path, caplog):
    grab = _black_white_split(1280, 720)
    src = tmp_path / "scene.png"
    Image.fromarray(grab).save(src)
    out = tmp_path / "t.jpg"
    with caplog.at_level("WARNING"):
        generate_single_style(str(src), "Title", None, ThumbnailStyle.MINIMAL, str(out))
    assert out.exists()
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "unfinished" in msgs and "contrast" in msgs, msgs
    assert not quality.thumbnail_finish_in_band(grab, quality.thumbnail_finish(grab))


def test_thumbnail_over_2mb_refused(tmp_path, monkeypatch):
    # Full-range white noise is incompressible: over 2 MB at every quality step.
    rng = np.random.default_rng(1)
    noise = Image.fromarray(rng.integers(0, 256, (2160, 3840, 3), dtype=np.uint8))
    out = tmp_path / "noise.jpg"
    with pytest.raises(quality.ThumbnailTooLargeError):
        quality.encode_thumbnail_jpeg(noise, str(out))
    assert not out.exists()

    # And the same refusal reaches the public entry point, before any write.
    src = tmp_path / "scene.png"
    Image.fromarray(_scene(1280, 720)).save(src)
    monkeypatch.setattr(quality, "THUMB_MAX_BYTES", 1000)
    out2 = tmp_path / "out" / "t.jpg"
    with pytest.raises(quality.ThumbnailTooLargeError):
        generate_single_style(str(src), "Title", None, ThumbnailStyle.MINIMAL, str(out2))
    assert not out2.exists()


def test_normal_thumbnail_is_finished_and_under_2mb(tmp_path):
    src = tmp_path / "scene.png"
    Image.fromarray(_scene(1280, 720)).save(src)
    out = tmp_path / "t.jpg"
    generate_single_style(str(src), "Title", None, ThumbnailStyle.MINIMAL, str(out))
    assert out.exists() and out.stat().st_size <= quality.THUMB_MAX_BYTES


def test_overlay_applies_finish_between_grab_and_text(tmp_path, monkeypatch):
    """The wiring: the style drawer receives the finished frame, not the grab."""
    from gomoveshift.pipeline import thumbnail_text_overlay as tto

    seen = {}
    real = tto._draw_minimal_style

    def spy(bg, title, subtitle):
        seen["bg"] = np.asarray(bg).copy()
        return real(bg, title, subtitle)

    monkeypatch.setattr(tto, "_draw_minimal_style", spy)
    grab = _scene(1280, 720)
    src = tmp_path / "scene.png"
    Image.fromarray(grab).save(src)
    generate_single_style(str(src), "Title", None, ThumbnailStyle.MINIMAL, str(tmp_path / "t.jpg"))
    d = quality.measure_finish_deltas(grab, seen["bg"])
    assert 0.15 <= d["contrast"] <= 0.25, d
    assert 0.15 <= d["saturation"] <= 0.20, d


def _spy_full_size_work(monkeypatch, full_shape):
    """Count full-resolution apply (_finish_pass) and measure calls."""
    calls = {"apply": 0, "measure": 0}
    real_pass = quality._finish_pass
    real_measure = quality.measure_finish_deltas

    def pass_spy(rgb, *a, **k):
        if rgb.shape == full_shape:
            calls["apply"] += 1
        return real_pass(rgb, *a, **k)

    def measure_spy(before, after):
        if before.shape == full_shape:
            calls["measure"] += 1
        return real_measure(before, after)

    monkeypatch.setattr(quality, "_finish_pass", pass_spy)
    monkeypatch.setattr(quality, "measure_finish_deltas", measure_spy)
    return calls


@pytest.mark.parametrize("make", [_clipped_scene, _black_white_split])
def test_finish_cost_is_one_full_res_apply_and_one_measure(tmp_path, monkeypatch, make):
    """Cost by construction: the gain search runs on a proxy, so the clipped and
    the unreachable frame each cost exactly one full-size apply and one measure,
    never a pass per candidate pair."""
    grab = make(1280, 720)
    src = tmp_path / "scene.png"
    Image.fromarray(grab).save(src)
    calls = _spy_full_size_work(monkeypatch, grab.shape)
    generate_single_style(str(src), "T", None, ThumbnailStyle.MINIMAL, str(tmp_path / "t.jpg"))
    assert calls == {"apply": 1, "measure": 1}, calls


def test_proxy_drift_retries_stay_within_cap(tmp_path, monkeypatch):
    """A textured frame drifts between proxy and full size; drift correction may
    retry at full size, but never past THUMB_FULL_RES_TRIES."""
    grab = _textured_scene(30)
    calls = _spy_full_size_work(monkeypatch, grab.shape)
    quality.thumbnail_finish(grab)
    cap = quality.THUMB_FULL_RES_TRIES
    assert 1 <= calls["apply"] <= cap and calls["measure"] <= cap, calls


def test_all_styles_share_one_finish_per_background(tmp_path, monkeypatch):
    grab = _clipped_scene(1280, 720)
    src = tmp_path / "scene.png"
    Image.fromarray(grab).save(src)
    calls = _spy_full_size_work(monkeypatch, grab.shape)
    finishes = []
    real_finish = tto.thumbnail_finish_measured
    monkeypatch.setattr(
        tto, "thumbnail_finish_measured", lambda a: finishes.append(1) or real_finish(a)
    )
    results = tto.generate_styled_thumbnails(str(src), "T", None, str(tmp_path / "out"))
    assert len(results) == len(ThumbnailStyle)
    assert len(finishes) == 1
    assert calls == {"apply": 1, "measure": 1}, calls


def _textured_scene(noise, w=1280, h=720, amp=90, sat=60) -> np.ndarray:
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:h, 0:w]
    base = 120 + amp * np.sin(x / 53.0) * np.cos(y / 41.0)
    img = np.stack([base + sat, base, base - sat], axis=-1) + rng.normal(0, noise, (h, w, 3))
    img += rng.normal(0, noise, (h, w, 1))
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("noise", [15, 30])
def test_proxy_chosen_pair_lands_in_band_at_full_size(noise):
    """The proxy drifts from full size; the search must still land in band."""
    grab = _textured_scene(noise)
    after = quality.thumbnail_finish(grab)
    d = quality.measure_finish_deltas(grab, after)
    assert quality.thumbnail_finish_in_band(grab, after), d


def test_unfinished_warning_does_not_claim_base_gain(tmp_path, caplog):
    src = tmp_path / "scene.png"
    Image.fromarray(_black_white_split(1280, 720)).save(src)
    with caplog.at_level("WARNING"):
        tto.finish_background(str(src))
    assert "base-gain" not in caplog.text and "unfinished" in caplog.text


def test_styles_do_not_share_drawn_pixels(tmp_path):
    bg = Image.fromarray(_scene(1280, 720))
    before = np.asarray(bg).copy()
    out = str(tmp_path / "t.jpg")
    generate_single_style("unused.png", "Title", None, ThumbnailStyle.IMMERSIVE, out, bg=bg)
    assert np.array_equal(np.asarray(bg), before)
