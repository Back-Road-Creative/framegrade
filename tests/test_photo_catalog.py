"""The photo effects have ONE catalog, built from the registries that render them.

The request was for "capability discovery". Every look, mask and style imports only core
deps (numpy, cv2, the hard `sun-track` dep), so a dependency probe would answer
"usable" for every entry -- a gate that cannot close. What was actually missing is
one enumerable source the console and CLI can read instead of listing everything
by hand: these tests hold that the catalog is derived, never typed.
"""

from __future__ import annotations

from gomoveshift.pipeline.photos import catalog
from gomoveshift.pipeline.photos.looks import BUILTIN_LOOKS
from gomoveshift.pipeline.photos.masks import MASK_PARAMS, MASKS
from gomoveshift.pipeline.photos.styles import STYLES


def test_every_registered_style_is_in_the_catalog_and_nothing_else():
    assert catalog.style_names() == list(STYLES)


def test_every_registered_mask_carries_its_own_params():
    masks = catalog.masks()
    assert set(masks) == set(MASKS)
    assert masks["linear"] == sorted(MASK_PARAMS["linear"])
    assert "angle" in masks["linear"] and "angle" not in masks["sky"]


def test_every_builtin_look_is_catalogued_with_the_masks_it_layers():
    looks = catalog.looks()
    assert set(looks) == set(BUILTIN_LOOKS)
    # split_world is the masked look: sky and its complement, no seam.
    assert looks["split_world"] == ["ground", "sky"]
    # a flat (unmasked) look layers nothing
    assert looks["cyberpunk"] == []


def test_a_configured_look_joins_the_catalog_and_wins_a_name_collision(monkeypatch):
    from gomoveshift import config as config_module

    class _Photos:
        looks = {"cyberpunk": {"layers": [{"mask": "hue"}]}, "mine": {}}

    class _Cfg:
        photos = _Photos()

    monkeypatch.setattr(config_module, "get_config", lambda: _Cfg())
    looks = catalog.looks(configured=config_module.get_config().photos.looks)
    assert looks["mine"] == []
    assert looks["cyberpunk"] == ["hue"]


def test_a_look_naming_an_unregistered_mask_is_reported_not_rendered():
    """A recipe can only be wrong in one place: the mask it names. Say which."""
    bad = catalog.invalid_looks(
        {"ok": {"layers": [{"mask": "sky"}]}, "no": {"layers": [{"mask": "moon"}]}}
    )
    assert bad == {"no": ["moon"]}


def test_the_catalog_is_one_json_document():
    doc = catalog.effects_catalog()
    assert set(doc) == {"looks", "masks", "styles"}
    assert doc["styles"] == list(STYLES)
    assert doc["masks"]["radial"] == sorted(MASK_PARAMS["radial"])
