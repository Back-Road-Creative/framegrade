"""A tenant ``brand.yaml`` carries the ``seo: cadence:`` block the hourly publisher reads.

Without that block ``schedule claim`` (SEO_BRAND_YAML) raises "no `seo: cadence:` block" and
nothing can claim. The packaged ``brand.yaml`` is the SEED of Waypost's cadence (the ``lf_days``
decisions overlay its days); its designed times are pinned to ``drive.publish_slots`` and its
days to ``seo_slots.cadence_for`` so the two cannot drift. The shape tests build a synthetic
``brand.yaml`` holding the documented cadence."""

from __future__ import annotations

from datetime import time as dt_time
from pathlib import Path

import pytest
from seo_manager.core.profile import DAY_ABBR, load_cadence, parse_weekday

from gomoveshift.config import Config
from gomoveshift.pipeline.schedule_constants import _SUPERLONG_DEFAULT_DAY, EST
from gomoveshift.pipeline.seo_slots import cadence_for

SEED = Path(__file__).resolve().parents[1] / "gomoveshift/console/brands/gomoveshift/brand.yaml"

_BRAND_YAML = """\
seo:
  cadence:
    timezone: America/New_York
    lead_hours: 72
    superlong_day: sat
    regions:
      ireland:
        dow: [sun]
        time: "14:00"
        shorts_times: ["08:00", "12:00"]
      usa:
        dow: [tue, fri]
        time: "16:00"
        day_times:
          fri: "15:00"
        shorts_times: ["14:00", "18:00"]
"""


@pytest.fixture
def brand(tmp_path):
    path = tmp_path / "brand.yaml"
    path.write_text(_BRAND_YAML, encoding="utf-8")
    return path


def _hhmm(text: str) -> dt_time:
    h, m = text.split(":")
    return dt_time(int(h), int(m))


def test_brand_yaml_carries_a_loadable_cadence_block(brand):
    cad = load_cadence(brand)
    assert sorted(cad.regions) == ["ireland", "usa"]
    assert cad.tz == EST


def test_block_matches_the_gms_config_cadence(tmp_path):
    """The seed's designed times equal ``drive.publish_slots`` (the shorts/lf-time table), and
    ``cadence_for`` over an empty decision store returns the seed's own days."""
    config = Config()
    config.seo.storage_path = str(tmp_path)
    block = load_cadence(SEED)
    assert set(block.regions) == set(cadence_for(config).regions)
    for name, region in block.regions.items():
        days = {DAY_ABBR[d].lower() for d in region.dow}
        assert days <= set(config.drive.publish_slots), (name, days)
        assert tuple(region.dow) == tuple(cadence_for(config, name).region(name).dow)
        for d in region.dow:
            designed = _hhmm(config.drive.publish_slots[DAY_ABBR[d].lower()].lf_time)
            assert region.day_times.get(d, region.time) == designed, (name, d)
    assert block.superlong_day == parse_weekday("superlong", _SUPERLONG_DEFAULT_DAY)


def test_block_pins_the_literal_region_days_and_times(brand):
    """The comparison above builds its expectation from the block's own days, so a swapped
    region (ireland to tue) would still pass it. These literals are the cadence contract."""
    cad = load_cadence(brand)
    ireland, usa = cad.region("ireland"), cad.region("usa")
    assert {DAY_ABBR[d].lower() for d in ireland.dow} == {"sun"}
    assert ireland.time == dt_time(14, 0) and not ireland.day_times
    assert {DAY_ABBR[d].lower() for d in usa.dow} == {"tue", "fri"}
    assert usa.time == dt_time(16, 0)
    assert dict(usa.day_times) == {parse_weekday("fri", "fri"): dt_time(15, 0)}
    assert cad.superlong_day == parse_weekday("superlong", "sat")
    assert cad.lead_hours == 72
