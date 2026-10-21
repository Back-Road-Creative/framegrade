"""The installed Waypost pin must support GMS's retained-region contract."""

from seo_manager.analytics import final_text_review
from seo_manager.analytics.title_selection import (
    DEFAULT_POLICY,
    EvidenceSnapshot,
    Place,
    TitleFacts,
    select_title,
)


def test_pinned_waypost_supports_bounded_region_and_region_only_title():
    assert final_text_review.BOUNDARY_PROOF_VERSION == "tiger2026-bounded-v1"
    facts = TitleFacts(
        format="longform",
        baseline_title="Beaufort County, SC in 4K",
        region=Place("Beaufort County, SC", "region-beaufort"),
        suffix=" in 4K",
    )
    result = select_title(facts, EvidenceSnapshot(as_of="2026-10-18T00:00:00"), DEFAULT_POLICY)
    assert result.chosen_title == facts.baseline_title
    assert result.candidates[0].template_id == "region_only_lf"
    assert result.candidates[0].claim_evidence == ("region-beaufort",)


def test_pinned_waypost_accepts_gms_gap_float_noise():
    # GMS rebuilds a gap end as start + length, so a gap equal to the proof's
    # 3-decimal max_gap_s carries float noise (35.999000000000024 > 35.999).
    proof = {
        "verified": True,
        "basis": "pinned_tiger_polygon_bounded_speed_and_error",
        "source": "TIGER2026/COUNTY/tl_2026_us_county.zip",
        "sha256": "b527c9ba7aeb8f17ef118f026eae5e5a938aa1d00b873bc32342d0dce59321c0",  # pragma: allowlist secret
        "region": "Beaufort County",
        "statefp": "45",
        "recording_date": "2026-02-20",
        "max_speed_m_s": 70.0,
        "gps_error_m": 100.0,
        "max_gap_s": 35.999,
    }
    kw = {"region": "Beaufort County", "recording_date": "2026-02-20"}
    assert final_text_review._bounded_proof(proof, gaps=((941.129, 941.129 + 35.999),), **kw)
    assert not final_text_review._bounded_proof(proof, gaps=((941.129, 941.129 + 36.001),), **kw)
