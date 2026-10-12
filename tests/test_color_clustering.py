"""Tests for color clustering across drive pipeline files."""

import itertools
from pathlib import Path

import pytest

from gomoveshift.pipeline.color import (
    SegmentStats,
    cluster_files_by_lighting,
    compute_cluster_correction,
    compute_correction,
    merge_adjacent_segments,
)


class TestMergeAdjacentSegments:
    """Fold consecutive similar 30s segments into LUT spans."""

    def _seg(self, black=16, white=235, mid=128, sat=0.3) -> SegmentStats:
        return SegmentStats(black_point=black, white_point=white, mid_tone=mid, saturation=sat)

    def test_stable_lighting_collapses_to_one_span(self):
        spans, durations = merge_adjacent_segments([self._seg()] * 10, segment_duration=30.0)
        assert len(spans) == 1
        assert durations == [300.0]

    def test_lighting_transition_splits_spans(self):
        """Daylight -> tunnel -> daylight yields three spans at the boundaries."""
        day = self._seg(black=20, white=220, mid=140, sat=0.35)
        tunnel = self._seg(black=2, white=80, mid=30, sat=0.15)
        segs = [day] * 4 + [tunnel] * 3 + [day] * 5
        spans, durations = merge_adjacent_segments(segs, segment_duration=30.0)
        assert len(spans) == 3
        assert durations == [120.0, 90.0, 150.0]
        # Aggregated span stats reflect their members, in order.
        assert spans[0].mid_tone == pytest.approx(140)
        assert spans[1].mid_tone == pytest.approx(30)

    def test_durations_sum_preserved(self):
        segs = [self._seg(mid=40 + (i % 5) * 40) for i in range(20)]
        _, durations = merge_adjacent_segments(segs, segment_duration=30.0)
        assert sum(durations) == pytest.approx(20 * 30.0)

    def test_span_count_capped(self):
        """Pathological oscillating lighting must coarsen below the cap —
        one LUT per raw segment (~600 on a 5h master) is unencodable."""
        segs = [
            self._seg(black=5 + (i % 7) * 30, white=90 + (i % 5) * 30, mid=40 + (i % 9) * 20)
            for i in range(100)
        ]
        spans, durations = merge_adjacent_segments(segs, segment_duration=30.0, max_spans=12)
        assert len(spans) <= 12
        assert sum(durations) == pytest.approx(100 * 30.0)

    def test_empty_input(self):
        assert merge_adjacent_segments([]) == ([], [])


class TestClusterFilesByLighting:
    """Group files into clusters with similar lighting conditions."""

    def _make_stats(self, black=16, white=235, mid=128, sat=0.3) -> SegmentStats:
        return SegmentStats(black_point=black, white_point=white, mid_tone=mid, saturation=sat)

    def test_single_file_one_cluster(self):
        """One file -> one cluster."""
        file_stats = [(Path("/tmp/a.mp4"), self._make_stats())]
        clusters = cluster_files_by_lighting(file_stats)
        assert len(clusters) == 1
        assert clusters[0] == [0]

    def test_identical_files_one_cluster(self):
        """Files with identical stats -> single cluster."""
        stats = self._make_stats()
        file_stats = [(Path(f"/tmp/{i}.mp4"), stats) for i in range(5)]
        clusters = cluster_files_by_lighting(file_stats)
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_distinct_lighting_multiple_clusters(self):
        """Files with very different lighting -> separate clusters."""
        # Sunrise: dark, low midtone
        sunrise = self._make_stats(black=5, white=180, mid=70, sat=0.4)
        # Midday: bright, high midtone
        midday = self._make_stats(black=20, white=245, mid=150, sat=0.3)
        # Sunset: warm, moderate
        sunset = self._make_stats(black=10, white=200, mid=90, sat=0.5)

        file_stats = [
            (Path("/tmp/0.mp4"), sunrise),
            (Path("/tmp/1.mp4"), sunrise),
            (Path("/tmp/2.mp4"), midday),
            (Path("/tmp/3.mp4"), midday),
            (Path("/tmp/4.mp4"), midday),
            (Path("/tmp/5.mp4"), sunset),
            (Path("/tmp/6.mp4"), sunset),
        ]
        clusters = cluster_files_by_lighting(file_stats, variance_threshold=0.15)
        # Should produce 2-3 clusters (sunrise+sunset might merge vs midday)
        assert len(clusters) >= 2
        # All file indices accounted for
        all_indices = sorted(idx for cluster in clusters for idx in cluster)
        assert all_indices == [0, 1, 2, 3, 4, 5, 6]

    def test_empty_input(self):
        """No files -> no clusters."""
        clusters = cluster_files_by_lighting([])
        assert clusters == []

    def test_cluster_order_preserves_file_indices(self):
        """Clusters contain correct file indices."""
        file_stats = [
            (Path("/tmp/0.mp4"), self._make_stats(mid=80)),
            (Path("/tmp/1.mp4"), self._make_stats(mid=180)),
            (Path("/tmp/2.mp4"), self._make_stats(mid=82)),
        ]
        clusters = cluster_files_by_lighting(file_stats, variance_threshold=0.1)
        # Files 0 and 2 should cluster together (mid ~80), file 1 separate (mid 180)
        found_dark = None
        found_bright = None
        for cluster in clusters:
            if 0 in cluster:
                found_dark = cluster
            if 1 in cluster:
                found_bright = cluster
        assert found_dark is not None
        assert found_bright is not None
        assert 2 in found_dark
        assert 1 not in found_dark


class TestClusterQuality:
    """Deterministic, chroma-aware file clustering — greedy assignment is
    order-dependent and a luma-only feature can't tell golden from blue hour."""

    def _stats(self, mid=128.0, r=0.5, g=0.5, b=0.5) -> SegmentStats:
        return SegmentStats(
            black_point=16,
            white_point=235,
            mid_tone=mid,
            saturation=0.3,
            r_mean=r,
            g_mean=g,
            b_mean=b,
        )

    def test_input_order_does_not_change_grouping(self):
        """The same file set in any order must yield the same clusters.

        Three files at evenly spaced mid_tones (each gap under the threshold,
        the ends over it) are the classic greedy trap: scanning left-to-right
        groups {a,b}, scanning right-to-left groups {b,c}.
        """
        named = [
            ("a", self._stats(mid=50)),
            ("b", self._stats(mid=120)),
            ("c", self._stats(mid=190)),
        ]
        groupings = set()
        for perm in itertools.permutations(named):
            file_stats = [(Path(f"/tmp/{name}.jpg"), stats) for name, stats in perm]
            clusters = cluster_files_by_lighting(file_stats)
            groupings.add(
                frozenset(frozenset(file_stats[i][0].name for i in cluster) for cluster in clusters)
            )
        assert len(groupings) == 1, f"input order changed the grouping: {groupings}"

    def test_opposite_casts_split_at_equal_luma(self):
        """Golden hour vs blue hour at identical luminance must not share a cluster.

        A luma-only feature scores them identical, so one averaged white
        balance would be applied across both — exactly the blend per-cluster
        correction exists to prevent.
        """
        golden = self._stats(r=0.60, g=0.50, b=0.40)
        blue = self._stats(r=0.40, g=0.50, b=0.60)
        file_stats = [
            (Path("/tmp/g0.jpg"), golden),
            (Path("/tmp/g1.jpg"), golden),
            (Path("/tmp/b0.jpg"), blue),
            (Path("/tmp/b1.jpg"), blue),
        ]
        clusters = cluster_files_by_lighting(file_stats)
        assert {frozenset(c) for c in clusters} == {frozenset({0, 1}), frozenset({2, 3})}

    def test_mild_chroma_wobble_stays_one_cluster(self):
        """Small per-channel wobble within one scene must not fragment the cluster."""
        file_stats = [
            (Path("/tmp/0.jpg"), self._stats(r=0.55, b=0.45)),
            (Path("/tmp/1.jpg"), self._stats(r=0.57, b=0.44)),
            (Path("/tmp/2.jpg"), self._stats(r=0.54, b=0.46)),
        ]
        clusters = cluster_files_by_lighting(file_stats)
        assert len(clusters) == 1

    def test_clusters_reported_in_input_index_order(self):
        """Indices ascend within a cluster; clusters order by first member."""
        file_stats = [
            (Path("/tmp/0.jpg"), self._stats(mid=180)),
            (Path("/tmp/1.jpg"), self._stats(mid=40)),
            (Path("/tmp/2.jpg"), self._stats(mid=182)),
        ]
        clusters = cluster_files_by_lighting(file_stats, variance_threshold=0.1)
        assert clusters == [[0, 2], [1]]


class TestComputeClusterCorrection:
    """Compute a single correction for an entire cluster."""

    def test_single_file_matches_direct_correction(self):
        """Cluster of one = same as compute_correction on that file."""
        stats = SegmentStats(black_point=10, white_point=240, mid_tone=120, saturation=0.25)
        cluster_result = compute_cluster_correction([stats])
        direct_result = compute_correction(stats)
        assert cluster_result.shadow_lift == direct_result.shadow_lift
        assert cluster_result.highlight_pull == direct_result.highlight_pull
        assert abs(cluster_result.saturation_mult - direct_result.saturation_mult) < 0.01

    def test_multiple_files_uses_median(self):
        """Cluster correction uses median stats across files."""
        stats_list = [
            SegmentStats(black_point=1, white_point=230, mid_tone=100, saturation=0.2),
            SegmentStats(black_point=7, white_point=240, mid_tone=130, saturation=0.3),
            SegmentStats(black_point=4, white_point=235, mid_tone=120, saturation=0.25),
        ]
        result = compute_cluster_correction(stats_list)
        # Median black_point = 4 → adaptive shadow_lift = max(2.0, 15.0 - 4*0.15) = 14.4
        assert result.shadow_lift == pytest.approx(14.4)

    def test_consistent_warm_cast_produces_white_balance(self):
        """A cluster whose files share a warm cast must grade with a real WB cast.

        The cluster path rebuilt its aggregate without the per-channel means,
        so they fell back to the neutral 0.5 defaults and every cluster shipped
        white_balance_r == white_balance_b == 0 — no white balance at all.
        """
        warm = [
            SegmentStats(
                black_point=12,
                white_point=235,
                mid_tone=125,
                saturation=0.3,
                r_mean=0.60,
                g_mean=0.50,
                b_mean=0.44,
            )
            for _ in range(3)
        ]
        result = compute_cluster_correction(warm)
        # Red-heavy / blue-light source: pull red down, push blue up.
        assert result.white_balance_r < 0
        assert result.white_balance_b > 0

    def test_single_file_wb_matches_direct_correction(self):
        """Cluster of one must produce the same WB cast as the direct path."""
        stats = SegmentStats(
            black_point=10,
            white_point=240,
            mid_tone=120,
            saturation=0.25,
            r_mean=0.42,
            g_mean=0.50,
            b_mean=0.56,
        )
        cluster_result = compute_cluster_correction([stats])
        direct_result = compute_correction(stats)
        assert cluster_result.white_balance_r == direct_result.white_balance_r
        assert cluster_result.white_balance_b == direct_result.white_balance_b
        # Cool source: push red up, pull blue down.
        assert cluster_result.white_balance_r > 0
        assert cluster_result.white_balance_b < 0

    def test_median_channel_means_resist_outlier_file(self):
        """One neutral outlier must not cancel the cluster's dominant cast."""
        warm = SegmentStats(
            black_point=12,
            white_point=235,
            mid_tone=125,
            saturation=0.3,
            r_mean=0.60,
            g_mean=0.50,
            b_mean=0.44,
        )
        neutral = SegmentStats(
            black_point=12,
            white_point=235,
            mid_tone=125,
            saturation=0.3,
            r_mean=0.50,
            g_mean=0.50,
            b_mean=0.50,
        )
        result = compute_cluster_correction([warm, warm, neutral])
        assert result.white_balance_r < 0
        assert result.white_balance_b > 0

    def test_empty_cluster_returns_neutral(self):
        """Empty cluster -> neutral correction."""
        result = compute_cluster_correction([])
        assert result.shadow_lift == 0.0
        assert result.highlight_pull == 0.0
        assert result.saturation_mult == 1.0
