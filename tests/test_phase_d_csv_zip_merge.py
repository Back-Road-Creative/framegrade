"""Position-based zip merge for ``drive_links_by_short_index``.

Closes the day-local filename collision regression: every short across
every LF must map to its correct Drive URL in the consolidated CSV — the
legacy parse-from-filename merge collided when manifests were keyed by
day-local short numbers (``short_01.mp4`` appearing under multiple LFs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    _build_drive_links_by_short_index,
)


@dataclass
class _FakePost:
    """Minimal stand-in for ``SocialPost`` — only ``short_index`` is read."""

    short_index: int


@dataclass
class _FakeManifest:
    """Minimal stand-in for ``UploadManifest`` — only ``files`` is read."""

    files: dict


def _mk_manifest(*pairs: tuple[str, str]) -> _FakeManifest:
    """Build a manifest preserving insertion order from positional pairs."""
    return _FakeManifest(files=dict(pairs))


# ── Happy paths ──────────────────────────────────────────────────────────


class TestPositionZipMerge:
    def test_two_lfs_three_shorts_each_daylocal_collision(self):
        """2 LFs × 3 shorts with day-local manifest filenames → all 6 mapped.

        Regression case: the writer used ``short_01.mp4`` for the first short of
        EACH LF. Parsing the filename collapsed both LFs onto short_index=1
        and one URL won. Position-zip must keep them distinct.
        """
        social_posts = {
            1: [_FakePost(1), _FakePost(2), _FakePost(3)],
            2: [_FakePost(4), _FakePost(5), _FakePost(6)],
        }
        manifests = {
            1: _mk_manifest(
                ("short_01.mp4", "https://drive.google.com/file/d/lf1_a"),
                ("short_02.mp4", "https://drive.google.com/file/d/lf1_b"),
                ("short_03.mp4", "https://drive.google.com/file/d/lf1_c"),
            ),
            2: _mk_manifest(
                ("short_01.mp4", "https://drive.google.com/file/d/lf2_a"),
                ("short_02.mp4", "https://drive.google.com/file/d/lf2_b"),
                ("short_03.mp4", "https://drive.google.com/file/d/lf2_c"),
            ),
        }

        links = _build_drive_links_by_short_index(
            social_posts_per_lf=social_posts,
            upload_manifests=manifests,
        )

        assert links == {
            1: "https://drive.google.com/file/d/lf1_a",
            2: "https://drive.google.com/file/d/lf1_b",
            3: "https://drive.google.com/file/d/lf1_c",
            4: "https://drive.google.com/file/d/lf2_a",
            5: "https://drive.google.com/file/d/lf2_b",
            6: "https://drive.google.com/file/d/lf2_c",
        }

    def test_seven_lfs_53_shorts_integration_shape(self):
        """Fixture: 7 LFs totalling 53 shorts → CSV writer gets all 53 keys."""
        # Match a realistic cohort shape — 7 LFs of varying short counts
        # totalling 53. Use day-local filenames for every LF.
        per_lf_counts = [8, 8, 8, 7, 8, 7, 7]
        assert sum(per_lf_counts) == 53

        social_posts: dict[int, list] = {}
        manifests: dict[int, _FakeManifest] = {}
        global_idx = 1
        for lf_idx, count in enumerate(per_lf_counts, start=1):
            social_posts[lf_idx] = [_FakePost(global_idx + i) for i in range(count)]
            manifests[lf_idx] = _mk_manifest(
                *(
                    (
                        f"short_{i + 1:02d}.mp4",
                        f"https://drive.google.com/file/d/lf{lf_idx}_short{i + 1}",
                    )
                    for i in range(count)
                )
            )
            global_idx += count

        links = _build_drive_links_by_short_index(
            social_posts_per_lf=social_posts,
            upload_manifests=manifests,
        )

        assert len(links) == 53
        # Every global index 1..53 maps to a non-empty drive.google.com URL.
        for i in range(1, 54):
            assert i in links
            assert links[i].startswith("https://drive.google.com/")
        # Spot-check that LF 7's last short (global 53) maps to the LF-7 URL,
        # NOT to LF 1's last short (which would happen under last-write-wins).
        assert "lf7_short7" in links[53]
        assert "lf1_short8" in links[8]


# ── Error / edge paths ───────────────────────────────────────────────────


class TestLengthMismatchAndEdge:
    def test_length_mismatch_logs_error_and_skips_lf(self, caplog):
        """3 posts vs 2 manifest entries → ERROR, no exception, LF skipped.

        The "skip" rule preserves the OTHER LFs' correct mappings.
        """
        social_posts = {
            1: [_FakePost(1), _FakePost(2), _FakePost(3)],
            2: [_FakePost(4)],
        }
        manifests = {
            1: _mk_manifest(
                ("short_01.mp4", "https://drive.google.com/lf1_a"),
                ("short_02.mp4", "https://drive.google.com/lf1_b"),
            ),
            2: _mk_manifest(
                ("short_01.mp4", "https://drive.google.com/lf2_a"),
            ),
        }

        with caplog.at_level(logging.ERROR):
            links = _build_drive_links_by_short_index(
                social_posts_per_lf=social_posts,
                upload_manifests=manifests,
            )

        # LF 1 is skipped due to mismatch; LF 2 maps cleanly.
        assert links == {4: "https://drive.google.com/lf2_a"}
        assert any(
            "length mismatch" in rec.message.lower() and "1" in rec.message
            for rec in caplog.records
        )

    def test_empty_manifest_for_lf_is_noop(self):
        """LF with empty manifest contributes nothing, no crash."""
        social_posts = {
            1: [_FakePost(1), _FakePost(2)],
            2: [_FakePost(3)],
        }
        manifests = {
            1: _mk_manifest(),  # empty
            2: _mk_manifest(("short_01.mp4", "https://drive.google.com/lf2_a")),
        }

        links = _build_drive_links_by_short_index(
            social_posts_per_lf=social_posts,
            upload_manifests=manifests,
        )

        assert links == {3: "https://drive.google.com/lf2_a"}

    def test_missing_social_posts_for_lf_logs_error_and_skips(self, caplog):
        """Manifest present but no posts entry → ERROR, no crash."""
        social_posts = {1: [_FakePost(1)]}
        manifests = {
            1: _mk_manifest(("short_01.mp4", "https://drive.google.com/lf1_a")),
            2: _mk_manifest(("short_01.mp4", "https://drive.google.com/lf2_a")),
        }

        with caplog.at_level(logging.ERROR):
            links = _build_drive_links_by_short_index(
                social_posts_per_lf=social_posts,
                upload_manifests=manifests,
            )

        assert links == {1: "https://drive.google.com/lf1_a"}
        assert any("social_posts_per_lf" in rec.message for rec in caplog.records)

    def test_manifest_with_empty_files_returns_empty(self):
        """``UploadManifest.files = {}`` (the contract shape for "no
        uploads recorded") → no entries, no crash. Post-contract, any
        non-``UploadManifest`` value is unrepresentable from a legitimate
        code path (``PipelineState.from_dict`` rehydrates), so we test the
        empty-files edge instead of an arbitrary attr-missing object.
        """
        empty = _mk_manifest()  # UploadManifest with files={}

        links = _build_drive_links_by_short_index(
            social_posts_per_lf={1: [_FakePost(1)]},
            upload_manifests={1: empty},
        )
        assert links == {}

    def test_post_missing_short_index_attr_skips_entry(self, caplog):
        """A post without ``short_index`` is skipped per-entry, others survive."""

        class _BadPost:
            pass

        social_posts = {
            1: [_BadPost(), _FakePost(2)],
        }
        manifests = {
            1: _mk_manifest(
                ("short_01.mp4", "https://drive.google.com/lf1_a"),
                ("short_02.mp4", "https://drive.google.com/lf1_b"),
            ),
        }

        with caplog.at_level(logging.ERROR):
            links = _build_drive_links_by_short_index(
                social_posts_per_lf=social_posts,
                upload_manifests=manifests,
            )

        # First entry skipped; second entry mapped.
        assert links == {2: "https://drive.google.com/lf1_b"}


# ── Regression sentinel ─────────────────────────────────────────────────


def test_no_filename_parsing_in_merge():
    """Sanity: filenames are irrelevant. Manifests with bogus filenames
    must still produce correct mappings as long as insertion order matches.
    """
    social_posts = {1: [_FakePost(42), _FakePost(43)]}
    manifests = {
        1: _mk_manifest(
            ("absolute_garbage_filename.mp4", "https://drive.google.com/x"),
            ("zzz_no_index_at_all.mov", "https://drive.google.com/y"),
        ),
    }

    links = _build_drive_links_by_short_index(
        social_posts_per_lf=social_posts,
        upload_manifests=manifests,
    )

    assert links == {
        42: "https://drive.google.com/x",
        43: "https://drive.google.com/y",
    }


# ── Resume / JSON-round-trip dual-shape ────────────


class TestManifestContractAfterStateResume:
    """JSON round-trips
    through ``pipeline_state.json`` rehydrate ``upload_manifests`` values
    to ``UploadManifest`` instances at ``PipelineState.from_dict`` time.
    The dict-shape pathway that once broke the merge is now
    unrepresentable from a legitimate code path; ``_zip_per_lf_drive_links``
    assumes the contract and uses attribute access directly.

    These tests pin that contract at the load boundary, not inside the
    merge helper (which the previous shim-era tests did, encoding the
    bug).
    """

    def test_dataclass_shape_maps_correctly(self):
        """Live, in-process manifests are dataclass-shaped → merge works."""
        social_posts = {1: [_FakePost(1), _FakePost(2)]}
        manifests = {
            1: _mk_manifest(
                ("short_01.mp4", "https://drive.google.com/file/d/a"),
                ("short_02.mp4", "https://drive.google.com/file/d/b"),
            ),
        }

        links = _build_drive_links_by_short_index(
            social_posts_per_lf=social_posts,
            upload_manifests=manifests,
        )

        assert links == {
            1: "https://drive.google.com/file/d/a",
            2: "https://drive.google.com/file/d/b",
        }

    def test_pipelinestate_roundtrip_rehydrates_manifests_to_dataclass(self):
        """End-to-end contract: build a real ``UploadManifest``, stash it
        on ``PipelineState``, ``to_dict`` → JSON → ``from_dict``, then run
        the rehydrated manifests through the merge. Asserts the manifests
        come back as dataclass instances (not dicts) and the merge produces
        the expected mapping.

        Regression coverage for the dict-shape resume bug — the
        bug class is now closed at the contract boundary.
        """
        import json

        from gomoveshift.pipeline.drive_state_model import PipelineState
        from gomoveshift.pipeline.uploader import UploadManifest

        live = UploadManifest(
            project_name="synthetic_lf01",
            folder_id="fld",
            files={
                "short_01.mp4": "https://drive.google.com/file/d/A",
                "short_02.mp4": "https://drive.google.com/file/d/B",
                "short_03.mp4": "https://drive.google.com/file/d/C",
            },
            completed=True,
        )
        st = PipelineState()
        st.upload_manifests = {1: live}

        # Full JSON round trip through PipelineState.
        on_disk = json.loads(json.dumps(st.to_dict()))
        reloaded = PipelineState.from_dict(on_disk)

        assert reloaded.upload_manifests is not None
        rehydrated = reloaded.upload_manifests[1]
        assert isinstance(rehydrated, UploadManifest), (
            f"Contract: PipelineState.from_dict must rehydrate "
            f"upload_manifests values to UploadManifest; got {type(rehydrated)!r}"
        )
        assert rehydrated.files == live.files

        social_posts = {1: [_FakePost(10), _FakePost(11), _FakePost(12)]}
        links = _build_drive_links_by_short_index(
            social_posts_per_lf=social_posts,
            upload_manifests=reloaded.upload_manifests,
        )

        assert links == {
            10: "https://drive.google.com/file/d/A",
            11: "https://drive.google.com/file/d/B",
            12: "https://drive.google.com/file/d/C",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
