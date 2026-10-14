"""Photo upload verification, re-upload and hard-fail.

``upload_photos`` only uploads files absent from the on-disk manifest, so a
re-upload that leaves the failed entries in place is a no-op that ships a dead
Drive link. An unverifiable batch must fail loudly, releasing its photo holds.
FakeUploader reproduces that resume semantics on purpose — a mock that just
returns a manifest for whatever it is handed cannot show the bug.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.photos.caption_generator import CaptionBatch, PhotoCaption
from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator
from gomoveshift.publishing.photo_run_hooks import photo_run_hooks
from gomoveshift.pipeline.photos.scheduler import PhotoSlot
from gomoveshift.pipeline.photos.selector import PhotoSelectionState
from gomoveshift.pipeline.uploader import UploadManifest, VerificationResult

WEEK = "2026-02-23"
SLUG = "test-journey"


def _site(tmp_path, n=2):
    root = tmp_path / "site"
    (root / "content" / "journeys").mkdir(parents=True)
    images = root / "static" / "images" / "journeys" / SLUG
    images.mkdir(parents=True)
    names = [f"photo-{i:03d}.jpg" for i in range(1, n + 1)]
    for name in names:
        (images / name).write_bytes(b"x")
    (root / "content" / "journeys" / f"{SLUG}.md").write_text(
        "---\ntitle: Test Journey\ndays:\n- day_number: 1\n  date: '2025-01-01'\n"
        "  summary: Test day\n  locations: [TestPlace]\n  photos:\n"
        f"  - location: TestPlace\n    photos: [{', '.join(names)}]\n---\n"
    )
    return root


def _captions(tmp_path, n=2):
    batch = CaptionBatch(
        week_start=WEEK,
        voice_style="voice_first_person",
        captions=[
            PhotoCaption(i, f"Caption {i}", "T", ["t"], SLUG, f"photo-{i + 1:03d}.jpg", "TestPlace")
            for i in range(n)
        ],
        reviewed=True,
    )
    batch.save(tmp_path / "captions.yaml")
    return str(tmp_path / "captions.yaml")


def _resized(index):
    """Uploaded filename for photo ``index`` (resizer prefixes the slug)."""
    return f"{SLUG}_photo-{index:03d}.jpg"


class FakeUploader:
    """Mimics PipelineUploader's resume semantics against a real manifest file."""

    def __init__(self, verify_script):
        self.uploaded = []
        self.verify_script = list(verify_script)
        self.verify_calls = 0

    def upload_photos(self, paths, manifest_path=None, **kwargs):
        path = (
            Path(manifest_path)
            if manifest_path
            else Path(paths[0]).parent / "photo_gdrive_manifest.json"
        )
        manifest = UploadManifest.load(path) if path.exists() else UploadManifest(project_name="t")
        for p in paths:
            name = Path(p).name
            if name in manifest.files:
                continue  # the resume filter that made the old re-upload a no-op
            self.uploaded.append(name)
            manifest.files[name] = f"https://drive.example/d/{name}/view"
        manifest.completed = True
        manifest.save(path)
        return manifest

    def verify_uploads(self, manifest):
        failed = (
            self.verify_script[self.verify_calls]
            if self.verify_calls < len(self.verify_script)
            else []
        )
        self.verify_calls += 1
        failed = [f for f in failed if f in manifest.files]
        return VerificationResult(
            verified=[f for f in manifest.files if f not in failed], failed=failed
        )


@pytest.fixture
def make_env(tmp_path):
    def _make(verify_script=()):
        site = _site(tmp_path)
        stack = [
            patch("gomoveshift.pipeline.photos.orchestrator.PipelineUploader"),
            patch("gomoveshift.publishing.photo_run_hooks.record_confirmed_upload"),
        ]
        up_cls, _book = [s.start() for s in stack]
        uploader = FakeUploader(verify_script)
        up_cls.return_value = uploader

        orch = PhotoOrchestrator(site_root=site, output_dir=tmp_path, **photo_run_hooks())
        (tmp_path / "resized").mkdir(exist_ok=True)

        def _resize(paths, week, journey_slugs=None):
            out = [
                tmp_path / "resized" / f"{slug}_{Path(p).name}"
                for p, slug in zip(paths, journey_slugs or [SLUG] * len(paths))
            ]
            for dest in out:
                dest.write_bytes(b"x")
            return out

        orch.resizer.resize_batch = MagicMock(side_effect=_resize)
        orch.scheduler.schedule_batch = MagicMock(
            return_value=[
                PhotoSlot(date=d, time="10:00:00", day_of_week="Monday")
                for d in ("2026-02-23", "2026-02-24")
            ]
        )
        return SimpleNamespace(orch=orch, uploader=uploader, tmp=tmp_path, stack=stack)

    envs = []

    def _tracked(verify_script=()):
        env = _make(verify_script)
        envs.append(env)
        return env

    yield _tracked
    for env in envs:
        for s in env.stack:
            s.stop()


def _held(env):
    return set(PhotoSelectionState.load(env.orch.selector.state_path).pending_photos)


def test_failed_verification_actually_reuploads(make_env):
    """The failed file must be purged and genuinely re-uploaded, not no-op'd."""
    env = make_env([[_resized(1)], []])
    env.orch.publish_batch(_captions(env.tmp))

    assert env.uploader.uploaded.count(_resized(1)) == 2, (
        f"failed file was never re-uploaded: {env.uploader.uploaded}"
    )
    assert env.uploader.uploaded.count(_resized(2)) == 1


def test_second_verification_failure_hard_fails(make_env):
    """A batch that still will not verify must raise, not report success."""
    env = make_env([[_resized(1)], [_resized(1)]])
    yaml_path = _captions(env.tmp)

    with pytest.raises(Exception, match="verif"):
        env.orch.publish_batch(yaml_path)

    assert not CaptionBatch.load(Path(yaml_path)).published
    assert not list((env.tmp / "photo_metricool").glob("*.csv"))


def test_hard_fail_releases_holds(make_env):
    """An unverifiable batch leaves its photos re-selectable (hold model)."""
    env = make_env([[_resized(1)], [_resized(1)]])

    with pytest.raises(Exception):
        env.orch.publish_batch(_captions(env.tmp))

    assert _held(env) == set()


def test_clean_verification_uploads_each_photo_once(make_env):
    """Happy path: no purge, no re-upload."""
    env = make_env([[]])
    env.orch.publish_batch(_captions(env.tmp))

    assert sorted(env.uploader.uploaded) == [_resized(1), _resized(2)]
    assert env.uploader.verify_calls == 1


def test_genuine_resume_does_not_reupload(make_env):
    """Republishing an already-uploaded batch must stay a no-op (no over-purge)."""
    env = make_env([[], []])
    yaml_path = _captions(env.tmp)

    env.orch.publish_batch(yaml_path)
    env.orch.publish_batch(yaml_path)

    assert sorted(env.uploader.uploaded) == [_resized(1), _resized(2)]


def test_orchestrator_has_no_dead_llm_endpoint_param():
    """The parameter had no callers and was never stored or read."""
    assert "llm_endpoint" not in inspect.signature(PhotoOrchestrator.__init__).parameters
