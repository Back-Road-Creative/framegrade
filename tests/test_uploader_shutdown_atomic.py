"""Regression tests for the Drive uploader.

``_next_chunk_with_timeout`` must surface the TimeoutError promptly even
when the stalled ``next_chunk()`` worker never returns — the ``with``-managed
pool blocked in ``__exit__``'s ``shutdown(wait=True)`` forever.

``UploadManifest.save`` must be atomic (a failed write never leaves a
partial/truncated file visible) and ``UploadManifest.load`` must tolerate a
truncated/corrupt manifest without raising (resume must not crash).
"""

import threading
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.uploader import PipelineUploader, UploadManifest


class TestNextChunkTimeoutNonBlocking:
    """A stalled next_chunk() must not wedge the caller in shutdown."""

    def test_timeout_propagates_without_blocking_on_stalled_worker(self):
        release = threading.Event()  # never set → worker never returns

        def stalled_chunk():
            release.wait()
            return None, {"id": "x"}

        mock_request = MagicMock()
        mock_request.next_chunk = stalled_chunk

        outcome = {}

        def run():
            try:
                PipelineUploader._next_chunk_with_timeout(mock_request, timeout=0.2)
            except BaseException as e:  # noqa: BLE001 — capture whatever surfaces
                outcome["exc"] = e

        t = threading.Thread(target=run, daemon=True)
        t.start()
        # Generous vs the 0.2s timeout: on the buggy `with`-managed pool the
        # call blocks in __exit__'s shutdown(wait=True) and never finishes.
        t.join(timeout=5)
        try:
            assert not t.is_alive(), "call blocked despite timeout (shutdown(wait=True) hang)"
            assert isinstance(outcome.get("exc"), TimeoutError)
        finally:
            release.set()


class TestManifestAtomicSaveTolerantLoad:
    """Atomic save + corrupt-tolerant load."""

    def test_load_tolerates_truncated_manifest(self, tmp_path):
        path = tmp_path / "manifest.json"
        path.write_text('{"project_name": "p", "files": {"a.mp4": "lin')  # truncated JSON

        m = UploadManifest.load(path)  # must NOT raise

        assert m.files == {}
        assert m.completed is False

    def test_save_failure_preserves_previous_manifest(self, tmp_path, monkeypatch):
        path = tmp_path / "manifest.json"
        first = UploadManifest(project_name="p", files={"a.mp4": "link_a"})
        first.save(path)
        original = path.read_text()

        def boom(*_a, **_k):
            raise RuntimeError("disk full mid-write")

        monkeypatch.setattr("json.dump", boom)
        first.files["b.mp4"] = "link_b"
        with pytest.raises(RuntimeError):
            first.save(path)

        # No partial/truncated file visible — the prior manifest is intact.
        assert path.read_text() == original
        recovered = UploadManifest.load(path)
        assert recovered.files == {"a.mp4": "link_a"}
