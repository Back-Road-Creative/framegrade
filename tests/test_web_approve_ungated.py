"""The web /api/approve VideoAssembler render path runs no stop/audio/blur
validation, so its job must be marked EXPLICITLY ungated (never silently passed
off as gated). The marker is set synchronously in the endpoint body, before the
background render task."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from gomoveshift.config import Config
from gomoveshift.models import ProcessingJob
from gomoveshift.web import app as web_app_module
from gomoveshift.web.app import create_app


def test_approve_marks_job_explicitly_ungated(tmp_path):
    web_app_module.jobs.clear()
    web_app_module.analyses.clear()
    video_path = "clip.mp4"

    with patch.object(web_app_module, "VideoAssembler") as mock_asm:
        mock_asm.return_value.create_assembly_plan.return_value = MagicMock()
        mock_asm.return_value.execute.return_value = []
        client = TestClient(create_app(config=Config()), raise_server_exceptions=False)

        web_app_module.analyses[video_path] = MagicMock()
        web_app_module.jobs[video_path] = ProcessingJob(
            video_path=Path(video_path), status="reviewed"
        )

        resp = client.post(
            "/api/approve",
            json={"video_path": video_path, "approved_trims": [], "approved_cuts": []},
        )

    assert resp.status_code == 200, resp.text
    job = web_app_module.jobs[video_path]
    assert job.ungated_reason is not None
    assert "no stop/audio/blur" in job.ungated_reason
    web_app_module.jobs.clear()
    web_app_module.analyses.clear()
