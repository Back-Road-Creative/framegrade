"""
TDD tests for CLI ↔ Orchestrator contract.

Regression: a run failed with "PipelineOrchestrator.run() got an
unexpected keyword argument 'skip_d_copy'" because the CLI passed kwargs
that the orchestrator didn't accept.

These tests ensure:
1. Every CLI flag maps to a valid RunOptions.from_legacy() parameter
2. The console script entry point is functional
3. The editable install is properly configured
"""

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

# ── CLI ↔ Orchestrator Parameter Contract ────────────────────────────────


class TestCliOrchestratorContract:
    """Every CLI argument that gets passed to RunOptions.from_legacy() must be accepted."""

    def test_cli_kwargs_subset_of_from_legacy_params(self):
        """All kwargs passed from cli/handlers/pipeline.py to RunOptions.from_legacy() must exist."""
        from gomoveshift.cli.handlers import pipeline as pipeline_handler
        from gomoveshift.pipeline.orchestrator import RunOptions

        from_legacy_params = set(inspect.signature(RunOptions.from_legacy).parameters.keys())
        from_legacy_params.discard("cls")

        # Parse pipeline handler source to find all kwargs passed to RunOptions.from_legacy()
        source = inspect.getsource(pipeline_handler)

        import re

        run_call_match = re.search(r"RunOptions\.from_legacy\((.*?)\)", source, re.DOTALL)
        assert run_call_match, (
            "Could not find RunOptions.from_legacy() call in cli/handlers/pipeline.py"
        )

        call_body = run_call_match.group(1)
        kwarg_names = set()
        for match in re.finditer(r"(\w+)\s*=", call_body):
            kwarg_names.add(match.group(1))

        missing = kwarg_names - from_legacy_params
        assert not missing, (
            f"CLI passes kwargs not accepted by RunOptions.from_legacy(): {missing}. "
            f"Available params: {sorted(from_legacy_params)}"
        )

    def test_run_options_required_fields(self):
        """RunOptions required fields: video_path, location, publish_day."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        sig = inspect.signature(RunOptions)
        required = []
        for name, param in sig.parameters.items():
            if param.default is inspect.Parameter.empty:
                required.append(name)

        assert set(required) == {"video_path", "location", "publish_day"}, (
            f"Unexpected required params: {required}. "
            f"Only video_path, location, and publish_day should be required."
        )

    def test_skip_d_copy_exists_in_from_legacy(self):
        """Regression: skip_d_copy must exist in RunOptions.from_legacy()."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        params = inspect.signature(RunOptions.from_legacy).parameters
        assert "skip_d_copy" in params, (
            "skip_d_copy missing from RunOptions.from_legacy() — this caused a run to fail"
        )

    def test_draft_param_exists_in_run_options(self):
        """Regression: draft must exist in RunOptions."""
        from gomoveshift.pipeline.orchestrator import RunOptions

        params = inspect.signature(RunOptions).parameters
        assert "draft" in params, "draft param missing from RunOptions"

    def test_run_accepts_run_options(self):
        """orchestrator.run() must accept a RunOptions instance."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        params = inspect.signature(PipelineOrchestrator.run).parameters
        assert "options" in params, "run() must accept 'options' parameter"

    def test_legacy_kwargs_still_accepted(self):
        """orchestrator.run() must still accept legacy skip_* kwargs."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        params = inspect.signature(PipelineOrchestrator.run).parameters
        assert "skip_d_copy" in params, "Legacy skip_d_copy kwarg must still be accepted"
        assert "draft" in params, "Legacy draft kwarg must still be accepted"


# ── Package Installation ────────────────────────────────────────────────


class TestPackageInstallation:
    """The gomoveshift package must be properly importable."""

    def test_gomoveshift_importable(self):
        """Package must be importable."""
        import gomoveshift

        assert hasattr(gomoveshift, "__file__")

    def test_cli_module_importable(self):
        """CLI module must be importable."""
        from gomoveshift.cli import main

        assert callable(main)

    def test_orchestrator_importable(self):
        """Orchestrator must be importable."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        assert callable(PipelineOrchestrator)

    def test_single_pass_importable(self):
        """SinglePassProcessor must be importable."""
        from gomoveshift.pipeline.single_pass import SinglePassProcessor

        assert callable(SinglePassProcessor)

    def test_console_script_exists(self):
        """The gomoveshift console script must exist in the venv."""
        venv_bin = Path(sys.executable).parent
        script = venv_bin / "gomoveshift"
        assert script.exists(), (
            f"Console script not found at {script}. Run: pip install -e . in the package directory"
        )

    def test_console_script_help(self):
        """Console script must respond to --help without errors."""
        venv_bin = Path(sys.executable).parent
        script = venv_bin / "gomoveshift"
        if not script.exists():
            pytest.skip("Console script not found")

        # --help imports the full CLI (googleapiclient, scipy, the pipeline
        # package): 9-20 s measured on the shared self-hosted runner under
        # load, so a 10 s cap is too tight.
        # The timeout only guards against a hang; it is not a speed budget.
        # 90 s matches the other full-CLI subprocess tests
        # (test_night_product_cli_subprocess.py).
        result = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, f"Console script --help failed: {result.stderr}"
        assert "pipeline" in result.stdout.lower(), "Help should mention pipeline command"

    def test_no_stale_pyc_for_wrong_python(self):
        """No .pyc files for Python versions other than current should exist."""
        major, minor = sys.version_info[:2]
        current_tag = f"cpython-{major}{minor}"

        package_dir = Path(__file__).parent.parent / "gomoveshift"
        stale = []
        for pyc in package_dir.rglob("*.pyc"):
            if current_tag not in pyc.name:
                stale.append(str(pyc))

        assert not stale, (
            f"Stale .pyc files found for wrong Python version: {stale[:5]}... "
            f"Delete with: find gomoveshift -name '*.pyc' -delete"
        )
