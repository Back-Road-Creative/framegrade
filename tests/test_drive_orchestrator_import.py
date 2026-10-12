"""``drive_orchestrator`` must be importable without the repo-root ``tools/``.

``pyproject.toml`` ships only the ``gomoveshift`` package; ``tools/`` lives at
the repo root and is resolved via the source checkout (or the editable
install's path hook). A plain wheel install has no ``tools`` — observed
when a CI subprocess importing the installed copy died at
``drive_orchestrator`` module scope with ``ModuleNotFoundError: tools``.

The contract: import always succeeds; the DoD validation gate fails loudly at
CALL time with a message naming the missing repo tooling.
"""

import subprocess
import sys
import textwrap


def test_importable_and_loud_without_tools(tmp_path):
    """Simulate a wheel install by blocking ``tools`` imports in a subprocess.

    A ``sys.meta_path`` blocker raising ``ModuleNotFoundError`` for ``tools``
    reproduces the missing-module condition independent of install mode (an
    editable install resolves ``tools`` via the repo root, so simply running
    elsewhere is not enough). The subprocess then checks both halves of the
    contract: the import succeeds, and calling the gate raises the loud
    ``RuntimeError`` — not an ``AttributeError`` from a bare ``None`` stub.
    """
    code = textwrap.dedent(
        """
        import sys

        class _BlockTools:
            def find_spec(self, name, path=None, target=None):
                if name == "tools" or name.startswith("tools."):
                    raise ModuleNotFoundError(f"blocked for test: {name}")
                return None

        sys.meta_path.insert(0, _BlockTools())

        import gomoveshift.pipeline.drive_orchestrator as d

        print("IMPORT-OK")

        for attempt in (lambda: d.run_validation(), lambda: d.ValidationMode.AUTO):
            try:
                attempt()
            except RuntimeError as exc:
                assert "tools/validate_pipeline_output" in str(exc), exc
            else:
                raise AssertionError("expected a loud RuntimeError")

        print("LOUD-OK")
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "IMPORT-OK" in res.stdout
    assert "LOUD-OK" in res.stdout
