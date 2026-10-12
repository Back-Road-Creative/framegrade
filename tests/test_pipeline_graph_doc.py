"""Drift test: the committed pipeline-graph doc must equal the generated one.

Fails if ``STAGE_ORDER`` or ``PHASE_SPECS`` change without regenerating
``docs/pipeline-graph.md`` via ``python -m gomoveshift.pipeline.graph_doc``.
"""

from pathlib import Path

from gomoveshift.pipeline.graph_doc import render_pipeline_graph

# Anchor to the committed doc in the SOURCE tree (this test's own repo
# location), NOT the installed package: CI installs non-editable, so
# graph_doc's package-relative DOC_PATH resolves into site-packages/ where the
# doc was never shipped. The test's job is to find the artifact in the repo.
DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "pipeline-graph.md"


def test_committed_graph_matches_generated():
    assert DOC_PATH.exists(), f"{DOC_PATH} missing — run: python -m gomoveshift.pipeline.graph_doc"
    assert render_pipeline_graph() == DOC_PATH.read_text(), (
        "docs/pipeline-graph.md is stale — regenerate with "
        "`python -m gomoveshift.pipeline.graph_doc`"
    )
