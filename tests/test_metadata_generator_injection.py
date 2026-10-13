"""The metadata stages take the long-form generator from the run context.

``pipeline.orchestrator`` and the metadata / analytics_sync / chapters stages no
longer import ``longform_metadata``. The caller passes the class to
``PipelineOrchestrator(config, metadata_generator=...)``; the orchestrator puts it on
``PipelineContext.metadata_generator`` and the stages read it from there.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.context import PipelineContext
from gomoveshift.pipeline.orchestrator import PipelineOrchestrator


def test_context_without_generator_raises_a_pointed_error():
    ctx = PipelineContext.__new__(PipelineContext)
    ctx.metadata_generator = None
    with pytest.raises(RuntimeError, match="metadata_generator"):
        ctx.require_metadata_generator()


def test_context_returns_the_injected_generator():
    ctx = PipelineContext.__new__(PipelineContext)
    gen = MagicMock()
    ctx.metadata_generator = gen
    assert ctx.require_metadata_generator() is gen


def test_orchestrator_stores_the_generator_and_defaults_to_none():
    sig = inspect.signature(PipelineOrchestrator.__init__)
    assert sig.parameters["metadata_generator"].default is None
    gen = MagicMock()
    assert PipelineOrchestrator(metadata_generator=gen).metadata_generator is gen
    assert PipelineOrchestrator().metadata_generator is None


def test_engine_modules_do_not_import_longform_metadata():
    import gomoveshift.pipeline.orchestrator as orch
    import gomoveshift.pipeline.stages.analytics_sync as a
    import gomoveshift.pipeline.stages.chapters as c
    import gomoveshift.pipeline.stages.metadata as m

    for mod in (orch, a, c, m):
        assert "longform_metadata" not in inspect.getsource(mod), mod.__name__
