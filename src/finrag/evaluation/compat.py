"""Compatibility shims for the evaluation dependencies.

Imported for its side effect, before anything that pulls in RAGAS::

    from finrag.evaluation import compat  # noqa: F401

RAGAS 0.4.3 — the current release — imports
``langchain_community.chat_models.vertexai`` at module load. langchain-community 0.4
removed that path as part of splitting integrations into standalone packages, so importing
RAGAS raises ModuleNotFoundError on a current stack. There is no fixed RAGAS release to
move to.

Stubbing it is safe here specifically because nothing in this project uses Vertex AI: the
judge models are OpenAI and Anthropic. The stub exists only to satisfy an import RAGAS
performs eagerly and never exercises on our code paths.

Remove this when RAGAS drops the import. The test suite pins the behaviour, so the day it
becomes unnecessary the shim can go without guesswork.
"""

from __future__ import annotations

import sys
import types

VERTEX_PATH = "langchain_community.chat_models.vertexai"


def install_ragas_shim() -> bool:
    """Register a stub for the module RAGAS expects. Returns True if it was needed."""
    try:
        __import__(VERTEX_PATH)
        return False
    except ImportError:
        pass

    stub = types.ModuleType(VERTEX_PATH)
    # RAGAS only needs the name to exist at import time.
    stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[VERTEX_PATH] = stub
    return True


install_ragas_shim()
