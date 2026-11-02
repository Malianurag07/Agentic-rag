"""Agentic RAG pipeline: hierarchical retrieval (topic > type > text) controlled by an agent loop.

See Agentic_RAG_Pipeline_Plan.pdf for the design this package implements.
"""

from .config import Config
from .pipeline import Pipeline, Result
from .store import Store

__all__ = ["Config", "Pipeline", "Result", "Store"]
