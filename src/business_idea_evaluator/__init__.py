"""Business Idea Evaluator: a human-in-the-loop, parallelized multi-agent
system built with LangGraph."""

from .graph import build_graph
from .state import State

__all__ = ["build_graph", "State"]
__version__ = "0.1.0"
