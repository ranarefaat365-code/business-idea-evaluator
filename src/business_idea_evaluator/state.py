"""Graph state definition for the Business Idea Evaluator."""

import operator
from typing import Annotated, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    """Shared state that flows through the LangGraph pipeline.

    Attributes:
        idea: The raw business idea provided by the founder.
        messages: Running conversation history. ``add_messages`` appends new
            messages instead of overwriting the list.
        advisor_reports: One report per advisor. ``operator.or_`` merges the
            dictionaries produced by advisors running in parallel, so no
            report overwrites another.
        final_report: The consolidated report produced at the end of the run.
    """

    idea: str
    messages: Annotated[List[BaseMessage], add_messages]
    advisor_reports: Annotated[Dict[str, str], operator.or_]
    final_report: str
