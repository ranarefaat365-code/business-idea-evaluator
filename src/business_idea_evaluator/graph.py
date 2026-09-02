"""Graph wiring: human-in-the-loop clarification + parallel advisors."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .advisors import (
    legal_advisor,
    market_analyst_advisor,
    strategist_advisor,
    technical_advisor,
)
from .llm import get_llm
from .state import State

BASE_SYSTEM_MSG = SystemMessage(
    content=(
        "You are a helpful assistant.\n"
        "Your job: decide whether you have enough information about the "
        "start-up idea.\n"
        "If not, ask ONE precise follow-up question.\n"
        "If yes, respond with exactly: DONE"
    )
)

_REQUIRED_ADVISORS = 4


def _as_text(content) -> str:
    """Normalize message content that may be a string or a list of parts."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", ""))
        return "".join(parts)
    return content or ""


def decide_node(state: State) -> dict:
    conversation = [BASE_SYSTEM_MSG] + state["messages"]
    ai_reply: AIMessage = get_llm().invoke(conversation)
    return {"messages": [ai_reply]}


def route(state: State) -> str:
    last_ai_msg: AIMessage = state["messages"][-1]
    if _as_text(last_ai_msg.content).strip().upper().startswith("DONE"):
        return "fanout"
    return "ask_user_node"


def ask_user_node(state: State) -> dict:
    question = _as_text(state["messages"][-1].content)
    print(f"\nAssistant: {question}\n")
    human = input("You: ")
    return {"messages": [HumanMessage(content=human)]}


def collect_and_report(state: State) -> dict:
    if len(state["advisor_reports"]) < _REQUIRED_ADVISORS:
        return {}
    report_prompt = (
        "You are a senior consultant. Combine the advisor notes below into "
        "one clear, structured evaluation report for the founder.\n\n"
        f"{state['advisor_reports']}"
    )
    report = _as_text(get_llm().invoke(report_prompt).content)
    return {"final_report": report}


def build_graph():
    builder = StateGraph(State)

    builder.add_node("decide_node", decide_node)
    builder.add_node("ask_user_node", ask_user_node)
    builder.add_node("fanout", lambda state: {})
    builder.add_node("market_analyst_advisor", market_analyst_advisor)
    builder.add_node("legal_advisor", legal_advisor)
    builder.add_node("technical_advisor", technical_advisor)
    builder.add_node("strategist_advisor", strategist_advisor)
    builder.add_node("collect_and_report", collect_and_report)

    builder.set_entry_point("decide_node")
    builder.add_edge("ask_user_node", "decide_node")
    builder.add_conditional_edges(
        "decide_node",
        route,
        {"ask_user_node": "ask_user_node", "fanout": "fanout"},
    )

    for advisor in (
        "market_analyst_advisor",
        "legal_advisor",
        "technical_advisor",
        "strategist_advisor",
    ):
        builder.add_edge("fanout", advisor)
        builder.add_edge(advisor, "collect_and_report")

    builder.add_edge("collect_and_report", END)

    return builder.compile()