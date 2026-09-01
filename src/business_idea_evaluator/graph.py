"""Graph wiring: human-in-the-loop clarification + parallel advisors.

Flow:
    decide_node --(needs info)--> ask_user_node --> decide_node
    decide_node --(DONE)--> fanout --> [4 advisors in parallel] --> collect_and_report --> END
"""

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

# Minimum advisor reports required before the final report is generated.
_REQUIRED_ADVISORS = 4


def decide_node(state: State) -> dict:
    """Ask the model whether more context is needed, or signal DONE."""
    conversation = [BASE_SYSTEM_MSG] + state["messages"]
    ai_reply: AIMessage = get_llm().invoke(conversation)
    return {"messages": [ai_reply]}


def route(state: State) -> str:
    """Branch based on whether the decider produced DONE."""
    last_ai_msg: AIMessage = state["messages"][-1]
    if last_ai_msg.content.strip().upper().startswith("DONE"):
        return "fanout"
    return "ask_user_node"


def ask_user_node(state: State) -> dict:
    """Human-in-the-loop: print the assistant's question and read the reply.

    Returns only the new human message; ``add_messages`` appends it to the
    running conversation automatically.
    """
    question = state["messages"][-1].content
    print(f"\nAssistant: {question}\n")
    human = input("You: ")
    return {"messages": [HumanMessage(content=human)]}


def collect_and_report(state: State) -> dict:
    """Merge the advisor reports into one structured final report."""
    if len(state["advisor_reports"]) < _REQUIRED_ADVISORS:
        return {}
    report_prompt = (
        "You are a senior consultant. Combine the advisor notes below into "
        "one clear, structured evaluation report for the founder.\n\n"
        f"{state['advisor_reports']}"
    )
    report = get_llm().invoke(report_prompt).content
    return {"final_report": report}


def build_graph():
    """Assemble and compile the LangGraph pipeline."""
    builder = StateGraph(State)

    builder.add_node("decide_node", decide_node)
    builder.add_node("ask_user_node", ask_user_node)
    builder.add_node("fanout", lambda state: {})  # empty hub for parallel branch
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
