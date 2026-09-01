"""CLI entry point for the Business Idea Evaluator."""

from langchain_core.messages import HumanMessage

from .graph import build_graph
from .state import State


def run() -> None:
    graph = build_graph()

    print("What is your business idea?")
    idea = input("You: ")

    init_state: State = {
        "idea": idea,
        "messages": [HumanMessage(content=idea)],
        "advisor_reports": {},
        "final_report": "",
    }

    result = graph.invoke(
        init_state, config={"configurable": {"thread_id": "run-1"}}
    )

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60 + "\n")
    print(result["final_report"])


if __name__ == "__main__":
    run()
