"""Specialized advisor nodes.

Each advisor evaluates the business idea from a distinct professional angle.
They all share the same structure and only differ in their system prompt, so
they can be generated from a single template and run in parallel.
"""

from langchain_core.messages import SystemMessage

from .llm import get_llm
from .state import State

# Role name -> role-specific instructions injected into the prompt.
ADVISOR_PROMPTS: dict[str, str] = {
    "Market Analyst": (
        "You are a senior MARKET ANALYST. Evaluate the market potential, "
        "competition, target demographics, and trends:\n"
        "- Conduct market sizing and competitor research\n"
        "- Identify target customers and segments\n"
        "- Assess timing, trends, and macroeconomic influences"
    ),
    "Legal Advisor": (
        "You are a LEGAL ADVISOR. For the idea below:\n"
        "- Identify IP, licensing, and trademark needs\n"
        "- Spot compliance issues (e.g., GDPR, financial regulations)\n"
        "- Evaluate contract and partnership considerations"
    ),
    "Technical Advisor": (
        "You are a TECHNICAL / PRODUCT FEASIBILITY EXPERT. For the idea below:\n"
        "- Estimate development complexity and time\n"
        "- Recommend tech stacks or platforms\n"
        "- Evaluate risk in terms of infrastructure, scalability, and cost"
    ),
    "Strategist Advisor": (
        "You are a STRATEGIST ADVISOR. For the idea below:\n"
        "- Define launch milestones\n"
        "- Select distribution channels and positioning strategies\n"
        "- Craft early traction tactics (e.g., community, influencers, PR)"
    ),
}


def _run_advisor(role: str, state: State) -> dict:
    """Run a single advisor and return its report keyed by role name."""
    prompt = (
        f"{ADVISOR_PROMPTS[role]}\n\n"
        f"Idea / conversation so far:\n{state['messages']}"
    )
    report = get_llm().invoke([SystemMessage(content=prompt)])
    return {"advisor_reports": {role: report.content}}


def market_analyst_advisor(state: State) -> dict:
    return _run_advisor("Market Analyst", state)


def legal_advisor(state: State) -> dict:
    return _run_advisor("Legal Advisor", state)


def technical_advisor(state: State) -> dict:
    return _run_advisor("Technical Advisor", state)


def strategist_advisor(state: State) -> dict:
    return _run_advisor("Strategist Advisor", state)
