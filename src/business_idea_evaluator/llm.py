"""LLM factory.

Centralizes model creation so the whole project shares one configuration and
the API key is loaded from the environment (never hard-coded).
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a shared ChatOpenAI instance.

    Raises:
        RuntimeError: if ``OPENAI_API_KEY`` is not set.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
