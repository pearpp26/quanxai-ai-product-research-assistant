from __future__ import annotations

import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_tavily = TavilyClient(api_key=os.getenv("TAVILY_API"))


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web via Tavily; return a formatted string for the LLM."""
    try:
        response = _tavily.search(
            query,
            max_results=max_results,
            search_depth="basic",
        )
    except Exception:
        return "Web search is currently unavailable. Please rely on local knowledge."

    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        return "No web results found."

    blocks = [
        f"{i}. {r.get('title', '?')}\n"
        f"   URL: {r.get('url', '?')}\n"
        f"   {r.get('content', '')}\n"
        for i, r in enumerate(results, start=1)
    ]
    return "\n".join(blocks)
