from __future__ import annotations

import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

_tavily = TavilyClient(api_key=os.getenv("TAVILY_API"))


def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web via Tavily; return a structured result dict."""
    try:
        response = _tavily.search(
            query,
            max_results=max_results,
            search_depth="basic",
        )
    except Exception as e:
        return {
            "answer": "Web search is currently unavailable. Please rely on local knowledge.",
            "results": [],
            "count": 0,
            "error": str(e),
            "sources": [],
        }

    raw = response.get("results", []) if isinstance(response, dict) else []
    if not raw:
        return {
            "answer": "No web results found.",
            "results": [],
            "count": 0,
            "sources": [],
        }

    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in raw
    ]
    titles = "; ".join(r["title"] for r in results[:3] if r["title"])
    answer = f"Found {len(results)} web results. Top: {titles}."

    return {
        "answer": answer,
        "results": results,
        "count": len(results),
        "sources": [r["url"] for r in results if r["url"]],
    }
