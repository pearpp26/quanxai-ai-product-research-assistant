from __future__ import annotations

import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "products_catalog"
EMBED_MODEL = "embed-english-v3.0"

_qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
_cohere = cohere.ClientV2(api_key=os.getenv("COHERE_API"))


def _format_hit(rank: int, payload: dict, score: float) -> str:
    description = payload.get("description", "") or ""
    excerpt = description if len(description) <= 200 else description[:197] + "..."
    return (
        f"{rank}. {payload.get('product_name', '?')} "
        f"({payload.get('product_id', '?')})\n"
        f"   Category: {payload.get('category', '?')} | "
        f"Brand: {payload.get('brand', '?')}\n"
        f"   Price: ${payload.get('current_price', 0):.2f} | "
        f"Rating: {payload.get('average_rating', 0)} "
        f"({payload.get('review_count', 0)} reviews) | "
        f"Stock: {payload.get('stock_quantity', 0)}\n"
        f"   Description: {excerpt}\n"
        f"   Relevance: {score:.3f}"
    )


def search_product_catalog(query: str, limit: int = 5) -> str:
    """Embed `query`, search Qdrant, return a formatted string for the LLM."""
    response = _cohere.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    query_vector = response.embeddings.float_[0]

    result = _qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    if not result.points:
        return "No matching products found."

    blocks = [
        _format_hit(rank, point.payload or {}, point.score)
        for rank, point in enumerate(result.points, start=1)
    ]
    return "\n\n".join(blocks)
