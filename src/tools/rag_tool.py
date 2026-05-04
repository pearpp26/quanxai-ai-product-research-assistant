from __future__ import annotations

import os

import cohere
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "products_catalog"
EMBED_MODEL = "embed-english-v3.0"
OVER_FETCH_MULTIPLIER = 3  # fetch limit*N candidates so chunk dedup still fills top-k

_qdrant = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
_cohere = cohere.ClientV2(api_key=os.getenv("COHERE_API"))


def _hit_to_dict(payload: dict, score: float) -> dict:
    return {
        "product_id": payload.get("product_id"),
        "product_name": payload.get("product_name"),
        "category": payload.get("category"),
        "brand": payload.get("brand"),
        "current_price": payload.get("current_price"),
        "average_rating": payload.get("average_rating"),
        "review_count": payload.get("review_count"),
        "stock_quantity": payload.get("stock_quantity"),
        "description": payload.get("description"),
        "matching_chunk": payload.get("chunk_text"),
        "chunk_index": payload.get("chunk_index"),
        "total_chunks": payload.get("total_chunks"),
        "relevance_score": round(float(score), 3),
    }


def search_product_catalog(query: str, limit: int = 5) -> dict:
    """Embed `query`, search Qdrant chunks, dedupe by product_id, and return
    a structured result dict with up to `limit` unique products."""
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
        limit=limit * OVER_FETCH_MULTIPLIER,
        with_payload=True,
    )

    if not result.points:
        return {
            "answer": "No matching products found.",
            "products": [],
            "count": 0,
            "sources": ["qdrant:products_catalog"],
        }

    # Dedupe by product_id, keeping the highest-scoring chunk per product.
    best_per_product: dict[str, tuple[dict, float]] = {}
    for point in result.points:
        payload = point.payload or {}
        pid = payload.get("product_id")
        if pid is None:
            continue
        existing = best_per_product.get(pid)
        if existing is None or point.score > existing[1]:
            best_per_product[pid] = (payload, point.score)

    ranked = sorted(best_per_product.values(), key=lambda x: x[1], reverse=True)[:limit]
    products = [_hit_to_dict(payload, score) for payload, score in ranked]

    top_names = ", ".join(p["product_name"] for p in products[:3] if p["product_name"])
    answer = f"Found {len(products)} matching products. Top: {top_names}."

    return {
        "answer": answer,
        "products": products,
        "count": len(products),
        "sources": ["qdrant:products_catalog"],
    }
