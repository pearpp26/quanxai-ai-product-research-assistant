from __future__ import annotations

import os
from pathlib import Path

import cohere
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

"""Ingest products_catalog.csv into the Qdrant `products_catalog` collection.

Run manually: `python -m src.ingest` (or `python src/ingest.py`).

Pipeline: CSV row → chunked description → per-chunk embedding text →
Cohere embedding → Qdrant point.

Idempotent: per-chunk point IDs are derived deterministically from
`product_id` and chunk index, so re-running upserts updated chunks in
place. Hard-deletes (rows removed from the CSV) and orphan chunks (a row
whose new description chunks fewer pieces than the previous run) are not
propagated to Qdrant in this MVP — documented as a known limitation.
"""

CSV_PATH = Path("data/products_catalog.csv")
COLLECTION_NAME = "products_catalog"
EMBED_MODEL = "embed-english-v3.0"
VECTOR_SIZE = 1024
BATCH_SIZE = 96  # Cohere embed API max texts per call

CHUNK_SIZE = 400  # chars
CHUNK_OVERLAP = 50
CHUNK_ID_STRIDE = 1000  # max chunks per product before IDs collide


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split `text` into overlapping chunks, preferring sentence/word boundaries.

    Short texts (≤ chunk_size) return a single chunk so we don't fragment
    descriptions that are already brief. Longer texts walk forward
    `chunk_size` chars at a time, backing up to the nearest natural break
    (sentence-end, then comma, then space) within the second half of the
    window so chunks don't slice mid-word."""
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= chunk_size:
        return [text]

    separators = [". ", "? ", "! ", "; ", ", ", " "]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window_floor = start + chunk_size // 2
            for sep in separators:
                idx = text.rfind(sep, window_floor, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks or [text]


def build_chunk_embedding_text(row: pd.Series, chunk: str) -> str:
    """Header (name | category | brand) + chunk content. The header is
    repeated on every chunk so each embedding has product context."""
    return (
        f"{row['product_name']} | "
        f"Category: {row['category']} | "
        f"Brand: {row['brand']} | "
        f"{chunk}"
    )


def build_chunk_payload(
    row: pd.Series, chunk: str, chunk_index: int, total_chunks: int
) -> dict:
    return {
        "product_id": str(row["product_id"]),
        "product_name": str(row["product_name"]),
        "category": str(row["category"]),
        "brand": str(row["brand"]),
        "current_price": float(row["current_price"]),
        "cost": float(row["cost"]),
        "stock_quantity": int(row["stock_quantity"]),
        "monthly_sales": int(row["monthly_sales"]),
        "average_rating": float(row["average_rating"]),
        "review_count": int(row["review_count"]),
        "description": str(row["description"]),
        "chunk_text": chunk,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
    }


def product_int_id(product_id: str) -> int:
    return int(product_id.split("-")[1])


def chunk_point_id(product_id: str, chunk_index: int) -> int:
    """Stable point ID per (product, chunk). Up to 1000 chunks per product."""
    return product_int_id(product_id) * CHUNK_ID_STRIDE + chunk_index


def expand_rows_to_chunks(df: pd.DataFrame) -> list[tuple[pd.Series, str, int, int]]:
    """Expand each product row into (row, chunk_text, chunk_index, total_chunks)
    records. A row with a short description yields a single record; longer
    descriptions yield multiple."""
    records: list[tuple[pd.Series, str, int, int]] = []
    for _, row in df.iterrows():
        chunks = chunk_text(str(row.get("description", "")))
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            records.append((row, chunk, idx, total))
    return records


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists, upserting in place")


def main() -> None:
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API")
    if not cohere_api_key:
        raise RuntimeError("COHERE_API is not set in the environment")

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    df = pd.read_csv(CSV_PATH).fillna("")
    print(f"Loaded {len(df)} rows from {CSV_PATH}")

    records = expand_rows_to_chunks(df)
    print(
        f"Expanded {len(df)} rows into {len(records)} chunks "
        f"(avg {len(records) / max(len(df), 1):.2f} chunks/product)"
    )

    qdrant = QdrantClient(url=qdrant_url)
    ensure_collection(qdrant)

    co = cohere.ClientV2(api_key=cohere_api_key)

    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        texts = [build_chunk_embedding_text(row, chunk) for row, chunk, _, _ in batch]

        response = co.embed(
            texts=texts,
            model=EMBED_MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        vectors = response.embeddings.float_

        points = [
            PointStruct(
                id=chunk_point_id(row["product_id"], idx),
                vector=vec,
                payload=build_chunk_payload(row, chunk, idx, total_chunks),
            )
            for vec, (row, chunk, idx, total_chunks) in zip(vectors, batch)
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"Batch {i // BATCH_SIZE + 1}: upserted {len(points)} chunks")

    print(f"Done. Upserted {total} chunks into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
