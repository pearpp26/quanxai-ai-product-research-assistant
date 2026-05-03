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

Idempotent: re-running upserts by integer point ID derived from `product_id`,
so updated rows overwrite, new rows are added, untouched rows stay intact.
Hard-deletes from the CSV are not propagated to Qdrant in this MVP.
"""

CSV_PATH = Path("data/products_catalog.csv")
COLLECTION_NAME = "products_catalog"
EMBED_MODEL = "embed-english-v3.0"
VECTOR_SIZE = 1024
BATCH_SIZE = 96  # Cohere embed API max texts per call


def build_embedding_text(row: pd.Series) -> str:
    return (
        f"{row['product_name']} | "
        f"Category: {row['category']} | "
        f"Brand: {row['brand']} | "
        f"{row['description']}"
    )


def build_payload(row: pd.Series) -> dict:
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
    }


def point_id_from_product_id(product_id: str) -> int:
    return int(product_id.split("-")[1])


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

    qdrant = QdrantClient(url=qdrant_url)
    ensure_collection(qdrant)

    co = cohere.ClientV2(api_key=cohere_api_key)

    total = 0
    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        texts = [build_embedding_text(row) for _, row in batch.iterrows()]

        response = co.embed(
            texts=texts,
            model=EMBED_MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        vectors = response.embeddings.float_

        points = [
            PointStruct(
                id=point_id_from_product_id(row["product_id"]),
                vector=vec,
                payload=build_payload(row),
            )
            for vec, (_, row) in zip(vectors, batch.iterrows())
        ]
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"Batch {i // BATCH_SIZE + 1}: upserted {len(points)} docs")

    print(f"Done. Upserted {total} docs into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
