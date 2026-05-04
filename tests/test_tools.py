from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src import ingest
from src.database.models import Product
from src.database.session import Base
from src.main import app
from src.tools import pricing_tool, rag_tool

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.fixture
def seeded_session(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False
    )
    Base.metadata.create_all(bind=engine)

    session = TestingSessionLocal()
    session.add_all(
        [
            Product(
                product_id="P-1",
                product_name="Alpha",
                category="Electronics",
                current_price=100.0,
                cost=60.0,
            ),
            Product(
                product_id="P-2",
                product_name="Beta",
                category="Electronics",
                current_price=50.0,
                cost=40.0,
            ),
            Product(
                product_id="P-3",
                product_name="Gamma",
                category="Fitness",
                current_price=20.0,
                cost=10.0,
            ),
        ]
    )
    session.commit()
    session.close()

    monkeypatch.setattr(pricing_tool, "SessionLocal", TestingSessionLocal)
    yield TestingSessionLocal


def test_analyze_profit_margins_summary(seeded_session):
    result = pricing_tool.analyze_profit_margins()
    assert isinstance(result, dict)
    assert result["count"] == 3
    assert result["summary"]["count"] == 3
    assert result["summary"]["scope"] == "all products"
    assert result["sources"] == ["sqlite:products"]
    # Beta has the lowest margin (20%), so it should be ranked #1
    names = [p["product_name"] for p in result["products"]]
    assert names[0] == "Beta"
    assert result["products"][0]["margin_percentage"] == 20.0


def test_analyze_profit_margins_category_filter(seeded_session):
    result = pricing_tool.analyze_profit_margins(category="electronics")
    assert result["count"] == 2
    names = [p["product_name"] for p in result["products"]]
    assert "Gamma" not in names
    assert result["filters"]["category"] == "electronics"


def test_analyze_profit_margins_max_margin_filter(seeded_session):
    result = pricing_tool.analyze_profit_margins(max_margin=30.0)
    # Only Beta has margin < 30% (Beta=20%, Alpha=40%, Gamma=50%)
    names = [p["product_name"] for p in result["products"]]
    assert names == ["Beta"]
    assert result["filters"]["max_margin"] == 30.0


def test_analyze_profit_margins_empty(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    EmptySessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(pricing_tool, "SessionLocal", EmptySessionLocal)

    result = pricing_tool.analyze_profit_margins()
    assert result["count"] == 0
    assert result["products"] == []
    assert "No products found" in result["answer"]


class _FakeEmbeddings:
    float_ = [[0.0] * 1024]


class _FakeEmbedResponse:
    embeddings = _FakeEmbeddings()


class _FakePoint:
    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class _FakeResult:
    def __init__(self, points):
        self.points = points


def test_search_product_catalog_no_matches(monkeypatch):
    monkeypatch.setattr(
        rag_tool._cohere, "embed", lambda **kw: _FakeEmbedResponse()
    )
    monkeypatch.setattr(
        rag_tool._qdrant, "query_points", lambda **kw: _FakeResult([])
    )

    result = rag_tool.search_product_catalog("anything")
    assert result["count"] == 0
    assert result["products"] == []
    assert result["answer"] == "No matching products found."
    assert result["sources"] == ["qdrant:products_catalog"]


def test_search_product_catalog_formats_hits(monkeypatch):
    fake_points = [
        _FakePoint(
            payload={
                "product_id": "P-42",
                "product_name": "Test Headphones",
                "category": "Electronics",
                "brand": "AudioMax",
                "current_price": 199.99,
                "average_rating": 4.7,
                "review_count": 100,
                "stock_quantity": 5,
                "description": "Great sound.",
                "chunk_text": "Great sound.",
                "chunk_index": 0,
                "total_chunks": 1,
            },
            score=0.875,
        )
    ]
    monkeypatch.setattr(
        rag_tool._cohere, "embed", lambda **kw: _FakeEmbedResponse()
    )
    monkeypatch.setattr(
        rag_tool._qdrant, "query_points", lambda **kw: _FakeResult(fake_points)
    )

    result = rag_tool.search_product_catalog("headphones")
    assert result["count"] == 1
    product = result["products"][0]
    assert product["product_name"] == "Test Headphones"
    assert product["brand"] == "AudioMax"
    assert product["current_price"] == 199.99
    assert product["relevance_score"] == 0.875
    assert product["matching_chunk"] == "Great sound."
    assert result["sources"] == ["qdrant:products_catalog"]


def test_search_product_catalog_dedupes_chunks(monkeypatch):
    """Two chunks of the same product should collapse into one result, and
    the chunk with the higher score should win."""
    payload_a_chunk0 = {
        "product_id": "P-42",
        "product_name": "Headphones",
        "category": "Electronics",
        "brand": "AudioMax",
        "current_price": 199.99,
        "average_rating": 4.7,
        "review_count": 100,
        "stock_quantity": 5,
        "description": "full description",
        "chunk_text": "first half of the description",
        "chunk_index": 0,
        "total_chunks": 2,
    }
    payload_a_chunk1 = {**payload_a_chunk0, "chunk_text": "second half", "chunk_index": 1}
    payload_b = {
        "product_id": "P-7",
        "product_name": "Speaker",
        "category": "Electronics",
        "brand": "SoundWave",
        "current_price": 89.99,
        "average_rating": 4.2,
        "review_count": 50,
        "stock_quantity": 12,
        "description": "loud speaker",
        "chunk_text": "loud speaker",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    fake_points = [
        _FakePoint(payload_a_chunk0, score=0.70),
        _FakePoint(payload_b, score=0.65),
        _FakePoint(payload_a_chunk1, score=0.90),  # higher-scoring chunk of P-42
    ]
    monkeypatch.setattr(rag_tool._cohere, "embed", lambda **kw: _FakeEmbedResponse())
    monkeypatch.setattr(
        rag_tool._qdrant, "query_points", lambda **kw: _FakeResult(fake_points)
    )

    result = rag_tool.search_product_catalog("audio", limit=5)
    pids = [p["product_id"] for p in result["products"]]
    assert pids == ["P-42", "P-7"]  # one entry per product, ranked by best chunk
    headphones = result["products"][0]
    assert headphones["relevance_score"] == 0.9
    assert headphones["matching_chunk"] == "second half"


def test_chunk_text_short_passthrough():
    chunks = ingest.chunk_text("Short description.")
    assert chunks == ["Short description."]


def test_chunk_text_empty_returns_one_empty():
    assert ingest.chunk_text("") == [""]
    assert ingest.chunk_text("   ") == [""]


def test_chunk_text_long_splits_with_overlap():
    sentence = (
        "Premium wireless headphones with active noise cancellation. "
        "Up to 30 hours of battery life. "
        "Bluetooth 5.3 with multipoint pairing. "
        "Soft memory-foam ear cushions for extended wear. "
        "Built-in mic for crystal-clear voice calls. "
        "Foldable design for portability. "
        "Comes with travel case and USB-C charging cable. "
        "Compatible with iOS and Android. "
        "Hi-Res Audio certified."
    )
    chunks = ingest.chunk_text(sentence, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)  # chunk_size + small overhead
    # Chunks should preserve all content (allowing for overlap repetition).
    assert "noise cancellation" in chunks[0]
    assert "Hi-Res Audio certified" in chunks[-1]


def test_chunk_point_id_is_stable_and_unique():
    assert ingest.chunk_point_id("PROD-001", 0) == 1000
    assert ingest.chunk_point_id("PROD-001", 1) == 1001
    assert ingest.chunk_point_id("PROD-002", 0) == 2000
    # Different (product, chunk) → different ID
    assert ingest.chunk_point_id("PROD-001", 5) != ingest.chunk_point_id("PROD-002", 5)
