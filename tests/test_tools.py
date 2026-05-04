from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
    assert "Products: 3" in result
    assert "Top 10 by lowest margin" in result
    # Beta has the lowest margin (20%), so it should be ranked #1
    assert result.index("Beta") < result.index("Alpha")
    assert result.index("Beta") < result.index("Gamma")


def test_analyze_profit_margins_category_filter(seeded_session):
    result = pricing_tool.analyze_profit_margins(category="electronics")
    assert "Products: 2" in result
    assert "Gamma" not in result


def test_analyze_profit_margins_max_margin_filter(seeded_session):
    result = pricing_tool.analyze_profit_margins(max_margin=30.0)
    # Only Beta has margin < 30% (Beta=20%, Alpha=40%, Gamma=50%)
    assert "Beta" in result
    assert "Alpha" not in result
    assert "Gamma" not in result


def test_analyze_profit_margins_empty(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    EmptySessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(pricing_tool, "SessionLocal", EmptySessionLocal)

    result = pricing_tool.analyze_profit_margins()
    assert "No products found" in result


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
    assert result == "No matching products found."


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
    assert "Test Headphones" in result
    assert "AudioMax" in result
    assert "199.99" in result
    assert "0.875" in result
