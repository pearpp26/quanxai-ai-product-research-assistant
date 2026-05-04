from __future__ import annotations

from sqlalchemy import func

from src.database.models import Product
from src.database.session import SessionLocal


def analyze_profit_margins(
    category: str | None = None,
    max_margin: float | None = None,
) -> dict:
    """Compute profit margins from the products table; return a structured dict
    with summary statistics and the top 10 lowest-margin products."""
    session = SessionLocal()
    try:
        query = session.query(Product).filter(
            Product.current_price.isnot(None),
            Product.cost.isnot(None),
            Product.current_price != 0,
        )
        if category is not None:
            query = query.filter(func.lower(Product.category) == category.lower())
        products = query.all()
    finally:
        session.close()

    filters = {"category": category, "max_margin": max_margin}
    scope = f"category '{category}'" if category else "all products"

    if not products:
        return {
            "answer": f"No products found for {scope}.",
            "summary": None,
            "products": [],
            "count": 0,
            "filters": filters,
            "sources": ["sqlite:products"],
        }

    enriched = [
        (p, ((p.current_price - p.cost) / p.current_price) * 100) for p in products
    ]
    if max_margin is not None:
        enriched = [(p, m) for p, m in enriched if m < max_margin]

    if not enriched:
        return {
            "answer": "No products match the given filters.",
            "summary": None,
            "products": [],
            "count": 0,
            "filters": filters,
            "sources": ["sqlite:products"],
        }

    margins = [m for _, m in enriched]
    count = len(margins)
    avg_margin = sum(margins) / count
    min_margin = min(margins)
    max_margin_val = max(margins)

    summary = {
        "scope": scope,
        "count": count,
        "avg_margin": round(avg_margin, 2),
        "min_margin": round(min_margin, 2),
        "max_margin": round(max_margin_val, 2),
    }

    enriched.sort(key=lambda pair: pair[1])
    top = enriched[:10]
    products_list = [
        {
            "product_id": p.product_id,
            "product_name": p.product_name,
            "category": p.category,
            "current_price": round(float(p.current_price), 2),
            "cost": round(float(p.cost), 2),
            "margin_percentage": round(float(m), 2),
        }
        for p, m in top
    ]

    answer = (
        f"Analyzed {count} products in {scope}. "
        f"Avg margin: {avg_margin:.2f}%, Min: {min_margin:.2f}%, "
        f"Max: {max_margin_val:.2f}%. Lowest-margin product: "
        f"{products_list[0]['product_name']} "
        f"({products_list[0]['margin_percentage']:.2f}%)."
    )

    return {
        "answer": answer,
        "summary": summary,
        "products": products_list,
        "count": count,
        "filters": filters,
        "sources": ["sqlite:products"],
    }
