from __future__ import annotations

from sqlalchemy import func

from src.database.models import Product
from src.database.session import SessionLocal


def analyze_profit_margins(
    category: str | None = None,
    max_margin: float | None = None,
) -> str:
    """Compute profit margins by querying the products table; return a summary
    line plus the top 10 lowest-margin products as a formatted string."""
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

    if not products:
        if category is not None:
            return f"No products found for category '{category}'."
        return "No products found."

    enriched = [
        (p, ((p.current_price - p.cost) / p.current_price) * 100) for p in products
    ]

    if max_margin is not None:
        enriched = [(p, m) for p, m in enriched if m < max_margin]

    if not enriched:
        return "No products match the given filters."

    margins = [m for _, m in enriched]
    count = len(margins)
    avg_margin = sum(margins) / count
    min_margin = min(margins)
    max_margin_val = max(margins)

    scope = f"category '{category}'" if category else "all products"
    summary = (
        f"Summary for {scope}:\n"
        f"  Products: {count} | Avg Margin: {avg_margin:.2f}% | "
        f"Min: {min_margin:.2f}% | Max: {max_margin_val:.2f}%"
    )

    enriched.sort(key=lambda pair: pair[1])
    top = enriched[:10]
    lines = [
        f"{i}. {p.product_name} ({p.category})\n"
        f"   Price: ${p.current_price:.2f} | Cost: ${p.cost:.2f} | "
        f"Margin: {m:.2f}%"
        for i, (p, m) in enumerate(top, start=1)
    ]
    return summary + "\n\nTop 10 by lowest margin:\n" + "\n".join(lines)
