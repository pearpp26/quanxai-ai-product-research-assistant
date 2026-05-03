from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.sql import func

from src.database.session import Base


class Product(Base):
    __tablename__ = "products"

    product_id = Column(String, primary_key=True)
    product_name = Column(String, nullable=False)
    category = Column(String, nullable=False, index=True)
    brand = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    current_price = Column(Float, nullable=True)
    cost = Column(Float, nullable=True)
    stock_quantity = Column(Integer, nullable=True)
    monthly_sales = Column(Integer, nullable=True)
    average_rating = Column(Float, nullable=True)
    review_count = Column(Integer, nullable=True)
    supplier = Column(String, nullable=True)
    last_updated = Column(String, nullable=True)


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    query_text = Column(Text, nullable=False)
    tools_used = Column(JSON, nullable=False, default=list)
    reasoning = Column(Text, nullable=True)
    response_text = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    query_log_id = Column(
        Integer, ForeignKey("query_logs.id"), nullable=False, index=True
    )
    rating = Column(String(8), nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
