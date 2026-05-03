from pathlib import Path

import pandas as pd

from src.database import models
from src.database.session import Base, SessionLocal, engine

_CATALOG_PATH = Path(__file__).parents[2] / "data" / "products_catalog.csv"


def _ingest_products_if_empty() -> None:
    session = SessionLocal()
    try:
        if session.query(models.Product).count() > 0:
            return
        df = pd.read_csv(_CATALOG_PATH)
        df = df.where(pd.notna(df), None)
        rows = [models.Product(**row) for row in df.to_dict(orient="records")]
        session.add_all(rows)
        session.commit()
    finally:
        session.close()


def init_db() -> None:
    Path("./data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _ingest_products_if_empty()
