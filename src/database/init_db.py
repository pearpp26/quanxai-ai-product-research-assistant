from pathlib import Path

from src.database import models
from src.database.session import Base, engine


def init_db() -> None:
    Path("./data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
