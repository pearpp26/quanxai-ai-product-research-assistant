from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from src.agent.graph import run_agent
from src.database.init_db import init_db
from src.database.models import Feedback, QueryLog
from src.database.session import get_db


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    query_text: str
    response_text: str
    reasoning: Optional[str]
    tools_used: list[str]
    created_at: datetime


class FeedbackRequest(BaseModel):
    query_log_id: int
    rating: Literal["up", "down"]
    comment: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="AI Product Research Assistant", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)) -> QueryLog:
    try:
        result = run_agent(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    log = QueryLog(
        query_text=request.query,
        response_text=result["answer"],
        reasoning=result["reasoning"],
        tools_used=result["tools_used"],
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


@app.get("/queries", response_model=list[QueryResponse])
def list_queries(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db),
) -> list[QueryLog]:
    return (
        db.query(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest, db: Session = Depends(get_db)
) -> dict[str, str]:
    log = db.query(QueryLog).filter(QueryLog.id == request.query_log_id).first()
    if log is None:
        raise HTTPException(
            status_code=404, detail=f"QueryLog {request.query_log_id} not found"
        )

    feedback = Feedback(
        query_log_id=request.query_log_id,
        rating=request.rating,
        comment=request.comment,
    )
    db.add(feedback)
    db.commit()
    return {"status": "success", "message": "Feedback recorded."}
