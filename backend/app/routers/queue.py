from fastapi import APIRouter, Header, HTTPException
from app.models import AddItemRequest, QueueResponse, CartItem
from app import db

router = APIRouter(prefix="/api", tags=["queue"])


def _require_session(x_session_id: str | None) -> str:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header required")
    return x_session_id


@router.get("/items", response_model=QueueResponse)
def get_queue(x_session_id: str | None = Header(default=None)):
    sid = _require_session(x_session_id)
    items = db.get_items(sid)
    return QueueResponse(session_id=sid, items=items, count=len(items))


@router.post("/items", response_model=QueueResponse, status_code=201)
def add_to_queue(
    body: AddItemRequest,
    x_session_id: str | None = Header(default=None),
):
    sid = _require_session(x_session_id)
    item_dict = body.item.model_dump()
    updated = db.add_item(sid, item_dict)
    return QueueResponse(session_id=sid, items=updated, count=len(updated))


@router.delete("/items/{item_id}", response_model=QueueResponse)
def remove_from_queue(
    item_id: str,
    x_session_id: str | None = Header(default=None),
):
    sid = _require_session(x_session_id)
    updated = db.remove_item(sid, item_id)
    return QueueResponse(session_id=sid, items=updated, count=len(updated))


@router.delete("/items", status_code=204)
def clear_queue(x_session_id: str | None = Header(default=None)):
    sid = _require_session(x_session_id)
    db.clear_session(sid)
