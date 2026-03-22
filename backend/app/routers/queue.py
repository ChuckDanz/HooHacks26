from fastapi import APIRouter, Header, HTTPException
from app.models import AddItemRequest, QueueResponse, CartItem
from app import db

router = APIRouter(prefix="/api", tags=["queue"])


def _require_session(x_session_id: str | None) -> str:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header required")
    return x_session_id


@router.get("/items", response_model=QueueResponse)
def get_queue(
    session_id: str | None = None,
    x_session_id: str | None = Header(default=None),
):
    # Accept session from either query param (frontend) or header (extension)
    sid = session_id or x_session_id
    if not sid:
        raise HTTPException(status_code=400, detail="session_id required")
    items = db.get_items(sid)
    return QueueResponse(session_id=sid, items=items, count=len(items))


DEMO_MAX_ITEMS = 2

@router.post("/items", response_model=QueueResponse, status_code=201)
def add_to_queue(
    body: AddItemRequest,
    x_session_id: str | None = Header(default=None),
):
    sid = _require_session(x_session_id)
    current = db.get_items(sid)
    if len(current) >= DEMO_MAX_ITEMS:
        raise HTTPException(
            status_code=409,
            detail=f"Demo limit: cart holds max {DEMO_MAX_ITEMS} items. Remove one first."
        )
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
