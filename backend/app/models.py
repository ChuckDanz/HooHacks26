from pydantic import BaseModel, HttpUrl
from typing import Optional
import time
import uuid


class CartItem(BaseModel):
    id: str = ""
    name: str
    image_url: str
    product_url: str
    retailer: str
    price: Optional[str] = None
    size: Optional[str] = None
    added_at: float = 0.0

    def model_post_init(self, __context):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.added_at:
            self.added_at = time.time()


class AddItemRequest(BaseModel):
    item: CartItem


class RemoveItemRequest(BaseModel):
    item_id: str


class QueueResponse(BaseModel):
    session_id: str
    items: list[CartItem]
    count: int
