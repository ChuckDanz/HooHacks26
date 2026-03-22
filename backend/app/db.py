"""
Aerospike client wrapper.
Session items are stored as a list bin under key = session_id.
TTL: never expire (Aerospike CE test namespace has nsup-period=0).
"""

import os
import json
import aerospike
from aerospike import exception as ex

AEROSPIKE_HOST = os.getenv("AEROSPIKE_HOST", "localhost")
AEROSPIKE_PORT = int(os.getenv("AEROSPIKE_PORT", 3000))
NAMESPACE       = os.getenv("AEROSPIKE_NAMESPACE", "test")
SET             = "sessions"
TTL             = aerospike.TTL_NEVER_EXPIRE

_client: aerospike.Client | None = None


def get_client() -> aerospike.Client:
    global _client
    if _client is None or not _client.is_connected():
        config = {"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}
        _client = aerospike.client(config).connect()
    return _client


def _key(session_id: str):
    return (NAMESPACE, SET, session_id)


def get_items(session_id: str) -> list[dict]:
    client = get_client()
    try:
        _, _, bins = client.get(_key(session_id))
        raw = bins.get("items", [])
        return [json.loads(i) if isinstance(i, str) else i for i in raw]
    except ex.RecordNotFound:
        return []


def add_item(session_id: str, item: dict) -> list[dict]:
    client = get_client()
    items = get_items(session_id)
    # Deduplicate by product_url
    if not any(i.get("product_url") == item.get("product_url") for i in items):
        items.append(item)
    policy = {"ttl": TTL}
    client.put(_key(session_id), {"items": [json.dumps(i) for i in items]}, policy=policy)
    return items


def remove_item(session_id: str, item_id: str) -> list[dict]:
    client = get_client()
    items = get_items(session_id)
    items = [i for i in items if i.get("id") != item_id]
    policy = {"ttl": TTL}
    client.put(_key(session_id), {"items": [json.dumps(i) for i in items]}, policy=policy)
    return items


def clear_session(session_id: str) -> None:
    client = get_client()
    try:
        client.remove(_key(session_id))
    except ex.RecordNotFound:
        pass


# ── Person image (path stored in Aerospike, file lives on disk) ──────────────
# Aerospike CE has a 1 MB record limit — photos are much larger, so we store
# the file on disk and keep only the path in Aerospike.

import hashlib

_PERSON_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "person_cache")
os.makedirs(_PERSON_CACHE_DIR, exist_ok=True)


def save_person_image(session_id: str, image_bytes: bytes, suffix: str = ".jpg") -> str:
    """Write photo to disk, store path in Aerospike. Returns the file path."""
    fname = f"{session_id}{suffix}"
    fpath = os.path.abspath(os.path.join(_PERSON_CACHE_DIR, fname))
    with open(fpath, "wb") as f:
        f.write(image_bytes)
    client = get_client()
    client.put(_key(session_id), {"person_path": fpath}, policy={"ttl": TTL})
    return fpath


def get_person_image(session_id: str) -> str | None:
    """Return the file path of the cached person photo, or None."""
    client = get_client()
    try:
        _, _, bins = client.get(_key(session_id))
        path = bins.get("person_path")
        if path and os.path.exists(path):
            return path
        return None
    except ex.RecordNotFound:
        return None
