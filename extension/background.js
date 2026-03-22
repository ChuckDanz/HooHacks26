/**
 * FitCheck – Background Service Worker
 * Manages session ID, relays items to the backend API, and updates the badge.
 */

const API_BASE = 'http://localhost:8000';

// ── Session ID ────────────────────────────────────────────────────────────────

async function getSessionId() {
  const data = await chrome.storage.local.get('sessionId');
  if (data.sessionId) return data.sessionId;
  const id = crypto.randomUUID();
  await chrome.storage.local.set({ sessionId: id });
  return id;
}

// ── Badge ─────────────────────────────────────────────────────────────────────

async function refreshBadge() {
  const sessionId = await getSessionId();
  try {
    const res = await fetch(`${API_BASE}/api/items`, {
      headers: { 'X-Session-ID': sessionId },
    });
    if (!res.ok) return;
    const data = await res.json();
    const count = data.count || 0;
    chrome.action.setBadgeText({ text: count > 0 ? String(count) : '' });
    chrome.action.setBadgeBackgroundColor({ color: '#7c3aed' });
  } catch (_) {
    // API unreachable — clear badge silently
    chrome.action.setBadgeText({ text: '' });
  }
}

// ── Message handler (from content scripts) ───────────────────────────────────

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === 'ADD_ITEM') {
    handleAddItem(msg.item).then(sendResponse).catch(err => {
      console.error('[FitCheck] ADD_ITEM error', err);
      sendResponse({ ok: false, error: err.message });
    });
    return true; // keep channel open for async response
  }

  if (msg.type === 'REMOVE_ITEM') {
    handleRemoveItem(msg.itemId).then(sendResponse).catch(err => {
      sendResponse({ ok: false, error: err.message });
    });
    return true;
  }

  if (msg.type === 'GET_ITEMS') {
    handleGetItems().then(sendResponse).catch(err => {
      sendResponse({ ok: false, items: [], error: err.message });
    });
    return true;
  }

  if (msg.type === 'CLEAR_ITEMS') {
    handleClearItems().then(sendResponse).catch(err => {
      sendResponse({ ok: false, error: err.message });
    });
    return true;
  }

  if (msg.type === 'GET_SESSION') {
    getSessionId().then(id => sendResponse({ sessionId: id }));
    return true;
  }
});

// ── API calls ─────────────────────────────────────────────────────────────────

async function handleAddItem(item) {
  const sessionId = await getSessionId();
  const res = await fetch(`${API_BASE}/api/items`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Session-ID': sessionId,
    },
    body: JSON.stringify({ item }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = body.detail || `API error ${res.status}`;
    if (res.status === 409) {
      // Demo cap reached — surface cleanly to content script
      return { ok: false, limitReached: true, error: detail };
    }
    throw new Error(detail);
  }
  const data = await res.json();
  chrome.action.setBadgeText({ text: String(data.count) });
  chrome.action.setBadgeBackgroundColor({ color: '#7c3aed' });
  return { ok: true, count: data.count };
}

async function handleRemoveItem(itemId) {
  const sessionId = await getSessionId();
  const res = await fetch(`${API_BASE}/api/items/${itemId}`, {
    method: 'DELETE',
    headers: { 'X-Session-ID': sessionId },
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  const data = await res.json();
  chrome.action.setBadgeText({ text: data.count > 0 ? String(data.count) : '' });
  return { ok: true, count: data.count };
}

async function handleGetItems() {
  const sessionId = await getSessionId();
  const res = await fetch(`${API_BASE}/api/items`, {
    headers: { 'X-Session-ID': sessionId },
  });
  if (!res.ok) throw new Error(`API ${res.status}`);
  const data = await res.json();
  return { ok: true, items: data.items, count: data.count };
}

async function handleClearItems() {
  const sessionId = await getSessionId();
  await fetch(`${API_BASE}/api/items`, {
    method: 'DELETE',
    headers: { 'X-Session-ID': sessionId },
  });
  chrome.action.setBadgeText({ text: '' });
  return { ok: true };
}

// ── Init ──────────────────────────────────────────────────────────────────────
refreshBadge();
