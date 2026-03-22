/**
 * FitCheck – Popup
 * Lists queued cart items; opens the try-on page with the session ID.
 */

const API_BASE = 'http://localhost:8000';

let items = [];

// ── Load items ────────────────────────────────────────────────────────────────

function load() {
  chrome.runtime.sendMessage({ type: 'GET_ITEMS' }, (resp) => {
    if (chrome.runtime.lastError || !resp?.ok) {
      renderError();
      return;
    }
    items = resp.items || [];
    render();
  });
}

// ── Render ────────────────────────────────────────────────────────────────────

function render() {
  const list     = document.getElementById('list');
  const empty    = document.getElementById('empty');
  const tryonBtn = document.getElementById('tryonBtn');

  if (!items.length) {
    list.innerHTML = '';
    empty.style.display = 'block';
    tryonBtn.disabled = true;
    return;
  }

  empty.style.display = 'none';
  tryonBtn.disabled = false;

  list.innerHTML = items.map(item => `
    <div class="item" data-id="${item.id}">
      <img
        src="${item.image_url}"
        alt="${item.name}"
        onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2252%22 height=%2268%22><rect fill=%22%23222%22 width=%2252%22 height=%2268%22/></svg>'"
      />
      <div class="meta">
        <div class="retailer">${item.retailer}</div>
        <div class="name" title="${item.name}">${item.name}</div>
        <div class="price">${item.price || ''}</div>
      </div>
      <button class="remove" data-id="${item.id}" title="Remove">×</button>
    </div>
  `).join('');

  // Attach remove listeners
  list.querySelectorAll('.remove').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      removeItem(btn.dataset.id);
    });
  });
}

function renderError() {
  document.getElementById('list').innerHTML = '';
  document.getElementById('empty').style.display = 'block';
  document.getElementById('empty').querySelector('p').textContent =
    'Could not load items. Is Docker running?';
}

// ── Remove ────────────────────────────────────────────────────────────────────

function removeItem(itemId) {
  chrome.runtime.sendMessage({ type: 'REMOVE_ITEM', itemId }, () => {
    items = items.filter(i => i.id !== itemId);
    render();
  });
}

// ── Clear all ─────────────────────────────────────────────────────────────────

document.getElementById('clearBtn').addEventListener('click', () => {
  if (!items.length) return;
  chrome.runtime.sendMessage({ type: 'CLEAR_ITEMS' }, () => {
    items = [];
    render();
  });
});

// ── Open try-on page ──────────────────────────────────────────────────────────

document.getElementById('tryonBtn').addEventListener('click', () => {
  chrome.runtime.sendMessage({ type: 'GET_SESSION' }, ({ sessionId }) => {
    const url = `${API_BASE}/tryon?session=${sessionId}`;
    chrome.tabs.create({ url });
  });
});

// ── Init ──────────────────────────────────────────────────────────────────────
load();
