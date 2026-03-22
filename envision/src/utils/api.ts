const BASE = '/api';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface CartItem {
  id: string;
  name: string;
  image_url: string;
  price?: string;
  retailer?: string;
  product_url?: string;
  category: string; // 'upper_body' | 'lower_body' | 'dress'
  added_at: string;
}

export type TryOnCategory = 'upper_body' | 'lower_body' | 'dress';

// ── Cart queue ────────────────────────────────────────────────────────────────

export async function removeCartItem(sessionId: string, itemId: string): Promise<void> {
  await fetch(`${BASE}/items/${itemId}`, {
    method: 'DELETE',
    headers: { 'X-Session-ID': sessionId },
  });
}

export async function getCartItems(sessionId: string): Promise<CartItem[]> {
  try {
    const res = await fetch(`${BASE}/items?session_id=${sessionId}`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.items ?? [];
  } catch {
    return [];
  }
}

// ── Person image ──────────────────────────────────────────────────────────────

export async function cachePerson(sessionId: string, file: File): Promise<void> {
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('person', file);
  const res = await fetch(`${BASE}/person`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail ?? 'Upload failed');
  }
}

// ── Try-on ────────────────────────────────────────────────────────────────────

export async function tryOnGarment(
  sessionId: string,
  garmentUrl: string,
  category: TryOnCategory,
  personFile?: File,
): Promise<string> {
  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('garment_url', garmentUrl);
  form.append('category', category);
  if (personFile) form.append('person', personFile);

  // Try-on can take 30-120s — no artificial timeout
  const res = await fetch(`${BASE}/tryon`, { method: 'POST', body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Try-on failed' }));
    throw new Error(err.detail ?? 'Try-on failed');
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Map API category string → frontend tab key */
export function categoryToTab(category: string): 'tops' | 'pants' {
  return category === 'upper_body' ? 'tops' : 'pants';
}

/** Map frontend tab key → API category string */
export function tabToCategory(tab: 'tops' | 'pants'): TryOnCategory {
  return tab === 'tops' ? 'upper_body' : 'lower_body';
}
