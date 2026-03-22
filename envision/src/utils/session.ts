const KEY = 'envision_session_id';

export function getOrCreateSession(): string {
  // If the extension opened us with ?session=<uuid>, adopt that session
  const param = new URLSearchParams(window.location.search).get('session');
  if (param) {
    localStorage.setItem(KEY, param);
    // Clean the URL without reloading
    window.history.replaceState({}, '', window.location.pathname);
    return param;
  }
  let id = localStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(KEY, id);
  }
  return id;
}

export function getSession(): string | null {
  return localStorage.getItem(KEY);
}
