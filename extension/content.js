/**
 * FitCheck – Content Script
 * Runs on Amazon, H&M, Depop, Shein product pages.
 * 1. Detects "Add to Cart / Bag / Buy" button clicks.
 * 2. Extracts product info (OG meta + site-specific fallbacks).
 * 3. Sends to background service worker.
 */

(function () {
  'use strict';

  const host = window.location.hostname;

  // ── Site detection ──────────────────────────────────────────────────────────

  const SITE =
    host.includes('amazon.')  ? 'amazon'  :
    host.includes('hm.com')   ? 'hm'      :
    host.includes('depop.com')? 'depop'   :
    host.includes('shein.com')? 'shein'   : null;

  if (!SITE) return;

  // ── OG meta helpers ─────────────────────────────────────────────────────────

  function ogMeta(prop) {
    const el = document.querySelector(`meta[property="${prop}"], meta[name="${prop}"]`);
    return el ? el.content.trim() : '';
  }

  function bestImage() {
    // OG image is usually the cleanest product shot
    const og = ogMeta('og:image');
    if (og) return og;
    // Amazon landing image fallback
    const img = document.querySelector('#landingImage, #imgBlkFront, .product-image__img');
    return img ? img.src : '';
  }

  function bestTitle() {
    return ogMeta('og:title') ||
      (document.querySelector('h1')?.innerText.trim() ?? document.title);
  }

  // ── Site-specific extractors ─────────────────────────────────────────────────

  const extractors = {
    amazon() {
      return {
        name:        document.querySelector('#productTitle')?.innerText.trim() || bestTitle(),
        image_url:   bestImage(),
        price:       document.querySelector('.a-price .a-offscreen, .a-price-whole')?.innerText.trim() || ogMeta('og:price:amount'),
        size:        document.querySelector('#native_dropdown_selected_size_name, #size_name')?.value.trim() || '',
        product_url: window.location.href.split('/ref=')[0],
        retailer:    'amazon',
      };
    },

    hm() {
      return {
        name:        document.querySelector('h1.product-detail-name, h1')?.innerText.trim() || bestTitle(),
        image_url:   bestImage(),
        price:       document.querySelector('.price-value, [data-testid="price"]')?.innerText.trim() || ogMeta('og:price:amount'),
        size:        document.querySelector('.product-sizes .active, .size-picker__item--active')?.innerText.trim() || '',
        product_url: window.location.href,
        retailer:    'h&m',
      };
    },

    depop() {
      return {
        name:        document.querySelector('[data-testid="product-name"], h1')?.innerText.trim() || bestTitle(),
        image_url:   bestImage(),
        price:       document.querySelector('[data-testid="product-price"]')?.innerText.trim() || ogMeta('og:price:amount'),
        size:        document.querySelector('[data-testid="product-attributes"] li')?.innerText.trim() || '',
        product_url: window.location.href,
        retailer:    'depop',
      };
    },

    shein() {
      return {
        name:        document.querySelector('.product-intro__head-name, h1')?.innerText.trim() || bestTitle(),
        image_url:   bestImage(),
        price:       document.querySelector('.product-intro__head-price .from, .she-productIntro__price')?.innerText.trim() || ogMeta('og:price:amount'),
        size:        document.querySelector('.product-intro__size-choose .active, .j-sku-item.selected')?.innerText.trim() || '',
        product_url: window.location.href,
        retailer:    'shein',
      };
    },
  };

  // ── ATC button selectors (one per site, plus generic fallback) ───────────────

  const ATC_SELECTORS = {
    amazon: '#add-to-cart-button, #submit.add-to-cart',
    hm:     '[data-testid*="add"], button.add-to-bag, .js-add-to-bag',
    depop:  '[data-testid*="buy"], button[class*="BuyButton"], button[class*="buy"]',
    shein:  '.product-intro__add-btn, .j-btn-add-to-cart, button[class*="addToBag"]',
  };

  const GENERIC_ATC_RE = /add\s+to\s+(cart|bag|basket)|buy\s+now/i;

  // ── Show toast notification ──────────────────────────────────────────────────

  function showToast(text) {
    const el = document.createElement('div');
    el.textContent = text;
    Object.assign(el.style, {
      position:     'fixed',
      bottom:       '24px',
      right:        '24px',
      background:   '#7c3aed',
      color:        '#fff',
      padding:      '12px 20px',
      borderRadius: '10px',
      fontFamily:   'sans-serif',
      fontSize:     '14px',
      fontWeight:   '600',
      zIndex:       '2147483647',
      boxShadow:    '0 4px 20px rgba(0,0,0,0.35)',
      transition:   'opacity 0.4s',
      opacity:      '1',
    });
    document.body.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; }, 2500);
    setTimeout(() => el.remove(), 3000);
  }

  // ── Capture & send ───────────────────────────────────────────────────────────

  function captureAndSend() {
    const extract = extractors[SITE];
    if (!extract) return;

    const item = extract();

    if (!item.image_url) {
      console.warn('[FitCheck] No image found on this page, skipping.');
      return;
    }

    // Infer category from product name
    const lower = (item.name || '').toLowerCase();
    const isBottom = /\b(pant|jean|denim|trouser|short|skirt|legging|bottom|chino|jogger|cargo)\b/.test(lower);
    item.category = isBottom ? 'lower_body' : 'upper_body';

    // Add a UUID so Aerospike / background can deduplicate
    item.id = crypto.randomUUID();
    item.added_at = Date.now() / 1000;

    chrome.runtime.sendMessage({ type: 'ADD_ITEM', item }, (resp) => {
      if (chrome.runtime.lastError) {
        console.error('[FitCheck]', chrome.runtime.lastError.message);
        return;
      }
      if (resp?.ok) {
        showToast(`✓ Added to FitCheck (${resp.count}/2)`);
      } else if (resp?.limitReached) {
        showToast('FitCheck cart full (2/2) — open the app to try on or remove an item');
      } else {
        showToast('FitCheck: could not save item (is Docker running?)');
      }
    });
  }

  // ── Event delegation — ATC click ────────────────────────────────────────────

  function isATCButton(el) {
    // Walk up a few levels (click may land on icon/span inside button)
    for (let i = 0; i < 5; i++) {
      if (!el || el === document.body) break;
      const tag = el.tagName?.toLowerCase();
      const text = el.innerText || el.value || '';
      const id   = (el.id || '').toLowerCase();
      const cls  = (el.className || '').toLowerCase();

      const siteSelector = ATC_SELECTORS[SITE];
      if (el.matches?.(siteSelector)) return true;
      if ((tag === 'button' || tag === 'a') && GENERIC_ATC_RE.test(text)) return true;
      if (id.includes('add-to-cart') || id.includes('add-to-bag')) return true;
      if (cls.includes('add-to-cart') || cls.includes('add-to-bag')) return true;

      el = el.parentElement;
    }
    return false;
  }

  document.addEventListener('click', (e) => {
    if (isATCButton(e.target)) {
      // Small delay to allow site JS to update DOM / sizes before we read
      setTimeout(captureAndSend, 300);
    }
  }, true); // capture phase = fires before site JS can stopPropagation

  // ── Also scan existing cart page (amazon.com/cart, etc.) ────────────────────

  function scrapeCartPage() {
    // Amazon cart page items
    if (SITE === 'amazon' && window.location.pathname.startsWith('/cart')) {
      document.querySelectorAll('.sc-list-item').forEach(row => {
        const name      = row.querySelector('.a-truncate-cut, .a-size-medium')?.innerText.trim();
        const image_url = row.querySelector('img')?.src;
        const price     = row.querySelector('.sc-price')?.innerText.trim();
        const product_url = row.querySelector('a')?.href;
        if (name && image_url) {
          const item = { id: crypto.randomUUID(), name, image_url, price: price || '', size: '', product_url: product_url || '', retailer: 'amazon', added_at: Date.now() / 1000 };
          chrome.runtime.sendMessage({ type: 'ADD_ITEM', item });
        }
      });
    }
  }

  // Run cart scrape after a brief settle (dynamic carts take a moment to render)
  setTimeout(scrapeCartPage, 1500);
})();
