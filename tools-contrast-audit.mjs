// Worst-case text contrast over the animated halftone banners.
//
// Measures REAL rendered pixels: hides the copy, screenshots the banner, then
// decodes that PNG back inside the page and samples the area each text element
// occupied. That folds in the dot canvas, the .tex scrim and anything else in
// the stack — no analytic compositing to get wrong. Repeats over several
// animation frames and keeps the worst ratio seen.
const [,, url, label] = process.argv;
const list = await (await fetch('http://localhost:9333/json/list')).json();
const page = list.find(t => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
let id = 0; const p = new Map();
const send = (m, q = {}) => new Promise(r => { const i = ++id; p.set(i, r); ws.send(JSON.stringify({ id: i, method: m, params: q })); });
ws.onmessage = e => { const m = JSON.parse(e.data); if (m.id && p.has(m.id)) { p.get(m.id)(m.result); p.delete(m.id); } };
await new Promise(r => ws.onopen = r);
await send('Page.enable'); await send('Runtime.enable');
await send('Page.navigate', { url });
await new Promise(r => setTimeout(r, 3500));
// reveals are scroll-triggered; without this the lower banners are still at
// opacity 0 and we would sample the white page behind them
await send('Runtime.evaluate', { expression:
  `(() => { const s = document.createElement('style');
    s.textContent = '[data-reveal]{opacity:1 !important;transform:none !important}';
    document.head.appendChild(s); return 1; })()` });
await new Promise(r => setTimeout(r, 400));

const ev = async (expression) =>
  (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result.value;

// 1. catalogue the banners and the text laid over them
const banners = await ev(`
(() => {
  const parse = c => { const m = c.match(/[\\d.]+/g).map(Number); return { r:m[0],g:m[1],b:m[2],a:m[3] === undefined ? 1 : m[3] }; };
  window.__b = [...document.querySelectorAll('canvas.halftone')].map((cv, bi) => {
    const host = cv.parentElement;
    const texts = [...host.querySelectorAll('h1,h2,h3,p,span,a,b,em,i,dt,dd')].filter(el => {
      if (!el.textContent.trim()) return false;
      if (el.children.length) return false;   // leaf text only — a wrapper's OWN text (if any) is not what's rendered by its styled children
      let n = el;                                    // skip anything on its own opaque chip
      while (n && n !== host) { if (parse(getComputedStyle(n).backgroundColor).a >= .95) return false; n = n.parentElement; }
      const r = el.getBoundingClientRect();
      return r.width > 8 && r.height > 6;
    });
    return { bi, host, texts };
  });
  return window.__b.map(b => ({
    bi: b.bi,
    host: b.host.className.toString().slice(0, 46),
    rect: (r => ({ x: r.x + scrollX, y: r.y + scrollY, w: r.width, h: r.height }))(b.host.getBoundingClientRect()),
    texts: b.texts.map(el => ({
      label: (el.className.toString().slice(0, 26) || el.tagName) + ' | ' + el.textContent.trim().slice(0, 40),
      color: getComputedStyle(el).color,
      px: parseFloat(getComputedStyle(el).fontSize),
      weight: +getComputedStyle(el).fontWeight,
    })),
  }));
})()`);

const worst = new Map();   // key -> {ratio, px, weight}

for (let frame = 0; frame < 8; frame++) {
  for (const b of banners) {
    if (!b.texts.length) continue;
    // bring the banner on screen so its canvas is actually painting
    await ev(`(() => { window.__b[${b.bi}].host.scrollIntoView({ block: 'center' }); return 1; })()`);
    await new Promise(r => setTimeout(r, 260));
    // hide the copy, capture, restore
    await ev(`(() => { window.__b[${b.bi}].texts.forEach(e => e.style.visibility = 'hidden'); return 1; })()`);
    await new Promise(r => setTimeout(r, 70));
    const shot = await send('Page.captureScreenshot', {
      format: 'png', captureBeyondViewport: true,
      clip: { x: b.rect.x, y: b.rect.y, width: b.rect.w, height: b.rect.h, scale: 1 },
    });
    // hand the PNG over as a call argument — inlining megabytes of base64 into
    // an evaluate() expression silently fails
    const glob = (await send('Runtime.evaluate', { expression: 'globalThis' })).result.objectId;
    await send('Runtime.callFunctionOn', {
      objectId: glob,
      functionDeclaration: 'function (b64) { this.__png = b64; }',
      arguments: [{ value: shot.data }],
    });
    const rows = await ev(`
    (async () => {
      const parse = c => { const m = c.match(/[\\d.]+/g).map(Number); return { r:m[0],g:m[1],b:m[2],a:m[3] === undefined ? 1 : m[3] }; };
      const lin = v => { v /= 255; return v <= .03928 ? v / 12.92 : Math.pow((v + .055) / 1.055, 2.4); };
      const lum = (r,g,b) => .2126*lin(r) + .7152*lin(g) + .0722*lin(b);
      const ratio = (a,b) => { const [hi,lo] = a > b ? [a,b] : [b,a]; return (hi + .05) / (lo + .05); };
      const img = new Image();
      img.src = 'data:image/png;base64,' + window.__png;
      await img.decode();
      const cv = document.createElement('canvas');
      cv.width = img.width; cv.height = img.height;
      const cx = cv.getContext('2d'); cx.drawImage(img, 0, 0);
      const b = window.__b[${b.bi}];
      const hr = b.host.getBoundingClientRect();
      b.texts.forEach(e => e.style.visibility = '');
      return b.texts.map(el => {
        const r = el.getBoundingClientRect();
        const x = Math.max(0, Math.round(r.left - hr.left)), y = Math.max(0, Math.round(r.top - hr.top));
        const w = Math.min(cv.width - x, Math.round(r.width)), h = Math.min(cv.height - y, Math.round(r.height));
        if (w < 2 || h < 2) return null;
        const d = cx.getImageData(x, y, w, h).data;
        const fg = parse(getComputedStyle(el).color);
        let lo = Infinity, bg = null;
        for (let i = 0; i < d.length; i += 4) {
          const R = d[i], G = d[i+1], B = d[i+2];
          const fr = fg.r*fg.a + R*(1-fg.a), fgn = fg.g*fg.a + G*(1-fg.a), fb = fg.b*fg.a + B*(1-fg.a);
          const c = ratio(lum(fr,fgn,fb), lum(R,G,B));
          if (c < lo) { lo = c; bg = [R,G,B]; }
        }
        return { key: (el.className.toString().slice(0,26) || el.tagName) + ' | ' + el.textContent.trim().slice(0,40),
                 ratio: lo, bg, px: parseFloat(getComputedStyle(el).fontSize), weight: +getComputedStyle(el).fontWeight };
      }).filter(Boolean);
    })()`);
    for (const r of rows) {
      const prev = worst.get(r.key);
      if (!prev || r.ratio < prev.ratio) worst.set(r.key, r);
    }
  }
}

console.log(`\n### ${label}`);
let fails = 0;
for (const r of [...worst.values()].sort((a, b) => a.ratio - b.ratio)) {
  const big = r.px >= 24 || (r.px >= 18.66 && r.weight >= 700);
  const need = big ? 3 : 4.5;
  const ok = r.ratio >= need;
  if (!ok) fails++;
  console.log(`${ok ? ' ok ' : 'FAIL'} ${r.ratio.toFixed(2).padStart(6)}:1 (need ${need})  ${String(r.px).padStart(5)}px ${String(r.weight).padStart(3)}  bg=${String(r.bg).padEnd(13)} ${r.key.slice(0, 62)}`);
}
console.log(fails ? `>>> ${fails} FAILING` : '>>> all pass');
ws.close();
