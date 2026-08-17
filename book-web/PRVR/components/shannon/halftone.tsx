'use client';

import { useEffect, useRef } from 'react';

/**
 * The Shannon Dynamics halftone: an animated moiré dot field with a magnetic
 * hover lens. Ported from the canvas script in the marketing site's book pages,
 * minus the layers those pages use and a banner does not (the scroll-driven
 * pixellation and the telemetry overlay).
 *
 * It is decoration and nothing else — `aria-hidden`, and under
 * `prefers-reduced-motion` it paints one static frame instead of animating.
 */

/** Dot pitch, in CSS pixels. */
const GAP = 7;
const LENS_R = 105;
const LENS_R2 = LENS_R * LENS_R;

interface HalftoneProps {
  /** Multiplier on the field's own clock. The site uses 1.6 for banners, 1.4 for the footer. */
  speed?: number;
  /** The cursor lens. Off for surfaces the reader never points at. */
  lens?: boolean;
}

export function Halftone({ speed = 1.6, lens = true }: HalftoneProps) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const host = canvas.parentElement;

    let W = 0;
    let H = 0;
    // target cursor, trailing cursor, and the eased hover energy between them
    let tx = -9e3;
    let ty = -9e3;
    let cx = -9e3;
    let cy = -9e3;
    let energy = 0;
    let targetEnergy = 0;
    let t = Math.random() * 100;
    let frame = 0;
    let raf = 0;

    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function resize() {
      const r = canvas!.getBoundingClientRect();
      const w = Math.ceil(r.width);
      const h = Math.ceil(r.height);
      if (canvas!.width !== w || canvas!.height !== h) {
        W = canvas!.width = w;
        H = canvas!.height = h;
      }
    }

    function paint() {
      resize();
      if (!W || !H) return;
      const c = ctx!;

      c.fillStyle = '#10181b';
      c.fillRect(0, 0, W, H);
      c.fillStyle = '#8094BC';

      const lensOn = energy > 0.02;

      for (let y = GAP / 2; y < H; y += GAP) {
        for (let x = GAP / 2; x < W; x += GAP) {
          // two interfering waves — the moiré comes from their beat, not noise
          const w1 = Math.sin(x * 0.011 + t) * Math.sin(y * 0.017 - t * 0.8);
          const w2 = Math.sin((x + y) * 0.006 + t * 0.6);
          const v = (w1 + w2 + 2) / 4; // 0..1

          let r = v * v * (GAP * 0.48);
          let a = Math.min(1, 0.16 + v * 0.5);
          let px = x;
          let py = y;
          let hot = 0;

          if (lensOn) {
            const dx = x - cx;
            const dy = y - cy;
            const d2 = dx * dx + dy * dy;
            if (d2 < LENS_R2) {
              const f = 1 - d2 / LENS_R2;
              const inf = f * f * f * energy; // cubic → the rim fades to nothing
              r += inf * GAP * 0.24; // gentle bloom
              a = Math.min(0.8, a + inf * 0.18);
              const d = Math.sqrt(d2) || 1; // slight magnetic push
              px += (dx / d) * inf * 4;
              py += (dy / d) * inf * 4;
              hot = inf;
            }
          }

          if (r < 0.3) continue;

          // continuous grey → brand-blue blend, so the lens dissolves into the
          // surrounding field with no seam
          if (hot > 0.01) {
            const m = hot * 0.85;
            c.fillStyle =
              'rgb(' +
              ((128 + (76 - 128) * m) | 0) +
              ',' +
              ((148 + (126 - 148) * m) | 0) +
              ',' +
              ((188 + (255 - 188) * m) | 0) +
              ')';
          }

          c.globalAlpha = a;
          c.beginPath();
          c.arc(px, py, Math.min(r, GAP * 0.6), 0, 6.2832);
          c.fill();

          if (hot > 0.01) c.fillStyle = '#8094BC';
        }
      }

      c.globalAlpha = 1;
    }

    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      tx = e.clientX - r.left;
      ty = e.clientY - r.top;
      if (energy < 0.05) {
        // the lens appears where the pointer entered, rather than flying in
        cx = tx;
        cy = ty;
      }
      targetEnergy = 1;
    };
    const onLeave = () => {
      targetEnergy = 0;
    };
    const onResize = () => paint();

    // listen on the banner, not the canvas, so overlaid copy still feeds the
    // effect as the pointer crosses it
    const interactive = lens && !reduce && host;
    if (interactive) {
      host.addEventListener('pointermove', onMove);
      host.addEventListener('pointerleave', onLeave);
    }

    if (reduce) {
      paint();
      window.addEventListener('resize', onResize);
    } else {
      const draw = () => {
        raf = requestAnimationFrame(draw);
        if (frame++ % 2) return; // 30fps is plenty for texture
        t += 0.012 * speed;
        cx += (tx - cx) * 0.16;
        cy += (ty - cy) * 0.16;
        energy += (targetEnergy - energy) * 0.16;
        paint();
      };
      draw();
    }

    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      if (interactive) {
        host.removeEventListener('pointermove', onMove);
        host.removeEventListener('pointerleave', onLeave);
      }
    };
  }, [speed, lens]);

  return <canvas ref={ref} className="sd-halftone" aria-hidden />;
}
