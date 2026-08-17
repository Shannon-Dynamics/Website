'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fitViewport,
  mutePalette,
  readPalette,
  toWorld,
  type Palette,
  type Viewport,
} from '@/lib/sim/draw';
import { useHoveredRole } from '@/lib/explorable/store';

export interface SimCanvasProps {
  /** World rectangle to fit into the canvas. */
  world: { minX: number; minY: number; maxX: number; maxY: number };
  /** Draw one frame. Called whenever `deps` change, on resize, and on theme flip. */
  draw: (ctx: CanvasRenderingContext2D, v: Viewport, p: Palette) => void;
  /** Values that should trigger a redraw (usually the simulation state/tick). */
  deps?: readonly unknown[];
  /** Canvas aspect ratio (width / height). */
  aspect?: number;
  padding?: number;
  className?: string;
  ariaLabel: string;
  onPointer?: (
    world: [number, number],
    phase: 'down' | 'move' | 'up',
    event: React.PointerEvent<HTMLCanvasElement>,
  ) => void;
  cursor?: string;
}

/**
 * A device-pixel-ratio-correct canvas with a world-coordinate viewport.
 *
 * Simulations draw in metres and radians; this component owns the mapping to
 * pixels, the resize handling, and re-reading the palette when the reader flips
 * theme — so no widget has to think about any of it.
 */
export function SimCanvas({
  world,
  draw,
  deps = [],
  aspect = 16 / 9,
  padding = 0.4,
  className,
  ariaLabel,
  onPointer,
  cursor,
}: SimCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 800 / aspect });
  const [paletteVersion, setPaletteVersion] = useState(0);
  // When the reader points at a term in an equation, every figure on the page
  // brings that role forward. Widgets need not know this happens.
  const hoveredRole = useHoveredRole();
  const drawRef = useRef(draw);
  drawRef.current = draw;

  // Track the container width; height follows the requested aspect.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = Math.max(entries[0].contentRect.width, 240);
      setSize({ w, h: w / aspect });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [aspect]);

  // Redraw when the reader switches between light and dark.
  useEffect(() => {
    const mo = new MutationObserver(() => setPaletteVersion((v) => v + 1));
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => mo.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size.w * dpr);
    canvas.height = Math.round(size.h * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const v = fitViewport(world, size.w, size.h, padding);
    const p = mutePalette(readPalette(canvas), hoveredRole);
    drawRef.current(ctx, v, p);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size.w, size.h, padding, paletteVersion, hoveredRole, world, ...deps]);

  const handlePointer = useCallback(
    (phase: 'down' | 'move' | 'up') => (e: React.PointerEvent<HTMLCanvasElement>) => {
      if (!onPointer) return;
      const rect = e.currentTarget.getBoundingClientRect();
      const v = fitViewport(world, rect.width, rect.height, padding);
      onPointer(toWorld(v, e.clientX - rect.left, e.clientY - rect.top), phase, e);
    },
    [onPointer, world, padding],
  );

  return (
    <div ref={wrapRef} className={className}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={ariaLabel}
        className="sim-canvas rounded-sm"
        // A pixel width here would become the element's min-content size, and a
        // grid track that sizes to content would then widen the whole page on a
        // phone. The bitmap is still sized from the measured width below.
        style={{
          width: '100%',
          maxWidth: '100%',
          height: size.h,
          cursor: cursor ?? (onPointer ? 'grab' : 'default'),
        }}
        onPointerDown={
          onPointer
            ? (e) => {
                e.currentTarget.setPointerCapture(e.pointerId);
                handlePointer('down')(e);
              }
            : undefined
        }
        onPointerMove={onPointer ? handlePointer('move') : undefined}
        onPointerUp={
          onPointer
            ? (e) => {
                e.currentTarget.releasePointerCapture(e.pointerId);
                handlePointer('up')(e);
              }
            : undefined
        }
      />
    </div>
  );
}
