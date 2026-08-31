import { useCallback, useEffect, useRef, useState } from "react";
import type { Detection } from "@/api/types";
import {
  boxToCanvasRect,
  clampZoom,
  imageToCanvasTransform,
  pointInRect,
  scoreColor,
  type Viewport,
} from "@/lib/scale";

interface Props {
  src: string | null;
  detections: Detection[];
  showBoxes?: boolean;
  showCenters?: boolean;
  activeIndex?: number | null;
  onHover?: (index: number | null) => void;
  /** Zoom/pan shared with another canvas (compare view). */
  externalView?: { zoom: number; offsetX: number; offsetY: number };
  onViewChange?: (view: { zoom: number; offsetX: number; offsetY: number }) => void;
  className?: string;
  busy?: boolean;
}

/**
 * Draws the X-ray plus its detection overlay on a canvas.
 *
 * Boxes arrive in original image pixels; every draw goes through the transform
 * in lib/scale so overlays stay aligned at any zoom level.
 */
export function ImageCanvas({
  src,
  detections,
  showBoxes = true,
  showCenters = false,
  activeIndex = null,
  onHover,
  externalView,
  onViewChange,
  className = "",
  busy = false,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);

  const [internalView, setInternalView] = useState({ zoom: 1, offsetX: 0, offsetY: 0 });
  const view = externalView ?? internalView;
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [loaded, setLoaded] = useState(false);

  const setView = useCallback(
    (next: { zoom: number; offsetX: number; offsetY: number }) => {
      if (onViewChange) onViewChange(next);
      else setInternalView(next);
    },
    [onViewChange],
  );

  // keep the backing store in sync with the element's CSS size
  useEffect(() => {
    const element = wrapRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.round(entry.contentRect.width),
        height: Math.round(entry.contentRect.height),
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!src) {
      imageRef.current = null;
      setLoaded(false);
      return;
    }
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setLoaded(true);
    };
    image.src = src;
    setView({ zoom: 1, offsetX: 0, offsetY: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src]);

  const viewport = useCallback((): Viewport => {
    const image = imageRef.current;
    return {
      canvasWidth: size.width,
      canvasHeight: size.height,
      imageWidth: image?.naturalWidth ?? 0,
      imageHeight: image?.naturalHeight ?? 0,
      zoom: view.zoom,
      offsetX: view.offsetX,
      offsetY: view.offsetY,
    };
  }, [size, view]);

  // draw
  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !size.width || !size.height) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.width, size.height);

    if (!image || !loaded) return;
    const t = imageToCanvasTransform(viewport());

    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(
      image,
      t.offsetX,
      t.offsetY,
      image.naturalWidth * t.scale,
      image.naturalHeight * t.scale,
    );

    if (!showBoxes) return;

    detections.forEach((det, index) => {
      const rect = boxToCanvasRect([det.x_min, det.y_min, det.x_max, det.y_max], t);
      const color = scoreColor(det.score);
      const active = index === activeIndex;

      ctx.save();
      ctx.lineWidth = active ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.shadowColor = color;
      ctx.shadowBlur = active ? 14 : 0;
      ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
      ctx.shadowBlur = 0;

      ctx.globalAlpha = active ? 0.22 : 0.1;
      ctx.fillStyle = color;
      ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
      ctx.globalAlpha = 1;

      if (showCenters) {
        ctx.beginPath();
        ctx.arc(rect.x + rect.width / 2, rect.y + rect.height / 2, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
      }

      // score chip above the box (flipped inside when it would clip the top)
      const text = `${(det.score * 100).toFixed(0)}%`;
      ctx.font = "600 11px ui-monospace, monospace";
      const textWidth = ctx.measureText(text).width;
      const chipWidth = textWidth + 12;
      const chipHeight = 17;
      const chipY = rect.y - chipHeight - 3 < 0 ? rect.y + 3 : rect.y - chipHeight - 3;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(rect.x, chipY, chipWidth, chipHeight, 4);
      ctx.fill();
      ctx.fillStyle = "#0b1120";
      ctx.fillText(text, rect.x + 6, chipY + 12);
      ctx.restore();
    });
  }, [detections, size, view, loaded, showBoxes, showCenters, activeIndex, viewport]);

  const handleMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    const px = event.clientX - bounds.left;
    const py = event.clientY - bounds.top;

    if (dragRef.current) {
      setView({
        zoom: view.zoom,
        offsetX: dragRef.current.ox + (event.clientX - dragRef.current.x),
        offsetY: dragRef.current.oy + (event.clientY - dragRef.current.y),
      });
      return;
    }
    if (!onHover) return;
    const t = imageToCanvasTransform(viewport());
    const hit = detections.findIndex((det) =>
      pointInRect(px, py, boxToCanvasRect([det.x_min, det.y_min, det.x_max, det.y_max], t)),
    );
    onHover(hit === -1 ? null : hit);
  };

  const handleWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    setView({ ...view, zoom: clampZoom(view.zoom * (event.deltaY < 0 ? 1.12 : 0.89)) });
  };

  return (
    <div ref={wrapRef} className={`relative h-full w-full overflow-hidden ${className}`}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label="Chest X-ray with detection overlay"
        style={{ width: size.width, height: size.height }}
        className={`block ${dragRef.current ? "cursor-grabbing" : "cursor-grab"}`}
        onMouseDown={(e) => {
          dragRef.current = { x: e.clientX, y: e.clientY, ox: view.offsetX, oy: view.offsetY };
        }}
        onMouseUp={() => {
          dragRef.current = null;
        }}
        onMouseLeave={() => {
          dragRef.current = null;
          onHover?.(null);
        }}
        onMouseMove={handleMove}
        onWheel={handleWheel}
        onDoubleClick={() => setView({ zoom: 1, offsetX: 0, offsetY: 0 })}
      />

      {!src && (
        <div className="absolute inset-0 grid place-items-center text-sm text-slate-600">
          No image loaded
        </div>
      )}

      {busy && (
        <div className="absolute inset-0 overflow-hidden bg-ink-950/50 backdrop-blur-[1px]">
          <div className="h-16 w-full animate-sweep bg-gradient-to-b from-transparent via-accent/25 to-transparent" />
          <div className="absolute inset-0 grid place-items-center text-xs uppercase tracking-[0.2em] text-accent">
            Analysing
          </div>
        </div>
      )}

      {src && (
        <div className="pointer-events-none absolute bottom-2 right-2 flex gap-1.5 text-[10px] text-slate-500">
          <span className="rounded bg-ink-950/70 px-1.5 py-0.5 font-mono">
            {Math.round(view.zoom * 100)}%
          </span>
          <span className="rounded bg-ink-950/70 px-1.5 py-0.5">
            scroll to zoom · drag to pan · double-click to reset
          </span>
        </div>
      )}
    </div>
  );
}
