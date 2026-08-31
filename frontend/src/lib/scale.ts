/**
 * Coordinate mapping between original image pixels and the canvas viewport.
 *
 * The backend always returns boxes in ORIGINAL image pixels, so every overlay
 * goes through here. Unit-tested because an off-by-one in this file silently
 * draws boxes in the wrong place.
 */

export interface Viewport {
  /** Canvas size in CSS pixels. */
  canvasWidth: number;
  canvasHeight: number;
  /** Natural image size in pixels. */
  imageWidth: number;
  imageHeight: number;
  /** User zoom multiplier applied on top of the fit-to-canvas scale. */
  zoom: number;
  /** Pan offset in canvas pixels. */
  offsetX: number;
  offsetY: number;
}

export interface Transform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/** Scale that fits the whole image inside the canvas ("contain"). */
export function fitScale(view: Viewport): number {
  if (!view.imageWidth || !view.imageHeight) return 1;
  return Math.min(view.canvasWidth / view.imageWidth, view.canvasHeight / view.imageHeight);
}

/** The full image -> canvas transform, centring the image and applying pan/zoom. */
export function imageToCanvasTransform(view: Viewport): Transform {
  const scale = fitScale(view) * view.zoom;
  return {
    scale,
    offsetX: (view.canvasWidth - view.imageWidth * scale) / 2 + view.offsetX,
    offsetY: (view.canvasHeight - view.imageHeight * scale) / 2 + view.offsetY,
  };
}

export function imagePointToCanvas(x: number, y: number, t: Transform): [number, number] {
  return [x * t.scale + t.offsetX, y * t.scale + t.offsetY];
}

export function canvasPointToImage(x: number, y: number, t: Transform): [number, number] {
  return [(x - t.offsetX) / t.scale, (y - t.offsetY) / t.scale];
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Map an [x_min, y_min, x_max, y_max] image box to a canvas rectangle. */
export function boxToCanvasRect(
  box: [number, number, number, number],
  t: Transform,
): Rect {
  const [x0, y0] = imagePointToCanvas(box[0], box[1], t);
  const [x1, y1] = imagePointToCanvas(box[2], box[3], t);
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

export function pointInRect(px: number, py: number, rect: Rect): boolean {
  return (
    px >= Math.min(rect.x, rect.x + rect.width) &&
    px <= Math.max(rect.x, rect.x + rect.width) &&
    py >= Math.min(rect.y, rect.y + rect.height) &&
    py <= Math.max(rect.y, rect.y + rect.height)
  );
}

export function clampZoom(zoom: number, min = 0.5, max = 8): number {
  return Math.min(max, Math.max(min, zoom));
}

/** Colour ramp for a confidence score: amber (low) -> rose (high). */
export function scoreColor(score: number): string {
  if (score >= 0.75) return "#fb7185";
  if (score >= 0.5) return "#fbbf24";
  return "#38bdf8";
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}
