import { describe, expect, it } from "vitest";
import {
  boxToCanvasRect,
  canvasPointToImage,
  clampZoom,
  fitScale,
  imageToCanvasTransform,
  pointInRect,
  type Viewport,
} from "./scale";

const view: Viewport = {
  canvasWidth: 800,
  canvasHeight: 400,
  imageWidth: 1000,
  imageHeight: 1000,
  zoom: 1,
  offsetX: 0,
  offsetY: 0,
};

describe("fitScale", () => {
  it("fits the limiting dimension", () => {
    expect(fitScale(view)).toBe(0.4); // height-limited: 400 / 1000
  });

  it("does not divide by zero before the image loads", () => {
    expect(fitScale({ ...view, imageWidth: 0, imageHeight: 0 })).toBe(1);
  });
});

describe("imageToCanvasTransform", () => {
  it("centres the image inside the canvas", () => {
    const t = imageToCanvasTransform(view);
    expect(t.scale).toBe(0.4);
    expect(t.offsetX).toBe(200); // (800 - 400) / 2
    expect(t.offsetY).toBe(0);
  });

  it("applies zoom and pan", () => {
    const t = imageToCanvasTransform({ ...view, zoom: 2, offsetX: 30, offsetY: -10 });
    expect(t.scale).toBe(0.8);
    expect(t.offsetX).toBe(30); // (800 - 800) / 2 + 30
    expect(t.offsetY).toBe(-210);
  });
});

describe("round tripping", () => {
  it("canvas -> image -> canvas is the identity", () => {
    const t = imageToCanvasTransform({ ...view, zoom: 1.75, offsetX: 12, offsetY: 34 });
    const [ix, iy] = canvasPointToImage(321, 123, t);
    const [cx, cy] = [ix * t.scale + t.offsetX, iy * t.scale + t.offsetY];
    expect(cx).toBeCloseTo(321);
    expect(cy).toBeCloseTo(123);
  });
});

describe("boxToCanvasRect", () => {
  it("maps an original-pixel box onto the canvas", () => {
    const t = imageToCanvasTransform(view);
    const rect = boxToCanvasRect([100, 200, 300, 700], t);
    expect(rect).toEqual({ x: 240, y: 80, width: 80, height: 200 });
  });

  it("hit-tests the drawn rectangle", () => {
    const t = imageToCanvasTransform(view);
    const rect = boxToCanvasRect([100, 200, 300, 700], t);
    expect(pointInRect(250, 100, rect)).toBe(true);
    expect(pointInRect(239, 100, rect)).toBe(false);
  });
});

describe("clampZoom", () => {
  it("keeps zoom within bounds", () => {
    expect(clampZoom(0.1)).toBe(0.5);
    expect(clampZoom(99)).toBe(8);
    expect(clampZoom(2)).toBe(2);
  });
});
