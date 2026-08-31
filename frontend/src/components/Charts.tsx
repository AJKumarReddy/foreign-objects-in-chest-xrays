import { useState } from "react";
import type { RocPoint, SweepPoint } from "@/api/types";

/**
 * Inline SVG charts.
 *
 * Palette: validated categorical slots 1-3 stepped for a dark surface
 * (blue / orange / aqua). Checked with the dataviz validator against the
 * #0b1120 panel surface - all-pairs CVD dE 9.4, normal-vision dE 20.9, >= 3:1
 * contrast. Do not swap these for arbitrary hues.
 */
export const SERIES = {
  s1: "#3987e5",
  s2: "#d95926",
  s3: "#199e70",
} as const;

const AXIS = "#475569";
const GRID = "rgba(148, 163, 184, 0.12)";
const INK = "#cbd5e1";

interface Box {
  width: number;
  height: number;
  pad: { top: number; right: number; bottom: number; left: number };
}

const BOX: Box = { width: 420, height: 320, pad: { top: 16, right: 16, bottom: 40, left: 46 } };

function plotArea(box: Box) {
  return {
    x0: box.pad.left,
    y0: box.pad.top,
    x1: box.width - box.pad.right,
    y1: box.height - box.pad.bottom,
    w: box.width - box.pad.left - box.pad.right,
    h: box.height - box.pad.top - box.pad.bottom,
  };
}

function Axes({ box, xLabel, yLabel }: { box: Box; xLabel: string; yLabel: string }) {
  const a = plotArea(box);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <g>
      {ticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={a.x0} x2={a.x1} y1={a.y1 - t * a.h} y2={a.y1 - t * a.h} stroke={GRID} />
          <text
            x={a.x0 - 8}
            y={a.y1 - t * a.h + 3.5}
            textAnchor="end"
            fontSize="10"
            fill={AXIS}
            fontFamily="ui-monospace, monospace"
          >
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {ticks.map((t) => (
        <text
          key={`x${t}`}
          x={a.x0 + t * a.w}
          y={a.y1 + 15}
          textAnchor="middle"
          fontSize="10"
          fill={AXIS}
          fontFamily="ui-monospace, monospace"
        >
          {t.toFixed(2)}
        </text>
      ))}
      <text x={a.x0 + a.w / 2} y={box.height - 4} textAnchor="middle" fontSize="10" fill={AXIS}>
        {xLabel}
      </text>
      <text
        transform={`translate(11, ${a.y0 + a.h / 2}) rotate(-90)`}
        textAnchor="middle"
        fontSize="10"
        fill={AXIS}
      >
        {yLabel}
      </text>
    </g>
  );
}

export function RocChart({
  points,
  auc,
  operating,
}: {
  points: RocPoint[];
  auc: number;
  operating?: { fpr: number; tpr: number };
}) {
  const a = plotArea(BOX);
  const [hover, setHover] = useState<RocPoint | null>(null);

  const toX = (fpr: number) => a.x0 + fpr * a.w;
  const toY = (tpr: number) => a.y1 - tpr * a.h;
  const path = points.map((p, i) => `${i ? "L" : "M"}${toX(p.fpr)},${toY(p.tpr)}`).join(" ");
  const area = `${path} L${toX(points.at(-1)?.fpr ?? 1)},${a.y1} L${a.x0},${a.y1} Z`;

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * BOX.width;
    const fpr = Math.min(1, Math.max(0, (x - a.x0) / a.w));
    const nearest = points.reduce((best, p) =>
      Math.abs(p.fpr - fpr) < Math.abs(best.fpr - fpr) ? p : best,
    );
    setHover(nearest);
  };

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${BOX.width} ${BOX.height}`}
        className="w-full"
        role="img"
        aria-label={`ROC curve, area under curve ${auc.toFixed(3)}`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <Axes box={BOX} xLabel="False positive rate" yLabel="True positive rate" />
        {/* chance reference */}
        <line
          x1={a.x0}
          y1={a.y1}
          x2={a.x1}
          y2={a.y0}
          stroke={AXIS}
          strokeDasharray="4 4"
          strokeWidth="1"
        />
        <path d={area} fill={SERIES.s1} opacity="0.12" />
        <path d={path} fill="none" stroke={SERIES.s1} strokeWidth="2" strokeLinejoin="round" />

        {operating && (
          <g>
            <circle
              cx={toX(operating.fpr)}
              cy={toY(operating.tpr)}
              r="4.5"
              fill={SERIES.s1}
              stroke="#0b1120"
              strokeWidth="2"
            />
            <text
              x={toX(operating.fpr) + 8}
              y={toY(operating.tpr) + 12}
              fontSize="10"
              fill={INK}
              fontFamily="ui-monospace, monospace"
            >
              operating point
            </text>
          </g>
        )}

        {hover && (
          <g pointerEvents="none">
            <line x1={toX(hover.fpr)} x2={toX(hover.fpr)} y1={a.y0} y2={a.y1} stroke={GRID} />
            <circle cx={toX(hover.fpr)} cy={toY(hover.tpr)} r="4" fill={SERIES.s1} stroke="#0b1120" strokeWidth="2" />
            <rect
              x={Math.min(toX(hover.fpr) + 8, a.x1 - 108)}
              y={toY(hover.tpr) - 34}
              width="104"
              height="30"
              rx="4"
              fill="#0b1120"
              stroke="rgba(148,163,184,0.25)"
            />
            <text
              x={Math.min(toX(hover.fpr) + 16, a.x1 - 100)}
              y={toY(hover.tpr) - 21}
              fontSize="10"
              fill={INK}
              fontFamily="ui-monospace, monospace"
            >
              TPR {hover.tpr.toFixed(3)}
            </text>
            <text
              x={Math.min(toX(hover.fpr) + 16, a.x1 - 100)}
              y={toY(hover.tpr) - 10}
              fontSize="10"
              fill={AXIS}
              fontFamily="ui-monospace, monospace"
            >
              FPR {hover.fpr.toFixed(3)}
            </text>
          </g>
        )}

        <text x={a.x1 - 4} y={a.y0 + 14} textAnchor="end" fontSize="13" fill={INK} fontWeight="600">
          AUC {auc.toFixed(3)}
        </text>
      </svg>
    </figure>
  );
}

const SWEEP_SERIES = [
  { key: "accuracy", label: "Accuracy", color: SERIES.s1 },
  { key: "sensitivity", label: "Sensitivity", color: SERIES.s2 },
  { key: "specificity", label: "Specificity", color: SERIES.s3 },
] as const;

export function SweepChart({ sweep, threshold }: { sweep: SweepPoint[]; threshold: number }) {
  const a = plotArea(BOX);
  const [hover, setHover] = useState<SweepPoint | null>(null);

  const toX = (t: number) => a.x0 + t * a.w;
  const toY = (v: number) => a.y1 - v * a.h;

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * BOX.width;
    const t = Math.min(1, Math.max(0, (x - a.x0) / a.w));
    setHover(
      sweep.reduce((best, p) => (Math.abs(p.threshold - t) < Math.abs(best.threshold - t) ? p : best)),
    );
  };

  return (
    <figure className="m-0">
      <div className="mb-2 flex flex-wrap gap-3">
        {SWEEP_SERIES.map((series) => (
          <span key={series.key} className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <span className="h-2 w-2 rounded-full" style={{ background: series.color }} />
            {series.label}
          </span>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${BOX.width} ${BOX.height}`}
        className="w-full"
        role="img"
        aria-label="Accuracy, sensitivity and specificity across confidence thresholds"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        <Axes box={BOX} xLabel="Confidence threshold" yLabel="Rate" />
        <line
          x1={toX(threshold)}
          x2={toX(threshold)}
          y1={a.y0}
          y2={a.y1}
          stroke={AXIS}
          strokeDasharray="3 3"
        />
        {SWEEP_SERIES.map((series) => {
          const path = sweep
            .map((p, i) => `${i ? "L" : "M"}${toX(p.threshold)},${toY(p[series.key])}`)
            .join(" ");
          const last = sweep.at(-1);
          return (
            <g key={series.key}>
              <path d={path} fill="none" stroke={series.color} strokeWidth="2" strokeLinejoin="round" />
              {last && (
                <circle cx={toX(last.threshold)} cy={toY(last[series.key])} r="3" fill={series.color} />
              )}
            </g>
          );
        })}

        {hover && (
          <g pointerEvents="none">
            <line x1={toX(hover.threshold)} x2={toX(hover.threshold)} y1={a.y0} y2={a.y1} stroke={GRID} />
            {SWEEP_SERIES.map((series) => (
              <circle
                key={series.key}
                cx={toX(hover.threshold)}
                cy={toY(hover[series.key])}
                r="4"
                fill={series.color}
                stroke="#0b1120"
                strokeWidth="2"
              />
            ))}
            <g transform={`translate(${Math.min(toX(hover.threshold) + 10, a.x1 - 118)}, ${a.y0 + 6})`}>
              <rect width="114" height="56" rx="4" fill="#0b1120" stroke="rgba(148,163,184,0.25)" />
              <text x="8" y="15" fontSize="10" fill={INK} fontFamily="ui-monospace, monospace">
                thr {hover.threshold.toFixed(2)}
              </text>
              {SWEEP_SERIES.map((series, index) => (
                <text
                  key={series.key}
                  x="8"
                  y={28 + index * 11}
                  fontSize="10"
                  fill={series.color}
                  fontFamily="ui-monospace, monospace"
                >
                  {series.label.slice(0, 4).toLowerCase()} {hover[series.key].toFixed(3)}
                </text>
              ))}
            </g>
          </g>
        )}
      </svg>
    </figure>
  );
}

export function FrocChart({ points }: { points: { fps_per_image: number; sensitivity: number }[] }) {
  const box: Box = { ...BOX, height: 240 };
  const a = plotArea(box);
  const maxFp = Math.max(...points.map((p) => p.fps_per_image), 1);
  const toX = (fp: number) => a.x0 + (Math.log2(fp + 0.125) - Math.log2(0.25)) / (Math.log2(maxFp + 0.125) - Math.log2(0.25)) * a.w;
  const toY = (s: number) => a.y1 - s * a.h;
  const path = points.map((p, i) => `${i ? "L" : "M"}${toX(p.fps_per_image)},${toY(p.sensitivity)}`).join(" ");

  return (
    <figure className="m-0">
      <svg viewBox={`0 0 ${box.width} ${box.height}`} className="w-full" role="img" aria-label="FROC curve">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={a.x0} x2={a.x1} y1={a.y1 - t * a.h} y2={a.y1 - t * a.h} stroke={GRID} />
            <text x={a.x0 - 8} y={a.y1 - t * a.h + 3.5} textAnchor="end" fontSize="10" fill={AXIS} fontFamily="ui-monospace, monospace">
              {t.toFixed(2)}
            </text>
          </g>
        ))}
        {points.map((p) => (
          <text key={p.fps_per_image} x={toX(p.fps_per_image)} y={a.y1 + 14} textAnchor="middle" fontSize="9" fill={AXIS} fontFamily="ui-monospace, monospace">
            {p.fps_per_image}
          </text>
        ))}
        <path d={path} fill="none" stroke={SERIES.s3} strokeWidth="2" />
        {points.map((p) => (
          <circle key={p.fps_per_image} cx={toX(p.fps_per_image)} cy={toY(p.sensitivity)} r="3.5" fill={SERIES.s3}>
            <title>{`${p.fps_per_image} FP/image → sensitivity ${p.sensitivity.toFixed(3)}`}</title>
          </circle>
        ))}
        <text x={a.x0 + a.w / 2} y={box.height - 3} textAnchor="middle" fontSize="10" fill={AXIS}>
          Average false positives per image
        </text>
      </svg>
    </figure>
  );
}
