import type { Detection, Prediction } from "@/api/types";
import { formatPercent, scoreColor } from "@/lib/scale";
import { Chip, EmptyState } from "./ui";

export function VerdictCard({ prediction }: { prediction: Prediction }) {
  const positive = prediction.has_foreign_object;
  return (
    <div
      className={`animate-fadeIn rounded-xl border p-4 ${
        positive ? "border-alert/40 bg-alert/[0.07]" : "border-clear/40 bg-clear/[0.07]"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="label">Verdict</p>
          <p
            className={`mt-1 text-lg font-semibold ${positive ? "text-alert" : "text-clear"}`}
          >
            {positive ? "Foreign object detected" : "No foreign object"}
          </p>
        </div>
        {prediction.source === "demo" && <Chip tone="warn">demo data</Chip>}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3">
        <div>
          <p className="label">Image score</p>
          <p className="stat mt-1 text-lg">{formatPercent(prediction.image_score)}</p>
        </div>
        <div>
          <p className="label">Objects</p>
          <p className="stat mt-1 text-lg">{prediction.detection_count}</p>
        </div>
        <div>
          <p className="label">Latency</p>
          <p className="stat mt-1 text-lg">
            {prediction.source === "demo" ? "—" : `${prediction.inference_ms.toFixed(0)} ms`}
          </p>
        </div>
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-ink-700">
        <div
          className={`h-full rounded-full transition-all ${positive ? "bg-alert" : "bg-clear"}`}
          style={{ width: `${Math.max(2, prediction.image_score * 100)}%` }}
        />
      </div>
      <p className="mt-2 text-[11px] text-slate-500">
        {prediction.model} · threshold {prediction.conf_threshold.toFixed(2)} ·{" "}
        {prediction.image_width}×{prediction.image_height}px
      </p>
    </div>
  );
}

export function DetectionList({
  detections,
  activeIndex,
  onHover,
}: {
  detections: Detection[];
  activeIndex: number | null;
  onHover: (index: number | null) => void;
}) {
  if (!detections.length) {
    return <EmptyState title="No detections above the threshold" />;
  }
  return (
    <ul className="divide-y divide-white/5">
      {detections.map((det, index) => (
        <li key={index}>
          <button
            type="button"
            onMouseEnter={() => onHover(index)}
            onMouseLeave={() => onHover(null)}
            onFocus={() => onHover(index)}
            className={`flex w-full items-center gap-3 px-1 py-2.5 text-left transition ${
              activeIndex === index ? "bg-white/5" : "hover:bg-white/[0.03]"
            }`}
          >
            <span
              className="h-8 w-1 shrink-0 rounded-full"
              style={{ background: scoreColor(det.score) }}
            />
            <span className="flex-1">
              <span className="block text-sm text-slate-200">
                #{index + 1} {det.label.replace("_", " ")}
              </span>
              <span className="block font-mono text-[11px] text-slate-500">
                centre ({det.center_x.toFixed(0)}, {det.center_y.toFixed(0)}) ·{" "}
                {(det.x_max - det.x_min).toFixed(0)}×{(det.y_max - det.y_min).toFixed(0)}px
              </span>
            </span>
            <span className="stat">{formatPercent(det.score, 0)}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

export function SetupCard({ device }: { device: string }) {
  return (
    <div className="animate-fadeIn rounded-xl border border-warn/30 bg-warn/[0.06] p-4">
      <p className="text-sm font-semibold text-warn">No trained checkpoint installed</p>
      <p className="mt-1 text-xs leading-relaxed text-slate-400">
        The API is running on <span className="font-mono text-slate-300">{device}</span> but neither
        detector has weights, so live inference is unavailable. The bundled synthetic samples still
        work and are served from recorded detections, badged{" "}
        <span className="text-warn">demo data</span>.
      </p>
      <div className="mt-3 space-y-1.5 font-mono text-[11px] text-slate-400">
        <p className="text-slate-500"># train on a machine with the dataset + a GPU</p>
        <p>cxr train frcnn --config configs/frcnn.yaml</p>
        <p>cxr train yolo --config configs/yolo.yaml</p>
        <p className="pt-1 text-slate-500"># or drop existing checkpoints in place</p>
        <p>models/frcnn/model.pt models/yolo/best.pt</p>
      </div>
    </div>
  );
}
