import { useRef, useState } from "react";
import type { ModelInfo, Sample } from "@/api/types";
import { Chip } from "./ui";

export function ModelPicker({
  models,
  value,
  onChange,
}: {
  models: ModelInfo[];
  value: string;
  onChange: (name: string) => void;
}) {
  return (
    <div className="space-y-2">
      <label className="label" htmlFor="model-picker">
        Detector
      </label>
      <div className="grid gap-2" id="model-picker">
        {models.map((model) => {
          const selected = model.name === value;
          return (
            <button
              key={model.name}
              type="button"
              onClick={() => onChange(model.name)}
              aria-pressed={selected}
              className={`rounded-lg border p-3 text-left transition ${
                selected
                  ? "border-accent/50 bg-accent/10"
                  : "border-white/10 bg-white/[0.02] hover:border-white/20"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-slate-200">{model.display_name}</span>
                {model.ready ? (
                  <Chip tone="clear">ready</Chip>
                ) : model.dependencies_available ? (
                  <Chip tone="warn">no weights</Chip>
                ) : (
                  <Chip tone="warn">not installed</Chip>
                )}
              </div>
              <p className="mt-1 text-xs leading-relaxed text-slate-500">{model.description}</p>
              <p className="mt-1 font-mono text-[10px] text-slate-600">
                input {model.input_size}px · default conf {model.default_conf}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function ConfidenceSlider({
  value,
  onChange,
}: {
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <label className="label" htmlFor="conf">
          Confidence threshold
        </label>
        <span className="stat">{value.toFixed(2)}</span>
      </div>
      <input
        id="conf"
        type="range"
        min={0.05}
        max={0.95}
        step={0.05}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <p className="text-[11px] text-slate-500">
        Boxes below this score are hidden; the image is called positive when the best box clears it.
      </p>
    </div>
  );
}

export function UploadZone({
  onFiles,
  multiple = false,
  hint,
}: {
  onFiles: (files: File[]) => void;
  multiple?: boolean;
  hint?: string;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        const files = Array.from(event.dataTransfer.files).filter((f) =>
          f.type.startsWith("image/"),
        );
        if (files.length) onFiles(multiple ? files : [files[0]]);
      }}
      className={`rounded-lg border-2 border-dashed p-6 text-center transition ${
        over ? "border-accent bg-accent/10" : "border-white/10 hover:border-white/20"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple={multiple}
        className="sr-only"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          if (files.length) onFiles(files);
          event.target.value = "";
        }}
      />
      <p className="text-sm text-slate-300">
        Drop {multiple ? "X-rays" : "an X-ray"} here, or{" "}
        <button
          type="button"
          className="font-medium text-accent underline-offset-2 hover:underline"
          onClick={() => inputRef.current?.click()}
        >
          browse
        </button>
      </p>
      <p className="mt-1 text-[11px] text-slate-500">{hint ?? "JPEG, PNG · up to 20 MB"}</p>
    </div>
  );
}

export function SampleStrip({
  samples,
  onPick,
  activeId,
}: {
  samples: Sample[];
  onPick: (sample: Sample) => void;
  activeId?: string | null;
}) {
  if (!samples.length) return null;
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="label">Sample images</span>
        <span className="text-[10px] text-slate-600">synthetic phantoms</span>
      </div>
      <div className="grid grid-cols-5 gap-1.5">
        {samples.map((sample) => (
          <button
            key={sample.id}
            type="button"
            title={`${sample.title} — ${sample.description}`}
            onClick={() => onPick(sample)}
            className={`overflow-hidden rounded-md border transition ${
              activeId === sample.id
                ? "border-accent ring-1 ring-accent/40"
                : "border-white/10 hover:border-white/30"
            }`}
          >
            <img src={sample.url} alt={sample.title} className="aspect-square object-cover" />
          </button>
        ))}
      </div>
    </div>
  );
}
