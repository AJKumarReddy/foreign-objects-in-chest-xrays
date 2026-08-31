import { useState } from "react";
import { ApiError, api } from "@/api/client";
import type { CompareResult, ModelList, Sample } from "@/api/types";
import { ConfidenceSlider, SampleStrip, UploadZone } from "@/components/Controls";
import { ImageCanvas } from "@/components/ImageCanvas";
import { VerdictCard } from "@/components/Results";
import { Chip, EmptyState, ErrorNote, Panel } from "@/components/ui";
import { useObjectUrl } from "@/hooks/useModels";

export function ComparePage({ backend, samples }: { backend: ModelList; samples: Sample[] }) {
  const [file, setFile] = useState<File | null>(null);
  const [conf, setConf] = useState(0.5);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ message: string; hint?: string } | null>(null);
  // one shared viewport keeps the two canvases locked together
  const [view, setView] = useState({ zoom: 1, offsetX: 0, offsetY: 0 });

  const url = useObjectUrl(file);

  const run = async (target: File, threshold = conf) => {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.compare(target, threshold));
    } catch (err) {
      setResult(null);
      setError(
        err instanceof ApiError ? { message: err.message, hint: err.hint } : { message: String(err) },
      );
    } finally {
      setBusy(false);
    }
  };

  const pickSample = async (sample: Sample) => {
    const asFile = await api.sampleFile(sample);
    setFile(asFile);
    void run(asFile);
  };

  const agreementTone = (): "clear" | "alert" | "warn" => {
    if (result?.agreement === "both") return "alert";
    if (result?.agreement === "neither") return "clear";
    return "warn";
  };

  return (
    <div className="space-y-4">
      <Panel title="Compare detectors on one image">
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_260px_200px]">
          <UploadZone
            onFiles={(files) => {
              setFile(files[0]);
              void run(files[0]);
            }}
          />
          <ConfidenceSlider
            value={conf}
            onChange={(value) => {
              setConf(value);
              if (file) void run(file, value);
            }}
          />
          <SampleStrip samples={samples.slice(0, 5)} onPick={pickSample} />
        </div>
      </Panel>

      {error && <ErrorNote message={error.message} hint={error.hint} />}

      {!result && !error && (
        <EmptyState title="Load an image to compare Faster R-CNN and YOLO side by side">
          Both models run on the same pixels; zoom and pan stay synchronised.
        </EmptyState>
      )}

      {result && (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <Chip tone={agreementTone()}>
              {result.agreement === "both"
                ? "both models: foreign object"
                : result.agreement === "neither"
                  ? "both models: clear"
                  : `disagreement (${result.agreement.replace("_only", " only")})`}
            </Chip>
            <span className="text-[11px] text-slate-500">
              {result.filename} · {result.image_width}×{result.image_height}
            </span>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {result.results.map((prediction) => (
              <div key={prediction.model} className="space-y-3">
                <Panel
                  title={
                    backend.models.find((m) => m.name === prediction.model)?.display_name ??
                    prediction.model
                  }
                  bodyClassName="p-0"
                  actions={
                    <span className="font-mono text-[11px] text-slate-500">
                      {prediction.detection_count} obj ·{" "}
                      {prediction.source === "demo" ? "demo" : `${prediction.inference_ms.toFixed(0)}ms`}
                    </span>
                  }
                >
                  <div className="h-[clamp(280px,42vh,460px)] bg-black/40">
                    <ImageCanvas
                      src={url}
                      detections={prediction.detections}
                      externalView={view}
                      onViewChange={setView}
                      busy={busy}
                    />
                  </div>
                </Panel>
                <VerdictCard prediction={prediction} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
