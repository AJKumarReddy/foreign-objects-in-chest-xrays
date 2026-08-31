import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "@/api/client";
import type { ModelList, Prediction, Sample } from "@/api/types";
import { ConfidenceSlider, ModelPicker, SampleStrip, UploadZone } from "@/components/Controls";
import { ImageCanvas } from "@/components/ImageCanvas";
import { DetectionList, SetupCard, VerdictCard } from "@/components/Results";
import { ErrorNote, Panel } from "@/components/ui";
import { useObjectUrl } from "@/hooks/useModels";

interface Props {
  backend: ModelList;
  samples: Sample[];
}

export function DetectPage({ backend, samples }: Props) {
  const [model, setModel] = useState(
    () => backend.models.find((m) => m.ready)?.name ?? backend.models[0]?.name ?? "frcnn",
  );
  const [conf, setConf] = useState(
    () => backend.models.find((m) => m.name === model)?.default_conf ?? 0.5,
  );
  const [file, setFile] = useState<File | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ message: string; hint?: string } | null>(null);
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showCenters, setShowCenters] = useState(false);

  const uploadUrl = useObjectUrl(file);
  const sampleUrl = samples.find((s) => s.id === sampleId)?.url ?? null;
  const imageUrl = file ? uploadUrl : sampleUrl;

  const run = useCallback(async () => {
    if (!file && !sampleId) return;
    setBusy(true);
    setError(null);
    try {
      const result = file
        ? await api.predict(file, model, conf)
        : await api.predictSample(sampleId!, model, conf);
      setPrediction(result);
    } catch (err) {
      setPrediction(null);
      setError(
        err instanceof ApiError
          ? { message: err.message, hint: err.hint }
          : { message: String(err) },
      );
    } finally {
      setBusy(false);
    }
  }, [file, sampleId, model, conf]);

  // re-run automatically when the inputs change
  useEffect(() => {
    if (file || sampleId) void run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, sampleId, model, conf]);

  // land on a sample so the viewer is never empty on arrival
  useEffect(() => {
    if (!file && !sampleId && samples.length) setSampleId(samples[0].id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [samples]);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement) return;
      if (event.key === "b") setShowBoxes((value) => !value);
      if (event.key === "c") setShowCenters((value) => !value);
      if (event.key === "[" || event.key === "]") {
        const count = prediction?.detections.length ?? 0;
        if (!count) return;
        setActiveIndex((current) => {
          const base = current ?? -1;
          const next = event.key === "]" ? base + 1 : base - 1 + count;
          return next % count;
        });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [prediction]);

  const pickSample = async (sample: Sample) => {
    setFile(null);
    setSampleId(sample.id);
  };

  return (
    <div className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
      <div className="space-y-4">
        <Panel title="Input">
          <div className="space-y-4">
            <UploadZone
              onFiles={(files) => {
                setFile(files[0]);
                setSampleId(null);
              }}
            />
            <SampleStrip samples={samples} onPick={pickSample} activeId={sampleId} />
          </div>
        </Panel>

        <Panel title="Settings">
          <div className="space-y-5">
            <ModelPicker
              models={backend.models}
              value={model}
              onChange={(name) => {
                setModel(name);
                const next = backend.models.find((m) => m.name === name);
                if (next) setConf(next.default_conf);
              }}
            />
            <ConfidenceSlider value={conf} onChange={setConf} />
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                className="btn-ghost"
                onClick={() => setShowBoxes((v) => !v)}
                aria-pressed={showBoxes}
              >
                {showBoxes ? "Hide" : "Show"} boxes <kbd className="text-[10px]">b</kbd>
              </button>
              <button
                type="button"
                className="btn-ghost"
                onClick={() => setShowCenters((v) => !v)}
                aria-pressed={showCenters}
              >
                Centres <kbd className="text-[10px]">c</kbd>
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={run}
                disabled={busy || (!file && !sampleId)}
              >
                {busy ? "Detecting…" : "Detect"}
              </button>
            </div>
          </div>
        </Panel>

        {!backend.any_ready && <SetupCard device={backend.device} />}
      </div>

      <div className="space-y-4">
        <Panel
          title={file?.name ?? samples.find((s) => s.id === sampleId)?.title ?? "Viewer"}
          bodyClassName="p-0"
          actions={
            prediction && (
              <span className="font-mono text-[11px] text-slate-500">
                {prediction.image_width}×{prediction.image_height}
              </span>
            )
          }
        >
          <div className="h-[clamp(340px,54vh,620px)] bg-black/40">
            <ImageCanvas
              src={imageUrl}
              detections={prediction?.detections ?? []}
              showBoxes={showBoxes}
              showCenters={showCenters}
              activeIndex={activeIndex}
              onHover={setActiveIndex}
              busy={busy}
            />
          </div>
        </Panel>

        {error && <ErrorNote message={error.message} hint={error.hint} />}

        {prediction && (
          <div className="grid gap-4 md:grid-cols-2">
            <VerdictCard prediction={prediction} />
            <Panel title={`Detections (${prediction.detection_count})`} bodyClassName="px-3 py-1">
              <DetectionList
                detections={prediction.detections}
                activeIndex={activeIndex}
                onHover={setActiveIndex}
              />
            </Panel>
          </div>
        )}
      </div>
    </div>
  );
}
