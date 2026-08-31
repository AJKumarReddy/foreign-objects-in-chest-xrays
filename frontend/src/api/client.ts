import type {
  BatchResult,
  CompareResult,
  Health,
  Metrics,
  ModelList,
  Prediction,
  Sample,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

/** An API error carrying the backend's actionable hint, when there is one. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly hint?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;

  let message = `${response.status} ${response.statusText}`;
  let hint: string | undefined;
  try {
    const body = await response.json();
    const detail = body?.detail;
    if (typeof detail === "string") {
      message = detail;
    } else if (detail && typeof detail === "object") {
      message = detail.detail ?? message;
      hint = detail.hint ?? undefined;
    }
  } catch {
    /* non-JSON error body: keep the status line */
  }
  throw new ApiError(message, response.status, hint);
}

export const api = {
  health: () => fetch(`${BASE}/health`).then(unwrap<Health>),

  models: () => fetch(`${BASE}/models`).then(unwrap<ModelList>),

  loadModel: (name: string) =>
    fetch(`${BASE}/models/${name}/load`, { method: "POST" }).then(
      unwrap<{ model: string; loaded: boolean; load_ms: number }>,
    ),

  samples: () => fetch(`${BASE}/samples`).then(unwrap<Sample[]>),

  predict: (file: File, model: string, conf: number) => {
    const form = new FormData();
    form.append("file", file);
    form.append("model", model);
    form.append("conf", String(conf));
    return fetch(`${BASE}/predict`, { method: "POST", body: form }).then(unwrap<Prediction>);
  },

  predictSample: (sampleId: string, model: string, conf: number) =>
    fetch(`${BASE}/predict/sample/${sampleId}?model=${model}&conf=${conf}`, {
      method: "POST",
    }).then(unwrap<Prediction>),

  compare: (file: File, conf: number) => {
    const form = new FormData();
    form.append("file", file);
    form.append("conf", String(conf));
    return fetch(`${BASE}/compare`, { method: "POST", body: form }).then(unwrap<CompareResult>);
  },

  batch: (files: File[], model: string, conf: number) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    form.append("model", model);
    form.append("conf", String(conf));
    return fetch(`${BASE}/predict/batch`, { method: "POST", body: form }).then(unwrap<BatchResult>);
  },

  metrics: (name: string) => fetch(`${BASE}/metrics/${name}`).then(unwrap<Metrics>),

  metricsSummary: () =>
    fetch(`${BASE}/metrics`).then(unwrap<{ models: Record<string, Record<string, number>> }>),

  sampleFile: async (sample: Sample): Promise<File> => {
    const response = await fetch(sample.url);
    const blob = await response.blob();
    return new File([blob], sample.filename, { type: blob.type || "image/png" });
  },
};
