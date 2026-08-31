import { useCallback, useEffect, useState } from "react";
import { api } from "@/api/client";
import type { Health, ModelList, Sample } from "@/api/types";

/** Loads the backend's capability report once and exposes a refresh. */
export function useBackend() {
  const [models, setModels] = useState<ModelList | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [modelList, healthInfo, sampleList] = await Promise.all([
        api.models(),
        api.health(),
        api.samples(),
      ]);
      setModels(modelList);
      setHealth(healthInfo);
      setSamples(sampleList);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error
          ? `${err.message} — is the API running on port 8000?`
          : "cannot reach the API",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { models, health, samples, loading, error, refresh };
}

/** Object URL for a File that is revoked when it changes or unmounts. */
export function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url;
}
