import { useMemo, useState } from "react";
import { ApiError, api } from "@/api/client";
import type { BatchResult, ModelList } from "@/api/types";
import { ConfidenceSlider, UploadZone } from "@/components/Controls";
import { Chip, EmptyState, ErrorNote, Panel, Spinner, Stat } from "@/components/ui";
import { formatPercent } from "@/lib/scale";

type SortKey = "filename" | "score" | "count";

export function BatchPage({ backend }: { backend: ModelList }) {
  const [model, setModel] = useState(
    () => backend.models.find((m) => m.ready)?.name ?? backend.models[0]?.name ?? "frcnn",
  );
  const [conf, setConf] = useState(0.5);
  const [result, setResult] = useState<BatchResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<{ message: string; hint?: string } | null>(null);
  const [sort, setSort] = useState<SortKey>("score");

  const run = async (files: File[]) => {
    setBusy(true);
    setError(null);
    try {
      setResult(await api.batch(files, model, conf));
    } catch (err) {
      setResult(null);
      setError(
        err instanceof ApiError ? { message: err.message, hint: err.hint } : { message: String(err) },
      );
    } finally {
      setBusy(false);
    }
  };

  const rows = useMemo(() => {
    if (!result) return [];
    const items = [...result.items];
    items.sort((a, b) => {
      if (sort === "filename") return a.filename.localeCompare(b.filename);
      if (sort === "count")
        return (b.prediction?.detection_count ?? -1) - (a.prediction?.detection_count ?? -1);
      return (b.prediction?.image_score ?? -1) - (a.prediction?.image_score ?? -1);
    });
    return items;
  }, [result, sort]);

  const exportCsv = () => {
    if (!result) return;
    const header = "image_name,prediction,verdict,detections,model\n";
    const body = result.items
      .map((item) =>
        item.ok && item.prediction
          ? [
              item.filename,
              item.prediction.image_score.toFixed(6),
              item.prediction.has_foreign_object ? "foreign_object" : "clear",
              item.prediction.detection_count,
              item.prediction.model,
            ].join(",")
          : `${item.filename},,error,0,`,
      )
      .join("\n");
    const blob = new Blob([header + body], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "cxr_batch_results.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <div className="space-y-4">
      <Panel title="Batch screening">
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_240px_220px]">
          <UploadZone
            multiple
            onFiles={run}
            hint="Up to 32 images per run · results exportable as CSV"
          />
          <ConfidenceSlider value={conf} onChange={setConf} />
          <div className="space-y-2">
            <label className="label" htmlFor="batch-model">
              Detector
            </label>
            <select
              id="batch-model"
              className="w-full rounded-lg border border-white/10 bg-ink-850 px-3 py-2 text-sm"
              value={model}
              onChange={(event) => setModel(event.target.value)}
            >
              {backend.models.map((m) => (
                <option key={m.name} value={m.name} disabled={!m.ready}>
                  {m.display_name}
                  {m.ready ? "" : " (no weights)"}
                </option>
              ))}
            </select>
            <button type="button" className="btn-ghost w-full" onClick={exportCsv} disabled={!result}>
              Export CSV
            </button>
          </div>
        </div>
      </Panel>

      {busy && <Spinner label="Scoring images…" />}
      {error && <ErrorNote message={error.message} hint={error.hint} />}

      {!result && !busy && !error && (
        <EmptyState title="Drop a folder of X-rays to screen them in one pass" />
      )}

      {result && (
        <>
          <Panel title="Summary">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
              <Stat label="Images" value={result.total} />
              <Stat label="Flagged" value={result.positive} />
              <Stat label="Clear" value={result.negative} />
              <Stat label="Failed" value={result.failed} />
              <Stat label="Elapsed" value={`${(result.total_ms / 1000).toFixed(1)} s`} />
            </div>
          </Panel>

          <Panel
            title="Results"
            bodyClassName="p-0"
            actions={
              <select
                aria-label="Sort results"
                className="rounded-md border border-white/10 bg-ink-850 px-2 py-1 text-xs"
                value={sort}
                onChange={(event) => setSort(event.target.value as SortKey)}
              >
                <option value="score">Sort by score</option>
                <option value="count">Sort by object count</option>
                <option value="filename">Sort by name</option>
              </select>
            }
          >
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 text-left text-[11px] uppercase tracking-wider text-slate-500">
                  <th className="px-4 py-2 font-medium">Image</th>
                  <th className="px-4 py-2 font-medium">Verdict</th>
                  <th className="px-4 py-2 text-right font-medium">Score</th>
                  <th className="px-4 py-2 text-right font-medium">Objects</th>
                  <th className="px-4 py-2 text-right font-medium">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {rows.map((item) => (
                  <tr key={item.filename} className="hover:bg-white/[0.03]">
                    <td className="max-w-[22ch] truncate px-4 py-2 font-mono text-xs" title={item.filename}>
                      {item.filename}
                    </td>
                    <td className="px-4 py-2">
                      {!item.ok ? (
                        <Chip tone="warn">error</Chip>
                      ) : item.prediction?.has_foreign_object ? (
                        <Chip tone="alert">foreign object</Chip>
                      ) : (
                        <Chip tone="clear">clear</Chip>
                      )}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">
                      {item.prediction ? formatPercent(item.prediction.image_score) : "—"}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs">
                      {item.prediction?.detection_count ?? "—"}
                    </td>
                    <td className="px-4 py-2 text-right font-mono text-xs text-slate-500">
                      {item.prediction ? `${item.prediction.inference_ms.toFixed(0)}ms` : item.error}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        </>
      )}
    </div>
  );
}
