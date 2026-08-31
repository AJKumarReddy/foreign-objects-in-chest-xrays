import { useEffect, useState } from "react";
import { ApiError, api } from "@/api/client";
import type { Metrics, ModelList } from "@/api/types";
import { FrocChart, RocChart, SweepChart } from "@/components/Charts";
import { Chip, EmptyState, Panel, Spinner, Stat } from "@/components/ui";
import { formatPercent } from "@/lib/scale";

export function MetricsPage({ backend }: { backend: ModelList }) {
  const [model, setModel] = useState(backend.models[0]?.name ?? "frcnn");
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [missing, setMissing] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .metrics(model)
      .then((data) => {
        if (!cancelled) {
          setMetrics(data);
          setMissing(null);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setMetrics(null);
        setMissing(err instanceof ApiError ? err.message : String(err));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [model]);

  const info = backend.models.find((m) => m.name === model);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {backend.models.map((m) => (
          <button
            key={m.name}
            type="button"
            onClick={() => setModel(m.name)}
            className={m.name === model ? "btn-primary" : "btn-ghost"}
          >
            {m.display_name}
          </button>
        ))}
      </div>

      {loading && <Spinner label="Loading evaluation artefacts…" />}

      {!loading && missing && (
        <EmptyState title="No evaluation for this model yet">
          <p className="mb-2">{missing}</p>
          <code className="rounded bg-black/40 px-2 py-1 font-mono text-[11px] text-slate-400">
            cxr evaluate --model {model} --split dev
          </code>
        </EmptyState>
      )}

      {!loading && metrics && (
        <>
          <Panel title="Headline">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-6">
              <Stat label="AUC" value={metrics.auc.toFixed(3)} hint="image level" />
              <Stat label="Accuracy" value={formatPercent(metrics.accuracy)} hint={`@ ${metrics.threshold.toFixed(2)}`} />
              <Stat label="Sensitivity" value={formatPercent(metrics.confusion.sensitivity)} />
              <Stat label="Specificity" value={formatPercent(metrics.confusion.specificity)} />
              <Stat label="Images" value={metrics.n_images} hint={`${metrics.n_positive} positive`} />
              <Stat
                label="FROC mean"
                value={metrics.localization.mean_sensitivity.toFixed(3)}
                hint={`${metrics.localization.total_objects} objects`}
              />
            </div>
          </Panel>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel title="ROC curve — image level classification">
              <RocChart
                points={metrics.roc.points}
                auc={metrics.roc.auc}
                operating={{ fpr: metrics.roc.best_fpr, tpr: metrics.roc.best_tpr }}
              />
              <p className="mt-2 text-[11px] text-slate-500">
                Youden-optimal threshold {metrics.roc.best_threshold.toFixed(3)}; the dashed
                diagonal is chance.
              </p>
            </Panel>

            <Panel title="Operating point sweep">
              <SweepChart sweep={metrics.sweep} threshold={metrics.threshold} />
              <p className="mt-2 text-[11px] text-slate-500">
                Dashed line marks the deployed threshold ({metrics.threshold.toFixed(2)}).
              </p>
            </Panel>

            <Panel title="Confusion matrix">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[11px] uppercase tracking-wider text-slate-500">
                    <th className="py-1 font-medium" />
                    <th className="py-1 font-medium">Predicted positive</th>
                    <th className="py-1 font-medium">Predicted negative</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  <tr className="border-t border-white/5">
                    <td className="py-2 text-[11px] uppercase text-slate-500">Actual positive</td>
                    <td className="py-2 text-clear">{metrics.confusion.tp}</td>
                    <td className="py-2 text-alert">{metrics.confusion.fn}</td>
                  </tr>
                  <tr className="border-t border-white/5">
                    <td className="py-2 text-[11px] uppercase text-slate-500">Actual negative</td>
                    <td className="py-2 text-alert">{metrics.confusion.fp}</td>
                    <td className="py-2 text-clear">{metrics.confusion.tn}</td>
                  </tr>
                </tbody>
              </table>
              <div className="mt-3 flex gap-4 text-[11px] text-slate-500">
                <span>precision {formatPercent(metrics.confusion.precision)}</span>
                <span>F1 {metrics.confusion.f1.toFixed(3)}</span>
              </div>
            </Panel>

            <Panel title="FROC — localization sensitivity">
              {metrics.localization.points.length ? (
                <FrocChart points={metrics.localization.points} />
              ) : (
                <EmptyState title="No localization data" />
              )}
              <p className="mt-2 text-[11px] text-slate-500">
                A detection counts as a hit when its centre falls inside an annotated object.
              </p>
            </Panel>
          </div>

          <Panel title="Model card">
            <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
              <Field label="Model" value={info?.display_name ?? metrics.model} />
              <Field label="Input size" value={`${info?.input_size ?? "?"} px`} />
              <Field label="Split" value={metrics.split} />
              <Field label="Evaluated" value={new Date(metrics.evaluated_at).toLocaleString()} />
              <Field label="Device" value={metrics.device} />
              <Field label="Weights" value={metrics.weights} mono />
            </dl>
            <div className="mt-4 space-y-2 rounded-lg border border-warn/25 bg-warn/[0.05] p-3">
              <p className="flex items-center gap-2 text-xs font-semibold text-warn">
                <Chip tone="warn">limitations</Chip>
              </p>
              <ul className="list-inside list-disc space-y-1 text-[11px] leading-relaxed text-slate-400">
                <li>
                  Trained on the object-CXR training split with un-annotated images excluded, so the
                  detector never saw clean radiographs during training.
                </li>
                <li>
                  Image level scores are the maximum box confidence — a single spurious box drives
                  the whole-image verdict.
                </li>
                <li>Frontal radiographs only; performance on other views or modalities is unknown.</li>
                <li>Research and educational use only. Not a medical device, not for diagnosis.</li>
              </ul>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="label">{label}</dt>
      <dd className={`mt-0.5 break-all text-slate-300 ${mono ? "font-mono text-[11px]" : "text-sm"}`}>
        {value}
      </dd>
    </div>
  );
}
