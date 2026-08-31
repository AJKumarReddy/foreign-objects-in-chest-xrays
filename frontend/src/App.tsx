import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { Chip, ErrorNote, Spinner } from "@/components/ui";
import { useBackend } from "@/hooks/useModels";
import { BatchPage } from "@/pages/Batch";
import { ComparePage } from "@/pages/Compare";
import { DetectPage } from "@/pages/Detect";
import { MetricsPage } from "@/pages/Metrics";

const TABS = [
  { to: "/detect", label: "Detect" },
  { to: "/compare", label: "Compare" },
  { to: "/batch", label: "Batch" },
  { to: "/metrics", label: "Metrics" },
];

export default function App() {
  const { models, health, samples, loading, error, refresh } = useBackend();

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-white/5 bg-ink-950/85 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-4 px-5 py-3">
          <div className="flex items-center gap-2.5">
            <Logo />
            <div>
              <p className="text-sm font-semibold leading-tight text-slate-100">
                Foreign Object Detection
              </p>
              <p className="text-[11px] leading-tight text-slate-500">
                Chest radiographs · object-CXR
              </p>
            </div>
          </div>

          <nav className="flex gap-1" aria-label="Sections">
            {TABS.map((tab) => (
              <NavLink
                key={tab.to}
                to={tab.to}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-1.5 text-sm transition ${
                    isActive
                      ? "bg-white/10 text-slate-100"
                      : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                  }`
                }
              >
                {tab.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            {health && (
              <>
                <Chip tone={health.models_ready.length ? "clear" : "warn"}>
                  {health.models_ready.length
                    ? `${health.models_ready.length} model ready`
                    : "demo mode"}
                </Chip>
                <Chip>{health.device}</Chip>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-5 py-5">
        {loading && <Spinner label="Connecting to the API…" />}

        {error && (
          <div className="space-y-3">
            <ErrorNote message={error} hint="Start it with: cxr serve --port 8000" />
            <button type="button" className="btn-ghost" onClick={() => void refresh()}>
              Retry
            </button>
          </div>
        )}

        {models && !error && (
          <Routes>
            <Route path="/detect" element={<DetectPage backend={models} samples={samples} />} />
            <Route path="/compare" element={<ComparePage backend={models} samples={samples} />} />
            <Route path="/batch" element={<BatchPage backend={models} />} />
            <Route path="/metrics" element={<MetricsPage backend={models} />} />
            <Route path="*" element={<Navigate to="/detect" replace />} />
          </Routes>
        )}
      </main>

      <footer className="mx-auto max-w-[1400px] px-5 pb-8 pt-2">
        <p className="text-[11px] text-slate-600">
          Research and educational use only — not a medical device. Predictions must not be used
          for diagnosis.
          {health && ` · v${health.version} · torch ${health.torch_version}`}
        </p>
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="8" fill="#0f172a" stroke="#1e293b" />
      <path
        d="M11 7c0 4-3 5-3 9s2 8 2 8M21 7c0 4 3 5 3 9s-2 8-2 8"
        stroke="#38bdf8"
        strokeWidth="1.6"
        fill="none"
        strokeLinecap="round"
        opacity="0.8"
      />
      <circle cx="16" cy="17" r="3.4" fill="none" stroke="#fb7185" strokeWidth="1.8" />
      <circle cx="16" cy="17" r="1" fill="#fb7185" />
    </svg>
  );
}
