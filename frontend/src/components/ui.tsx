import type { ReactNode } from "react";

export function Panel({
  title,
  actions,
  children,
  className = "",
  bodyClassName = "p-4",
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || actions) && (
        <header className="panel-header">
          <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
          {actions}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

export function Chip({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "alert" | "clear" | "warn" | "accent";
  children: ReactNode;
}) {
  const tones: Record<string, string> = {
    neutral: "border-white/10 bg-white/5 text-slate-300",
    alert: "border-alert/30 bg-alert/10 text-alert",
    clear: "border-clear/30 bg-clear/10 text-clear",
    warn: "border-warn/30 bg-warn/10 text-warn",
    accent: "border-accent/30 bg-accent/10 text-accent",
  };
  return <span className={`chip whitespace-nowrap ${tones[tone]}`}>{children}</span>;
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="stat mt-1 text-base">{value}</div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent/30 border-t-accent" />
      {label}
    </div>
  );
}

export function ErrorNote({ message, hint }: { message: string; hint?: string }) {
  return (
    <div className="animate-fadeIn rounded-lg border border-alert/30 bg-alert/10 p-3 text-sm text-alert">
      <p className="font-medium">{message}</p>
      {hint && <p className="mt-1 text-xs text-alert/80">{hint}</p>}
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="grid place-items-center rounded-lg border border-dashed border-white/10 p-8 text-center">
      <div>
        <p className="text-sm font-medium text-slate-300">{title}</p>
        {children && <div className="mt-1 text-xs text-slate-500">{children}</div>}
      </div>
    </div>
  );
}
