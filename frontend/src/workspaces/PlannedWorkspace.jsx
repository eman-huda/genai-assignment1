import WorkspaceHeader from "../components/WorkspaceHeader";
import Notice from "../components/Notice";

// Placeholder for Tasks 2 and 3: shows the planned layout so the app structure is complete.
export default function PlannedWorkspace({ title, description, task, bars, barLabel, shows }) {
  return (
    <section>
      <WorkspaceHeader title={title}>{description}</WorkspaceHeader>
      <div className="mb-5 max-w-3xl">
        <Notice tone="info">This workspace is waiting for the {task} model. It switches on once its ONNX files are added to the models folder.</Notice>
      </div>
      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]" aria-hidden="true">
        <div className="space-y-4 rounded-panel bg-sky-200/40 p-5 opacity-70 ring-1 ring-sky-300/50">
          <div className="h-4 w-24 rounded-full bg-white/80" />
          <div className="h-24 rounded-2xl bg-white/70" />
          <div className="h-4 w-28 rounded-full bg-white/80" />
          <div className="flex gap-2">{[0, 1, 2].map((i) => <div key={i} className="h-8 flex-1 rounded-full bg-white/70" />)}</div>
          <div className="h-11 rounded-full bg-lilac-300/60" />
        </div>
        <div className="space-y-5 opacity-80">
          <div className="grid max-w-3xl grid-cols-3 gap-3">
            {["Model input", "Restored", "Error map"].map((l) => (
              <div key={l}><div className="aspect-square rounded-xl bg-lilac-50 ring-1 ring-line" /><p className="mt-1.5 text-xs font-bold text-ink-faint">{l}</p></div>
            ))}
          </div>
          <div className="max-w-3xl rounded-2xl bg-white p-4 ring-1 ring-line">
            <p className="mb-3 text-sm font-bold text-ink-soft">{barLabel}</p>
            {bars.map((b, i) => (
              <div key={b} className="mb-2 grid grid-cols-[9rem_1fr] items-center gap-3 text-sm text-ink-faint">
                <span>{b}</span>
                <div className="h-2.5 rounded-full bg-sky-100"><div className="h-2.5 rounded-full bg-sky-300/70" style={{ width: `${[38, 22, 28, 12][i]}%` }} /></div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="mt-6 max-w-3xl text-sm text-ink-soft">
        <p className="font-bold text-ink">When live, this workspace will show</p>
        <ul className="mt-1 list-disc pl-5">{shows.map((s) => <li key={s}>{s}</li>)}</ul>
      </div>
    </section>
  );
}
