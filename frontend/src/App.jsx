import { useEffect, useState } from "react";
import { api } from "./api";
import UniversalRestoration from "./workspaces/UniversalRestoration";
import FaceToSketch from "./workspaces/FaceToSketch";
import HardRouting from "./workspaces/HardRouting";
import SoftMoE from "./workspaces/SoftMoE";

const WORKSPACES = [
  { id: "universal", key: "task1", name: "Universal Restoration", task: "Task 1" },
  { id: "hard", key: "task2", name: "Hard-Routed Restoration", task: "Task 2" },
  { id: "soft", key: "task3", name: "Soft Mixture-of-Experts Restoration", task: "Task 3" },
  { id: "sketch", key: "task4", name: "Face-to-Sketch Generator", task: "Task 4" },
];

const STATUS_DOT = { ready: "bg-sky-500", planned: "bg-transparent ring-1 ring-lilac-500", missing: "bg-rose-400" };

function useHashRoute() {
  const read = () => (WORKSPACES.some((w) => `#${w.id}` === window.location.hash) ? window.location.hash.slice(1) : "universal");
  const [route, setRoute] = useState(read);
  useEffect(() => { const f = () => setRoute(read()); window.addEventListener("hashchange", f); return () => window.removeEventListener("hashchange", f); }, []);
  return route;
}

export default function App() {
  const route = useHashRoute();
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState({});
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const poll = () => api.health().then((h) => { setHealth(h); setOffline(false); }).catch(() => setOffline(true));
    poll();
    api.models().then((list) => setModels(Object.fromEntries(list.map((m) => [m.key, m])))).catch(() => {});
    const t = setInterval(poll, 15000);
    return () => clearInterval(t);
  }, []);

  const status = (key) => health?.models?.[key] ?? "missing";

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[17rem_1fr]">
      <aside className="border-b border-line bg-white/80 px-5 py-5 lg:sticky lg:top-0 lg:h-screen lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-3">
          <img src="/favicon.svg" alt="" className="h-9 w-9" />
          <div>
            <p className="text-lg font-extrabold leading-tight">Restore &amp; Sketch</p>
            <p className="text-xs text-ink-faint">Generative AI, Assignment 1</p>
          </div>
        </div>

        <nav aria-label="Workspaces" className="mt-6 flex gap-1 overflow-x-auto lg:block lg:space-y-1">
          {WORKSPACES.map((w) => {
            const active = route === w.id;
            const s = status(w.key);
            return (
              <a key={w.id} href={`#${w.id}`} aria-current={active ? "page" : undefined}
                className={`flex min-w-max items-center gap-3 rounded-2xl px-3 py-2.5 transition-colors lg:min-w-0 ${
                  active ? "bg-lilac-100 text-lilac-700" : "text-ink-soft hover:bg-sky-50"}`}>
                <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${STATUS_DOT[s] ?? STATUS_DOT.missing}`} title={s} />
                <span className="leading-tight">
                  <span className="block text-sm font-bold">{w.name}</span>
                  <span className="block text-xs text-ink-faint">{w.task}{s === "planned" ? ", coming soon" : ""}</span>
                </span>
              </a>
            );
          })}
        </nav>

        <div className="mt-6 hidden rounded-2xl bg-sky-50 p-4 text-xs text-ink-soft ring-1 ring-sky-200 lg:block">
          <p className="mb-1 font-bold text-ink">Server</p>
          {offline ? (
            <p className="text-rose-700">The API is not responding. Check that the backend container is running.</p>
          ) : health ? (
            <>
              <p>Online for {Math.floor(health.uptime_s / 60)} min</p>
              <p>ONNX Runtime {health.onnxruntime}, CPU</p>
              <p>{Object.values(health.models).filter((v) => v === "ready").length} of 4 models loaded</p>
            </>
          ) : <p>Connecting…</p>}
        </div>
      </aside>

      <main className="px-5 py-8 sm:px-8 lg:px-12">
        {route === "universal" && <UniversalRestoration model={models.task1} />}
        {route === "sketch" && <FaceToSketch model={models.task4} />}
        {route === "hard" && <HardRouting model={models.task2} />}
        {route === "soft" && <SoftMoE model={models.task3} />}
      </main>
    </div>
  );
}
