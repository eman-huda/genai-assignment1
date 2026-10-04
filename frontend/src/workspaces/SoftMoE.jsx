import { useState } from "react";
import { api } from "../api";
import RestorationInputs, { useRestorationInputs } from "../components/RestorationInputs";
import RestorationResult from "../components/RestorationResult";
import Readout from "../components/Readout";
import ModelDetails from "../components/ModelDetails";
import Notice from "../components/Notice";
import WorkspaceHeader from "../components/WorkspaceHeader";

const BRANCHES = { clean: "Identity (keep input)", salt_pepper: "Salt-and-pepper expert", gaussian_blur: "Blur expert", occlusion: "Occlusion expert" };
const CLASS_NAMES = { clean: "clean", salt_pepper: "salt and pepper", gaussian_blur: "Gaussian blur", occlusion: "occlusion" };
const pct = (p) => `${(p * 100).toFixed(1)}%`;

// The four routing weights. Every branch with at least 10% weight is marked as a contributor.
function WeightPanel({ routing }) {
  const { weights, contributors, top_branch, true_class } = routing;
  const mixed = contributors.length > 1;
  return (
    <div className="max-w-3xl rounded-2xl bg-white p-4 ring-1 ring-line">
      <h2 className="mb-3 text-sm font-bold">Routing weights</h2>
      <ul className="space-y-2.5">
        {Object.entries(weights).map(([b, w]) => {
          const strong = contributors.includes(b);
          return (
            <li key={b} className={`rounded-xl px-3 py-2 ${b === top_branch ? "bg-lilac-50 ring-1 ring-lilac-200" : ""}`}>
              <div className="mb-1 flex flex-wrap items-center gap-2 text-sm">
                <span className={strong ? "font-bold text-ink" : "text-ink-soft"}>{BRANCHES[b]}</span>
                {b === top_branch && <span className="rounded-full bg-lilac-600 px-2 py-0.5 text-xs font-bold text-white">Strongest</span>}
                {strong && b !== top_branch && <span className="rounded-full bg-lilac-100 px-2 py-0.5 text-xs font-bold text-lilac-700">Contributes</span>}
                {true_class === b && <span className="rounded-full bg-white px-2 py-0.5 text-xs font-bold text-ink-soft ring-1 ring-line">True corruption</span>}
                <span className="ml-auto font-bold tabular-nums">{pct(w)}</span>
              </div>
              <div className="h-2.5 rounded-full bg-sky-100" role="presentation">
                <div className={`h-2.5 rounded-full ${b === top_branch ? "bg-lilac-600" : strong ? "bg-lilac-300" : "bg-sky-300"}`} style={{ width: pct(w) }} />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 rounded-xl bg-sky-50 px-3 py-2.5 text-sm leading-relaxed text-ink-soft">
        The restored image is the weighted sum of all four branches.{" "}
        {mixed
          ? <>Here the result is a blend of {(() => { const parts = contributors.map((c) => `${BRANCHES[c].toLowerCase()} (${pct(weights[c])})`); return parts.length > 1 ? `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}` : parts[0]; })()}.</>
          : <>Here one branch dominates: {BRANCHES[top_branch].toLowerCase()} ({pct(weights[top_branch])}).</>}
        {weights.clean >= 0.5 && true_class && true_class !== "clean" &&
          " The gate keeps most of the input unchanged, because for light damage the input is closer to the clean image than any expert's reconstruction."}
      </p>
    </div>
  );
}

export default function SoftMoE({ model }) {
  const inputs = useRestorationInputs();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const ready = model?.status === "ready";

  async function run() {
    setBusy(true); setError("");
    try { setResult(await api.softMoe(inputs.buildForm())); } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  const m = result?.metrics;
  return (
    <section>
      <WorkspaceHeader title="Soft Mixture-of-Experts Restoration">
        A gating network gives every branch a continuous weight, so the result can blend the unchanged input
        with one or more experts instead of choosing just one.
      </WorkspaceHeader>
      {!ready && model && (
        <div className="mb-5"><Notice>The Task 3 model is {model.status}. Add soft_moe.onnx and task3_metadata.json to the models folder and restart the app.</Notice></div>
      )}
      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <div className="space-y-5 rounded-panel bg-lilac-100/70 p-5 ring-1 ring-lilac-200/70">
          <RestorationInputs state={inputs.state} />
          <button type="button" className="btn-primary w-full" disabled={!ready || busy || !inputs.hasImage} onClick={run}>
            {busy ? "Blending…" : "Restore image"}
          </button>
          {error && <Notice>{error}</Notice>}
          <ModelDetails model={model} />
        </div>
        <div className="min-w-0">
          {!result ? (
            <div className="grid min-h-[24rem] place-items-center rounded-panel border-2 border-dashed border-sky-300 bg-sky-50/70 p-8 text-center">
              <p className="max-w-xs text-ink-soft">Choose an image and a corruption, then select <b className="text-ink">Restore image</b>. The four routing weights and the blended result appear here.</p>
            </div>
          ) : (
            <RestorationResult result={result} readouts={<>
              <Readout value={`${result.timing_ms.inference.toFixed(1)} ms`} label="Model inference (gate and all experts)" tone="lilac" />
              <Readout value={`${result.timing_ms.total.toFixed(0)} ms`} label="Total on server" tone="lilac" />
              <Readout value={result.routing.entropy.toFixed(2)} label="Routing entropy (0 = one branch, 1 = even blend)" />
              {m ? <Readout value={`${m.input.ssim.toFixed(3)} → ${m.restored.ssim.toFixed(3)}`} label="SSIM, input → restored" />
                 : <Readout value={CLASS_NAMES[result.routing.gate_class]} label="Gate's own class" />}
            </>}>
              <WeightPanel routing={result.routing} />
            </RestorationResult>
          )}
        </div>
      </div>
    </section>
  );
}
