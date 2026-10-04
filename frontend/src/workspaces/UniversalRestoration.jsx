import { useState } from "react";
import { api } from "../api";
import RestorationInputs, { useRestorationInputs } from "../components/RestorationInputs";
import RestorationResult from "../components/RestorationResult";
import Readout from "../components/Readout";
import ModelDetails from "../components/ModelDetails";
import Notice from "../components/Notice";
import WorkspaceHeader from "../components/WorkspaceHeader";

export default function UniversalRestoration({ model }) {
  const inputs = useRestorationInputs();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const ready = model?.status === "ready";

  async function run() {
    setBusy(true); setError("");
    try { setResult(await api.restore(inputs.buildForm())); } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  const m = result?.metrics;
  return (
    <section>
      <WorkspaceHeader title="Universal Restoration">
        One autoencoder restores every corruption type without being told which one was applied.
        Pick an image, choose a corruption, and compare what goes in with what comes out.
      </WorkspaceHeader>
      {!ready && model && (
        <div className="mb-5"><Notice>The Task 1 model is {model.status}. Add task1_universal_dae.onnx to the models folder and restart the app.</Notice></div>
      )}
      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <div className="space-y-5 rounded-panel bg-sky-200/60 p-5 ring-1 ring-sky-300/50">
          <RestorationInputs state={inputs.state} />
          <button type="button" className="btn-primary w-full" disabled={!ready || busy || !inputs.hasImage} onClick={run}>
            {busy ? "Restoring…" : "Restore image"}
          </button>
          {error && <Notice>{error}</Notice>}
          <ModelDetails model={model} />
        </div>
        <div className="min-w-0">
          {!result ? (
            <div className="grid min-h-[24rem] place-items-center rounded-panel border-2 border-dashed border-lilac-200 bg-lilac-50/50 p-8 text-center">
              <p className="max-w-xs text-ink-soft">Choose an image and a corruption, then select <b className="text-ink">Restore image</b>. The input and the restored result appear here.</p>
            </div>
          ) : (
            <RestorationResult result={result} readouts={<>
              <Readout value={`${result.timing_ms.inference.toFixed(1)} ms`} label="Model inference" tone="lilac" />
              <Readout value={`${result.timing_ms.total.toFixed(0)} ms`} label="Total on server" tone="lilac" />
              {m && <Readout value={`${m.input.psnr.toFixed(1)} → ${m.restored.psnr.toFixed(1)}`} label="PSNR in dB, input → restored" />}
              {m && <Readout value={`${m.input.ssim.toFixed(3)} → ${m.restored.ssim.toFixed(3)}`} label="SSIM, input → restored" />}
            </>} />
          )}
        </div>
      </div>
    </section>
  );
}
