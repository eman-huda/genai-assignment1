import { useState } from "react";
import { api } from "../api";
import Segmented from "../components/Segmented";
import RestorationInputs, { useRestorationInputs } from "../components/RestorationInputs";
import RestorationResult from "../components/RestorationResult";
import Readout from "../components/Readout";
import ModelDetails from "../components/ModelDetails";
import Notice from "../components/Notice";
import WorkspaceHeader from "../components/WorkspaceHeader";

const CLASS_NAMES = { clean: "Clean", salt_pepper: "Salt and pepper", gaussian_blur: "Gaussian blur", occlusion: "Occlusion" };
const BRANCH_FOR = { clean: "Identity bypass", salt_pepper: "Salt-and-pepper specialist", gaussian_blur: "Blur specialist", occlusion: "Occlusion specialist" };

// The four classifier probabilities, with the predicted class and the branch that actually ran marked.
function ProbabilityPanel({ classifier, routing }) {
  const pct = (p) => `${(p * 100).toFixed(1)}%`;
  return (
    <div className="max-w-3xl rounded-2xl bg-white p-4 ring-1 ring-line">
      <h2 className="mb-3 text-sm font-bold">Classifier probabilities</h2>
      <ul className="space-y-2.5">
        {Object.entries(classifier.probs).map(([cls, p]) => {
          const predicted = cls === classifier.predicted;
          const routed = cls === routing.routed_class;
          return (
            <li key={cls} className={`rounded-xl px-3 py-2 ${routed ? "bg-lilac-50 ring-1 ring-lilac-200" : ""}`}>
              <div className="mb-1 flex flex-wrap items-center gap-2 text-sm">
                <span className={routed ? "font-bold text-ink" : "text-ink-soft"}>{CLASS_NAMES[cls]}</span>
                {predicted && <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-bold text-ink">Predicted</span>}
                {routed && <span className="rounded-full bg-lilac-600 px-2 py-0.5 text-xs font-bold text-white">Sent to {BRANCH_FOR[cls].toLowerCase()}</span>}
                {routing.true_class === cls && <span className="rounded-full bg-white px-2 py-0.5 text-xs font-bold text-ink-soft ring-1 ring-line">True corruption</span>}
                <span className="ml-auto font-bold tabular-nums">{pct(p)}</span>
              </div>
              <div className="h-2.5 rounded-full bg-sky-100" role="presentation">
                <div className={`h-2.5 rounded-full ${routed ? "bg-lilac-600" : "bg-sky-300"}`} style={{ width: pct(p) }} />
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-3 rounded-xl bg-sky-50 px-3 py-2.5 text-sm leading-relaxed text-ink-soft">
        {routing.mode === "predicted"
          ? <>The highest probability ({CLASS_NAMES[classifier.predicted]}, {pct(classifier.confidence)}) decides the branch. </>
          : <>Oracle mode: the true corruption ({CLASS_NAMES[routing.true_class]}) decides the branch, whatever the classifier says. </>}
        {routing.routed_class === "clean"
          ? "The image was returned unchanged; no expert was run."
          : `Only the ${BRANCH_FOR[routing.routed_class].toLowerCase()} was run; the other experts stayed idle.`}
        {routing.classifier_correct === false && <b className="text-rose-700"> The classifier was wrong for this image.</b>}
        {routing.classifier_correct === true && " The classifier's prediction was correct."}
      </p>
    </div>
  );
}

export default function HardRouting({ model }) {
  const inputs = useRestorationInputs();
  const [routing, setRouting] = useState("predicted");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const ready = model?.status === "ready";
  const oracleAllowed = inputs.state.corruption !== "none";

  async function run() {
    setBusy(true); setError("");
    const form = inputs.buildForm();
    form.append("routing", oracleAllowed ? routing : "predicted");
    try { setResult(await api.hardRoute(form)); } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  const m = result?.metrics;
  return (
    <section>
      <WorkspaceHeader title="Hard-Routed Restoration">
        A classifier first decides which corruption an image has, then sends it to one specialist autoencoder.
        Clean images skip restoration entirely.
      </WorkspaceHeader>
      {!ready && model && (
        <div className="mb-5"><Notice>The Task 2 models are {model.status}. Add the classifier and the three specialist ONNX files to the models folder and restart the app.</Notice></div>
      )}
      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <div className="space-y-5 rounded-panel bg-sky-200/60 p-5 ring-1 ring-sky-300/50">
          <RestorationInputs state={inputs.state} />
          <div>
            <span className="field-label">Routing</span>
            <Segmented label="Routing mode" size="sm" value={oracleAllowed ? routing : "predicted"} onChange={setRouting}
              options={[{ value: "predicted", label: "Classifier" }, { value: "oracle", label: "True label (oracle)", disabled: !oracleAllowed }]} />
            <p className="mt-2 text-xs text-ink-soft">
              {oracleAllowed ? "Oracle routing uses the corruption applied here, to show what the specialists can do with perfect routing."
                             : "Oracle routing needs a known corruption, so it is off for already corrupted uploads."}
            </p>
          </div>
          <button type="button" className="btn-primary w-full" disabled={!ready || busy || !inputs.hasImage} onClick={run}>
            {busy ? "Routing…" : "Restore image"}
          </button>
          {error && <Notice>{error}</Notice>}
          <ModelDetails model={model} />
        </div>
        <div className="min-w-0">
          {!result ? (
            <div className="grid min-h-[24rem] place-items-center rounded-panel border-2 border-dashed border-lilac-200 bg-lilac-50/50 p-8 text-center">
              <p className="max-w-xs text-ink-soft">Choose an image and a corruption, then select <b className="text-ink">Restore image</b>. The classifier's decision and the restored result appear here.</p>
            </div>
          ) : (
            <RestorationResult result={result} readouts={<>
              <Readout value={`${result.timing_ms.classifier.toFixed(1)} ms`} label="Classifier" tone="lilac" />
              <Readout value={result.routing.routed_class === "clean" ? "skipped" : `${result.timing_ms.expert.toFixed(1)} ms`} label="Specialist" tone="lilac" />
              <Readout value={`${result.timing_ms.total.toFixed(0)} ms`} label="Total on server" tone="lilac" />
              {m ? <Readout value={`${m.input.ssim.toFixed(3)} → ${m.restored.ssim.toFixed(3)}`} label="SSIM, input → restored" />
                 : <Readout value={BRANCH_FOR[result.routing.routed_class]} label="Branch used" />}
            </>}>
              <ProbabilityPanel classifier={result.classifier} routing={result.routing} />
            </RestorationResult>
          )}
        </div>
      </div>
    </section>
  );
}
