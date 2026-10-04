import { useEffect, useState } from "react";
import { api } from "../api";
import Segmented from "./Segmented";
import UploadZone from "./UploadZone";
import Notice from "./Notice";

// Exact test levels from the corruption manifest (Tasks 1 to 3), shown as help text.
export const LEVELS = {
  salt_pepper: { low: "p = 0.03", medium: "p = 0.08", high: "p = 0.15", random: "p from 0.02 to 0.15" },
  gaussian_blur: { low: "kernel 3, σ 0.7", medium: "kernel 5, σ 1.5", high: "kernel 7, σ 2.5", random: "kernel 3, 5 or 7, σ from 0.5 to 2.5" },
  occlusion: { low: "1 box, 10% of the image", medium: "2 boxes, 20%", high: "3 boxes, 35%", random: "1 to 3 boxes, 10% to 35%" },
};

// Image source, corruption, severity and seed: shared by the restoration workspaces.
// Calls onChange with { ready, form } where form is a function that builds the FormData.
export function useRestorationInputs() {
  const [source, setSource] = useState("sample");
  const [samples, setSamples] = useState([]);
  const [sample, setSample] = useState(null);
  const [file, setFile] = useState(null);
  const [corruption, setCorruption] = useState("salt_pepper");
  const [severity, setSeverity] = useState("medium");
  const [seed, setSeed] = useState("");

  useEffect(() => {
    api.samples().then((r) => { setSamples(r.samples); setSample(r.samples[0] ?? null); }).catch(() => setSamples([]));
  }, []);
  useEffect(() => {
    if (source === "sample" && corruption === "none") setCorruption("salt_pepper");
  }, [source, corruption]);

  const state = { source, setSource, samples, sample, setSample, file, setFile, corruption, setCorruption,
                  severity, setSeverity, seed, setSeed };
  const hasImage = source === "sample" ? !!sample : !!file;
  const buildForm = () => {
    const form = new FormData();
    if (source === "sample") form.append("sample", sample); else form.append("file", file);
    form.append("corruption", corruption);
    form.append("severity", severity);
    if (seed !== "") form.append("seed", seed);
    return form;
  };
  return { state, hasImage, buildForm };
}

export default function RestorationInputs({ state }) {
  const { source, setSource, samples, sample, setSample, file, setFile, corruption, setCorruption,
          severity, setSeverity, seed, setSeed } = state;
  const corruptions = [
    { value: "salt_pepper", label: "Salt and pepper" },
    { value: "gaussian_blur", label: "Gaussian blur" },
    { value: "occlusion", label: "Occlusion" },
    { value: "clean", label: "No corruption" },
    ...(source === "upload" ? [{ value: "none", label: "Already corrupted" }] : []),
  ];
  const hasSeverity = !["clean", "none"].includes(corruption);

  return (
    <>
      <div>
        <span className="field-label">Image</span>
        <Segmented label="Image source" value={source} onChange={setSource}
          options={[{ value: "sample", label: "Sample" }, { value: "upload", label: "Upload" }]} />
        <div className="mt-3">
          {source === "sample" ? (
            samples.length ? (
              <div role="radiogroup" aria-label="Sample images" className="grid grid-cols-4 gap-2">
                {samples.map((s) => (
                  <button key={s} type="button" role="radio" aria-checked={sample === s} onClick={() => setSample(s)}
                    title={s} className={`overflow-hidden rounded-xl ring-2 transition ${sample === s ? "ring-lilac-600" : "ring-transparent hover:ring-lilac-200"}`}>
                    <img src={api.sampleUrl(s)} alt={s} className="aspect-square w-full object-cover" />
                  </button>
                ))}
              </div>
            ) : (
              <Notice tone="info">No sample images found. Add clean images to models/samples, or upload one.</Notice>
            )
          ) : (
            <UploadZone file={file} onFile={setFile} hint="PNG, JPEG, WEBP or BMP, up to 10 MB. Resized to 128 × 128." />
          )}
        </div>
      </div>

      <div>
        <span className="field-label">Corruption</span>
        <div className="flex flex-wrap gap-2">
          {corruptions.map((c) => (
            <button key={c.value} type="button" aria-pressed={corruption === c.value} onClick={() => setCorruption(c.value)}
              className={`rounded-full px-3.5 py-1.5 text-sm font-bold ring-1 transition-colors ${
                corruption === c.value ? "bg-lilac-600 text-white ring-lilac-600" : "bg-white text-ink-soft ring-sky-300 hover:bg-lilac-50"}`}>
              {c.label}
            </button>
          ))}
        </div>
        {corruption === "none" && <p className="mt-2 text-xs text-ink-soft">Your upload goes to the model unchanged, so no quality scores can be computed.</p>}
      </div>

      {hasSeverity && (
        <div>
          <span className="field-label">Severity</span>
          <Segmented label="Severity" size="sm" value={severity} onChange={setSeverity}
            options={[{ value: "low", label: "Low" }, { value: "medium", label: "Medium" }, { value: "high", label: "High" }, { value: "random", label: "Random" }]} />
          <p className="mt-2 text-xs text-ink-soft">{LEVELS[corruption][severity]}</p>
        </div>
      )}

      {hasSeverity && (
        <div>
          <label htmlFor="seed" className="field-label">Random seed <span className="font-semibold text-ink-faint">(optional)</span></label>
          <input id="seed" inputMode="numeric" value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))}
            placeholder="Leave empty for a new pattern each run"
            className="w-full rounded-xl border border-sky-300 bg-white px-3 py-2 text-sm placeholder:text-ink-faint" />
        </div>
      )}
    </>
  );
}
