import { useState } from "react";
import { download } from "../api";
import Segmented from "./Segmented";
import CompareSlider from "./CompareSlider";
import Tile from "./Tile";

const SETTING_NAMES = { type: "Corruption", severity: "Severity", p: "Noise probability", kernel: "Kernel size",
  sigma: "Sigma", n_rects: "Boxes", coverage: "Area covered", target_coverage: "Target area", rects: "Boxes (x, y, w, h)", seed: "Random seed", note: "Note" };
const fmt = (k, v) => {
  if (k === "coverage" || k === "target_coverage") return `${(v * 100).toFixed(1)}%`;
  if (k === "rects") return v.map((r) => `(${r.join(", ")})`).join(" ");
  if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(3);
  return String(v).replace("_", " ");
};

// Images (compare slider or grid), downloads and the corruption settings: shared by Tasks 1 and 2.
// `readouts` and `children` let each workspace add its own numbers and panels.
export default function RestorationResult({ result, readouts, children }) {
  const [view, setView] = useState("compare");
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented label="Result view" size="sm" value={view} onChange={setView}
          options={[{ value: "compare", label: "Compare" }, { value: "grid", label: "All images" }]} />
        <div className="flex gap-2">
          <button type="button" className="btn-quiet" onClick={() => download(result.input_image, "model_input.png")}>Download input</button>
          <button type="button" className="btn-quiet" onClick={() => download(result.restored_image, "restored.png")}>Download restored</button>
        </div>
      </div>

      {view === "compare" ? (
        <div className="max-w-[30rem]">
          <CompareSlider before={result.input_image} after={result.restored_image} beforeLabel="Model input" afterLabel="Restored" />
        </div>
      ) : (
        <div className="grid max-w-3xl grid-cols-2 gap-3 sm:grid-cols-4">
          {result.reference_image && <Tile src={result.reference_image} label="Clean target" />}
          <Tile src={result.input_image} label="Model input" />
          <Tile src={result.restored_image} label="Restored" />
          {result.error_map && <Tile src={result.error_map} label="Error map" note="(brighter = larger)" />}
        </div>
      )}

      {children}

      <div className="grid max-w-3xl grid-cols-2 gap-3 sm:grid-cols-4">{readouts}</div>

      <div className="max-w-3xl rounded-2xl bg-white p-4 ring-1 ring-line">
        <h2 className="mb-2 text-sm font-bold">Corruption settings used</h2>
        <dl className="grid grid-cols-[9rem_1fr] gap-x-3 gap-y-1 text-sm">
          {Object.entries(result.corruption).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-ink-soft">{SETTING_NAMES[k] ?? k}</dt>
              <dd className="break-words font-semibold">{fmt(k, v)}</dd>
            </div>
          ))}
          <dt className="text-ink-soft">Image size</dt>
          <dd className="font-semibold">{result.image_size} × {result.image_size} px</dd>
        </dl>
      </div>
    </div>
  );
}
