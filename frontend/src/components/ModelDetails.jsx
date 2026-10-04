// Collapsible technical details about the deployed model(s), read from /api/models.
const short = (v) => (typeof v === "number" && !Number.isInteger(v) ? v.toPrecision(3) : Array.isArray(v) ? `[${v.join(", ")}]` : v);
const cfg = (c) => c && Object.entries(c).filter(([, v]) => typeof v !== "object" || Array.isArray(v))
  .map(([k, v]) => `${k} ${short(v)}`).join(", ");

export default function ModelDetails({ model }) {
  if (!model) return null;
  const multi = model.parts && model.parts.length > 1;
  const rows = [
    ["Status", model.status],
    ...(multi
      ? model.parts.map((p) => [p.part === "classifier" ? "Classifier" : `Expert: ${p.part.replace("_", " ")}`,
          `${p.file} (${p.size_mb} MB), outputs ${p.outputs.map((o) => o.name).join(", ")}`])
      : [["ONNX file", model.file && `${model.file} (${model.size_mb} MB)`],
         ["Inputs", model.inputs?.map((i) => `${i.name} ${JSON.stringify(i.shape)} ${i.type}`).join(", ")],
         ["Outputs", model.outputs?.map((o) => `${o.name} ${JSON.stringify(o.shape)}`).join(", ")]]),
    ["Parameters", model.params?.toLocaleString()],
    ["Latent size", model.latent_dim?.toLocaleString()],
    ["Best epoch", model.best_epoch],
    ["Load time", model.load_ms && `${model.load_ms} ms`],
    ["Routing rule", model.routing_rule],
    ["Temperature", model.temperature?.toFixed?.(3)],
    ["Classifier", cfg(model.classifier_config)],
    ["Specialists", cfg(model.specialist_config)],
    ["Configuration", cfg(model.config)],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");
  return (
    <details className="group rounded-2xl bg-white/70 p-4 ring-1 ring-line">
      <summary className="cursor-pointer list-none text-sm font-bold text-ink">
        <span className="mr-1 inline-block transition-transform group-open:rotate-90">›</span> Model details
      </summary>
      <dl className="mt-3 grid grid-cols-[7.5rem_1fr] gap-x-3 gap-y-1.5 text-xs">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="font-bold text-ink-soft">{k}</dt>
            <dd className="break-words text-ink">{String(v)}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}
