import { useEffect, useRef, useState } from "react";
import { api, download } from "../api";
import Segmented from "../components/Segmented";
import UploadZone from "../components/UploadZone";
import Readout from "../components/Readout";
import ModelDetails from "../components/ModelDetails";
import Notice from "../components/Notice";
import WorkspaceHeader from "../components/WorkspaceHeader";

// Descriptions come from the FS2K training sketches of each style.
const STYLES = [
  { id: 1, name: "Style 1", text: "Light, thin outlines" },
  { id: 2, name: "Style 2", text: "Bold strokes, dense shading" },
  { id: 3, name: "Style 3", text: "Balanced shading and detail" },
];

function Webcam({ onCapture }) {
  const video = useRef(null);
  const stream = useRef(null);
  const [on, setOn] = useState(false);
  const [error, setError] = useState("");

  const stop = () => { stream.current?.getTracks().forEach((t) => t.stop()); stream.current = null; setOn(false); };
  useEffect(() => stop, []);

  async function start() {
    setError("");
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 640, facingMode: "user" } });
      video.current.srcObject = stream.current;
      setOn(true);
    } catch {
      setError("The camera could not be opened. Allow camera access in the browser, or upload a photo instead.");
    }
  }

  function capture() {
    const v = video.current;
    const side = Math.min(v.videoWidth, v.videoHeight);          // centre square crop
    const c = document.createElement("canvas");
    c.width = c.height = side;
    c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, side, side);
    c.toBlob((b) => { onCapture(new File([b], "webcam.png", { type: "image/png" })); stop(); }, "image/png");
  }

  return (
    <div className="space-y-2">
      <div className="relative aspect-square overflow-hidden rounded-2xl bg-sky-100 ring-1 ring-line">
        <video ref={video} autoPlay playsInline muted className={`h-full w-full -scale-x-100 object-cover ${on ? "" : "hidden"}`} />
        {!on && <p className="absolute inset-0 grid place-items-center p-6 text-center text-sm text-ink-soft">Centre your face in the frame, with good light from the front.</p>}
      </div>
      {on ? (
        <div className="flex gap-2">
          <button type="button" className="btn-primary flex-1" onClick={capture}>Take photo</button>
          <button type="button" className="btn-quiet" onClick={stop}>Close camera</button>
        </div>
      ) : (
        <button type="button" className="btn-quiet w-full" onClick={start}>Open camera</button>
      )}
      {error && <Notice>{error}</Notice>}
    </div>
  );
}

export default function FaceToSketch({ model }) {
  const [source, setSource] = useState("upload");
  const [file, setFile] = useState(null);
  const [style, setStyle] = useState(1);
  const [allStyles, setAllStyles] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [photoUrl, setPhotoUrl] = useState(null);

  useEffect(() => {
    if (!file) return setPhotoUrl(null);
    const u = URL.createObjectURL(file); setPhotoUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  const ready = model?.status === "ready";

  async function run() {
    setBusy(true); setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("style", String(style));
    form.append("all_styles", String(allStyles));
    try { setResult({ ...(await api.sketch(form)), requested: style }); } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  const main = result?.sketches.find((s) => s.style === result.requested) ?? result?.sketches[0];
  return (
    <section>
      <WorkspaceHeader title="Face-to-Sketch Generator">
        A conditional GAN draws a face photo as a pencil sketch in one of three artist styles from the FS2K dataset.
      </WorkspaceHeader>

      {!ready && model && (
        <div className="mb-5"><Notice>The Task 4 model is {model.status}. Add face2sketch_generator.onnx to the models folder and restart the app.</Notice></div>
      )}

      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <div className="space-y-5 rounded-panel bg-lilac-100/70 p-5 ring-1 ring-lilac-200/70">
          <div>
            <span className="field-label">Photo</span>
            <Segmented label="Photo source" value={source} onChange={setSource}
              options={[{ value: "upload", label: "Upload" }, { value: "webcam", label: "Webcam" }]} />
            <div className="mt-3">
              {source === "upload" ? (
                <UploadZone file={file} onFile={setFile} hint="A single, front-facing face works best." />
              ) : file && file.name === "webcam.png" ? (
                <div className="flex items-center gap-3">
                  <img src={photoUrl} alt="Captured photo" className="h-20 w-20 rounded-xl object-cover" />
                  <button type="button" className="btn-quiet" onClick={() => setFile(null)}>Retake photo</button>
                </div>
              ) : (
                <Webcam onCapture={setFile} />
              )}
            </div>
          </div>

          <div>
            <span className="field-label">Sketch style</span>
            <div role="radiogroup" aria-label="Sketch style" className="grid gap-2">
              {STYLES.map((s) => (
                <button key={s.id} type="button" role="radio" aria-checked={style === s.id} onClick={() => setStyle(s.id)}
                  className={`flex items-baseline justify-between rounded-2xl px-4 py-2.5 text-left ring-1 transition-colors ${
                    style === s.id ? "bg-white ring-2 ring-lilac-600" : "bg-white/60 ring-lilac-200 hover:bg-white"}`}>
                  <span className="font-bold">{s.name}</span>
                  <span className="text-xs text-ink-soft">{s.text}</span>
                </button>
              ))}
            </div>
            <label className="mt-3 flex items-center gap-2 text-sm text-ink-soft">
              <input type="checkbox" checked={allStyles} onChange={(e) => setAllStyles(e.target.checked)} className="h-4 w-4 accent-lilac-600" />
              Also draw the other two styles for comparison
            </label>
          </div>

          <button type="button" className="btn-primary w-full" disabled={!ready || !file || busy} onClick={run}>
            {busy ? "Drawing…" : "Generate sketch"}
          </button>
          {error && <Notice>{error}</Notice>}
          <ModelDetails model={model} />
        </div>

        <div className="min-w-0">
          {!result ? (
            <div className="grid min-h-[24rem] place-items-center rounded-panel border-2 border-dashed border-sky-300 bg-sky-50/70 p-8 text-center">
              <p className="max-w-xs text-ink-soft">Upload or capture a face photo, pick a style, then select <b className="text-ink">Generate sketch</b>. The photo and its sketch appear side by side here.</p>
            </div>
          ) : (
            <div className="space-y-5">
              <div className="grid max-w-3xl grid-cols-2 gap-4">
                <figure>
                  <div className="aspect-square overflow-hidden rounded-2xl bg-sky-100 ring-1 ring-line">
                    <img src={photoUrl ?? result.photo_input} alt="Original photo" className="h-full w-full object-cover" />
                  </div>
                  <figcaption className="mt-1.5 text-xs font-bold text-ink-soft">Original photo</figcaption>
                </figure>
                <figure>
                  <div className="aspect-square overflow-hidden rounded-2xl bg-white ring-1 ring-line">
                    <img src={main.image_512} alt={`Generated sketch, Style ${main.style}`} className="h-full w-full" />
                  </div>
                  <figcaption className="mt-1.5 text-xs font-bold text-ink-soft">Generated sketch, Style {main.style}</figcaption>
                </figure>
              </div>

              <div className="flex flex-wrap gap-2">
                <button type="button" className="btn-primary" onClick={() => download(main.image_512, `sketch_style${main.style}_512.png`)}>Download sketch</button>
                <button type="button" className="btn-quiet" onClick={() => download(main.image, `sketch_style${main.style}_128.png`)}>Download model output (128 px)</button>
              </div>

              {result.sketches.length > 1 && (
                <div className="max-w-3xl">
                  <h2 className="mb-2 text-sm font-bold">The same photo in all three styles</h2>
                  <div className="grid grid-cols-3 gap-3">
                    {result.sketches.map((s) => (
                      <figure key={s.style}>
                        <img src={s.image_512} alt={`Style ${s.style}`} className="aspect-square w-full rounded-xl bg-white ring-1 ring-line" />
                        <figcaption className="mt-1 text-xs font-bold text-ink-soft">Style {s.style}</figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              )}

              <div className="grid max-w-3xl grid-cols-2 gap-3 sm:grid-cols-4">
                <Readout value={`${result.timing_ms.inference.toFixed(1)} ms`} label={`Model inference (${result.sketches.length} ${result.sketches.length > 1 ? "sketches" : "sketch"})`} tone="lilac" />
                <Readout value={`${result.timing_ms.total.toFixed(0)} ms`} label="Total on server" tone="lilac" />
                <Readout value={`${result.image_size} px`} label="Model resolution" />
                <Readout value={`Style ${main.style}`} label="Condition used" />
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
