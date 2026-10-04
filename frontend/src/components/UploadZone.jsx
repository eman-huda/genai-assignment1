import { useRef, useState } from "react";

// Click or drag an image here. Shows a preview of the chosen file.
export default function UploadZone({ file, onFile, hint }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const preview = file ? URL.createObjectURL(file) : null;

  const pick = (f) => f && f.type.startsWith("image/") && onFile(f);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]); }}
      className={`flex items-center gap-4 rounded-2xl border-2 border-dashed p-3 transition-colors ${
        over ? "border-lilac-500 bg-lilac-50" : "border-sky-300 bg-white/70"
      }`}
    >
      <div className="grid h-20 w-20 shrink-0 place-items-center overflow-hidden rounded-xl bg-sky-100">
        {preview ? (
          <img src={preview} alt="Selected upload" className="h-full w-full object-cover" onLoad={() => URL.revokeObjectURL(preview)} />
        ) : (
          <svg aria-hidden="true" viewBox="0 0 24 24" className="h-8 w-8 text-sky-500"><path fill="currentColor" d="M5 20h14v-2H5v2Zm7-16-5 5h3v5h4V9h3l-5-5Z"/></svg>
        )}
      </div>
      <div className="min-w-0">
        <button type="button" className="btn-quiet" onClick={() => input.current.click()}>
          {file ? "Choose another image" : "Choose an image"}
        </button>
        <p className="mt-1.5 truncate text-xs text-ink-faint">{file ? file.name : hint}</p>
      </div>
      <input ref={input} type="file" accept="image/png,image/jpeg,image/webp,image/bmp" className="sr-only"
             onChange={(e) => pick(e.target.files[0])} />
    </div>
  );
}
