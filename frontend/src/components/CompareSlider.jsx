import { useState } from "react";

// Before/after viewer. The model input sits on the left of the handle, the result on the right.
export default function CompareSlider({ before, after, beforeLabel, afterLabel }) {
  const [pos, setPos] = useState(50);
  return (
    <figure className="w-full">
      <div className="relative aspect-square w-full overflow-hidden rounded-2xl bg-sky-100 ring-1 ring-line">
        <img src={after} alt={afterLabel} className="pixel absolute inset-0 h-full w-full" />
        <img src={before} alt={beforeLabel} className="pixel absolute inset-0 h-full w-full"
             style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }} />
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(116,98,192,.5)]"
             style={{ left: `${pos}%` }}>
          <div className="absolute top-1/2 -ml-3.5 -mt-3.5 grid h-7 w-7 place-items-center rounded-full bg-white text-lilac-600 shadow-soft ring-1 ring-lilac-200">
            <svg aria-hidden="true" viewBox="0 0 24 24" className="h-4 w-4"><path fill="currentColor" d="M9 6 3 12l6 6V6Zm6 0v12l6-6-6-6Z"/></svg>
          </div>
        </div>
        <span className="absolute left-3 top-3 rounded-full bg-white/85 px-2.5 py-1 text-xs font-bold text-ink">{beforeLabel}</span>
        <span className="absolute right-3 top-3 rounded-full bg-lilac-100/90 px-2.5 py-1 text-xs font-bold text-lilac-700">{afterLabel}</span>
        <input type="range" min="0" max="100" value={pos} onChange={(e) => setPos(Number(e.target.value))}
               aria-label={`Drag to compare ${beforeLabel} and ${afterLabel}`}
               className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0" />
      </div>
    </figure>
  );
}
