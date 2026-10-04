// A small row of mutually exclusive options (radio group semantics).
export default function Segmented({ options, value, onChange, label, size = "md" }) {
  const pad = size === "sm" ? "px-3 py-1.5 text-sm" : "px-4 py-2 text-sm";
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap gap-1 rounded-full bg-white/70 p-1 ring-1 ring-sky-300/60">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            disabled={o.disabled}
            onClick={() => onChange(o.value)}
            className={`${pad} rounded-full font-bold transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
              active ? "bg-lilac-200 text-lilac-700" : "text-ink-soft hover:bg-sky-100"
            }`}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
