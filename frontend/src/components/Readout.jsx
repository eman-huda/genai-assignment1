// Small figure with a label underneath, for timings and quality numbers.
export default function Readout({ value, label, tone = "sky" }) {
  const bg = tone === "lilac" ? "bg-lilac-50 ring-lilac-200" : "bg-sky-50 ring-sky-200";
  return (
    <div className={`rounded-2xl px-4 py-3 ring-1 ${bg}`}>
      <div className="text-lg font-extrabold tabular-nums whitespace-nowrap text-ink">{value}</div>
      <div className="text-xs font-semibold text-ink-soft">{label}</div>
    </div>
  );
}
