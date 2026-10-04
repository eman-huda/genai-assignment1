// Inline message for errors and missing models. States what happened and what to do.
export default function Notice({ children, tone = "error" }) {
  const style = tone === "error" ? "bg-rose-50 text-rose-800 ring-rose-200" : "bg-lilac-50 text-lilac-700 ring-lilac-200";
  return <p role={tone === "error" ? "alert" : "status"} className={`rounded-2xl px-4 py-3 text-sm font-semibold ring-1 ${style}`}>{children}</p>;
}
