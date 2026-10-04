export default function WorkspaceHeader({ title, children }) {
  return (
    <header className="mb-6 max-w-2xl">
      <h1 className="text-[1.9rem] font-extrabold leading-tight tracking-tight text-ink">{title}</h1>
      <p className="mt-1.5 text-[0.97rem] leading-relaxed text-ink-soft">{children}</p>
    </header>
  );
}
