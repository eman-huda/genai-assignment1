// One labelled image in a grid, shown pixel-sharp at its true resolution.
export default function Tile({ src, label, note }) {
  return (
    <figure>
      <div className="aspect-square overflow-hidden rounded-xl bg-sky-100 ring-1 ring-line">
        {src && <img src={src} alt={label} className="pixel h-full w-full" />}
      </div>
      <figcaption className="mt-1.5 text-xs font-bold text-ink-soft">
        {label}
        {note && <span className="font-semibold text-ink-faint"> {note}</span>}
      </figcaption>
    </figure>
  );
}
