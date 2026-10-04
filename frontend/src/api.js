// All requests go to /api, which nginx (in Docker) or Vite (in development) forwards to FastAPI.
async function handle(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status}).`;
    throw new Error(detail);
  }
  return body;
}

export const api = {
  health: () => fetch("/api/health").then(handle),
  models: () => fetch("/api/models").then(handle),
  samples: () => fetch("/api/samples").then(handle),
  sampleUrl: (name) => `/api/samples/${encodeURIComponent(name)}`,
  restore: (form) => fetch("/api/universal/restore", { method: "POST", body: form }).then(handle),
  hardRoute: (form) => fetch("/api/hard-routing/restore", { method: "POST", body: form }).then(handle),
  softMoe: (form) => fetch("/api/soft-moe/restore", { method: "POST", body: form }).then(handle),
  sketch: (form) => fetch("/api/sketch", { method: "POST", body: form }).then(handle),
};

export function download(dataUrl, filename) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
