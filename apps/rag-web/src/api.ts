
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

export type IngestResp = { paths: { normalized: string } };

export async function ingestFile(file: File): Promise<IngestResp> {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch(`${API_BASE}/ingest`, { method: "POST", body: fd });
  if (!resp.ok) throw new Error(`Ingest failed: ${resp.status}`);
  return resp.json();
}

export async function embedNormalized(path: string, docId: string, kind: string) {
  const resp = await fetch(`${API_BASE}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ normalized_path: path, doc_id: docId, kind })
  });
  if (!resp.ok) throw new Error(`Embed failed: ${resp.status}`);
  return resp.json();
}

export function streamGenerate(query: string, topK = 6): EventSource {
  const url = new URL(`${API_BASE}/generate_stream`);
  url.searchParams.set("query", query);
  url.searchParams.set("top_k", String(topK));
  return new EventSource(url.toString());
}
