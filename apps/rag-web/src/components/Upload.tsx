import React, { useState } from "react";
import { ingestFile, embedNormalized } from "../api";

type Props = { onEmbedded: (docId: string, normalized: string) => void };

export default function Upload({ onEmbedded }: Props) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function onSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setMsg("Uploading…");
    try {
      const ing = await ingestFile(file);
      const norm = ing.paths.normalized;
      const kind = file.name.split(".").pop()?.toLowerCase() || "txt";
      const docId = "doc_" + file.name.replace(/\W+/g, "_");
      setMsg("Embedding…");
      await embedNormalized(norm, docId, kind);
      setMsg(`Embedded ✓`);
      onEmbedded(docId, norm);
    } catch (err: any) {
      setMsg(err.message || "Upload failed");
    } finally {
      setBusy(false);
      // clear input so same file can be re-selected
      e.currentTarget.value = "";
    }
  }

  return (
    <div style={{ border: "1px dashed #999", padding: 16, borderRadius: 8, marginBottom: 12 }}>
      <input type="file" onChange={onSelect} disabled={busy} />
      <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
        Supports PDF, DOCX, PPTX, XLSX/CSV, PNG/JPG, TXT
      </div>
      {msg && <div style={{ fontSize: 12, marginTop: 8 }}>{msg}</div>}
    </div>
  );
}
