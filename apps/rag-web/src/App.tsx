import React, { useState } from "react";
import Upload from "./components/Upload";
import Chat from "./components/Chat";

export default function App() {
  const [readyDocs, setReadyDocs] = useState<{ docId: string; normalized: string }[]>([]);

  return (
    <div style={{ maxWidth: 900, margin: "32px auto", padding: "0 16px" }}>
      <h2>Production RAG Chat</h2>
      <p style={{ color: "#666" }}>
        Upload documents, then ask questions. Answers stream in with citations.
      </p>

      <Upload onEmbedded={(docId, normalized) => setReadyDocs((d) => [...d, { docId, normalized }])} />

      {readyDocs.length > 0 && (
        <div style={{ fontSize: 12, color: "#555", marginBottom: 12 }}>
          Indexed: {readyDocs.length} file(s)
        </div>
      )}

      <Chat />
    </div>
  );
}
