import React, { useRef, useState } from "react";
import Message from "./Message";
import Citation from "./Citation";
import { streamGenerate } from "../api";

type ChatMsg = { role: "user" | "assistant"; content: string; citations?: { n: number; source_path?: string }[] };

export default function Chat() {
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const controller = useRef<EventSource | null>(null);

  async function ask() {
    const q = input.trim();
    if (!q || busy) return;

    setMsgs((m) => [...m, { role: "user", content: q }]);
    setInput("");
    setBusy(true);

    // Start streaming
    const es = streamGenerate(q, 6);
    controller.current = es;
    let answer = "";

    es.addEventListener("start", () => {
      setMsgs((m) => [...m, { role: "assistant", content: "" }]);
    });

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "token") {
          answer += data.content;
          setMsgs((m) => {
            const last = m[m.length - 1];
            const updated = [...m];
            updated[updated.length - 1] = { ...last, content: answer };
            return updated;
          });
        }
      } catch {
        // ignore
      }
    };

    es.addEventListener("end", async () => {
      es.close();
      controller.current = null;
      // Optional: fetch citations via non-stream endpoint (uses same retrieval)
      try {
        const resp = await fetch("http://localhost:8000/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, top_k: 6 })
        });
        if (resp.ok) {
          const json = await resp.json();
          const cites = (json.citations || []) as { n: number; source_path?: string }[];
          setMsgs((m) => {
            const last = m[m.length - 1];
            const updated = [...m];
            updated[updated.length - 1] = { ...last, citations: cites };
            return updated;
          });
        }
      } finally {
        setBusy(false);
      }
    });

    es.onerror = () => {
      es.close();
      controller.current = null;
      setBusy(false);
    };
  }

  function onOpenSource(src?: string) {
    if (!src) return;
    // try to open the raw normalized text path in a new tab if served, or just alert path
    alert(src);
  }

  return (
    <div>
      <div style={{ maxHeight: 380, overflowY: "auto", border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
        {msgs.map((m, i) => (
          <div key={i}>
            <Message role={m.role} content={m.content} />
            {m.role === "assistant" && m.citations && m.citations.length > 0 && (
              <div style={{ margin: "4px 0 10px 0" }}>
                {m.citations.map((c) => (
                  <Citation key={c.n} n={c.n} source_path={c.source_path} onOpen={onOpenSource} />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question…"
          style={{ flex: 1, padding: 10, borderRadius: 6, border: "1px solid #ccc" }}
          onKeyDown={(e) => e.key === "Enter" ? ask() : undefined}
        />
        <button onClick={ask} disabled={busy} style={{ padding: "10px 16px" }}>
          {busy ? "Generating…" : "Ask"}
        </button>
      </div>
    </div>
  );
}
