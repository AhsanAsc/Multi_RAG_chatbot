import React from "react";
import { marked } from "marked";

type Props = { role: "user" | "assistant"; content: string };

export default function Message({ role, content }: Props) {
  const html = marked.parse(content || "");
  const bg = role === "user" ? "#eef" : "#f7f7f7";
  return (
    <div style={{ background: bg, padding: 12, borderRadius: 8, margin: "6px 0" }}>
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  );
}
