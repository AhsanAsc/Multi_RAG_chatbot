import React from "react";

type Props = { n: number; source_path?: string; onOpen?: (src?: string) => void };

export default function Citation({ n, source_path, onOpen }: Props) {
  return (
    <button
      onClick={() => onOpen?.(source_path)}
      style={{ fontSize: 12, marginRight: 6, borderRadius: 12, border: "1px solid #ccc", padding: "2px 8px", cursor: "pointer" }}
      title={source_path}
    >
      [{n}]
    </button>
  );
}
