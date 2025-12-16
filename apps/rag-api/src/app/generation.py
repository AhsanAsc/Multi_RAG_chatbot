import os
from typing import List, Dict, Any
from openai import OpenAI

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # Build [n] markers tied to unique sources in order of appearance
    msgs: List[Dict[str, str]] = []
    # System: strict grounding
    system = (
        "You are a concise, accurate assistant. Answer ONLY from the provided context snippets. "
        "If the answer is not in the snippets, say you don't know. "
        "Add citation markers like [1], [2] immediately after each claim they support."
    )
    msgs.append({"role": "system", "content": system})

    # User content with numbered snippets
    lines = ["Question:", query, "", "Context snippets:"]
    for i, c in enumerate(contexts, start=1):
        src = c.get("source_path")
        txt = (c.get("text") or "").strip()
        lines.append(f"[{i}] {txt}\n(Source: {src})")
    msgs.append({"role": "user", "content": "\n".join(lines)})
    return msgs


def generate_answer(query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = build_prompt(query, contexts)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
    )
    answer = resp.choices[0].message.content

    # Build citation mapping [1..n] -> sources
    citations = []
    for i, c in enumerate(contexts, start=1):
        citations.append(
            {
                "n": i,
                "source_path": c.get("source_path"),
                "doc_id": c.get("doc_id"),
                "chunk_index": c.get("chunk_index"),
            }
        )
    return {"answer": answer, "citations": citations}
