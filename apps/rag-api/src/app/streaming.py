import os
import json
import asyncio
from typing import AsyncGenerator, List, Dict, Any
from openai import OpenAI

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def build_messages(query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = (
        "You are a concise, accurate assistant. Answer ONLY from the provided context snippets. "
        "If the answer is not in the snippets, say you don't know. "
        "Add citation markers like [1], [2] immediately after claims they support."
    )
    lines = ["Question:", query, "", "Context snippets:"]
    for i, c in enumerate(contexts, start=1):
        src = c.get("source_path")
        txt = (c.get("text") or "").strip()
        lines.append(f"[{i}] {txt}\n(Source: {src})")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


async def stream_answer(query: str, contexts: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = build_messages(query, contexts)

    # Start streaming
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,
        stream=True,  # <— important
    )

    # Emit SSE lines: "data: {json}\n\n"
    yield "event: start\ndata: {}\n\n"
    for event in stream:
        delta = event.choices[0].delta
        chunk = delta.content or ""
        if chunk:
            # Stream token chunk
            payload = json.dumps({"type": "token", "content": chunk})
            yield f"data: {payload}\n\n"
        await asyncio.sleep(0)  # be friendly to the loop

    yield "event: end\ndata: {}\n\n"
