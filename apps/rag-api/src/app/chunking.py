import tiktoken
from typing import List

_enc = tiktoken.get_encoding("cl100k_base")


def tokenize(s: str) -> List[int]:
    return _enc.encode(s)


def detokenize(ids: List[int]) -> str:
    return _enc.decode(ids)


def chunk_text(text: str, chunk_tokens: int = 400, overlap: int = 60) -> List[str]:
    ids = tokenize(text)
    if not ids:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_tokens - overlap)
    while i < len(ids):
        piece = ids[i : i + chunk_tokens]
        chunks.append(detokenize(piece))
        i += step
    return chunks
