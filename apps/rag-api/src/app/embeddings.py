import os
from typing import List
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    resp = _client.embeddings.create(model=_MODEL, input=texts)
    return [d.embedding for d in resp.data]
