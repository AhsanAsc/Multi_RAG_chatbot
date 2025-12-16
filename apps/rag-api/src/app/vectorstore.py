import os
import uuid
from typing import Iterable, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException
import math

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_docs")
DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

_client = QdrantClient(url=QDRANT_URL, timeout=30.0)


def _validate_vec(v: list[float], dim: int):
    if not isinstance(v, list) or len(v) != dim:
        raise ValueError(f"Vector length {len(v) if isinstance(v, list) else 'n/a'} != {dim}")
    if not all((x is not None) and math.isfinite(float(x)) for x in v):
        raise ValueError("Vector contains non-finite values")


def ensure_collection():
    collections = _client.get_collections().collections
    names = {c.name for c in collections}
    if COLLECTION in names:
        return
    _client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
    )


def upsert_vectors(items: Iterable[Dict[str, Any]]):
    # items: {"id": <optional>, "vector": list[float], "payload": dict}
    points = []
    for it in items:
        v = it["vector"]
        _validate_vec(v, DIM)
        pid = it.get("id")
        if not pid:
            pid = str(uuid.uuid4())
        else:
            # ensure it's a UUID string; if not, convert deterministically
            try:
                _ = uuid.UUID(str(pid))
                pid = str(pid)
            except Exception:
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(pid)))
        points.append(
            qm.PointStruct(
                id=pid,
                vector=it["vector"],
                payload=it["payload"],
            )
        )
    _client.upsert(collection_name=COLLECTION, points=points)


def search_vector(vector: List[float], top_k: int = 5):
    return _client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
    )


def points_count() -> int:
    try:
        return _client.count(COLLECTION, exact=True).count
    except Exception:
        return 0


def safe_search_vector(vector: List[float], top_k: int = 5):
    _validate_vec(vector, DIM)
    if points_count() == 0:
        return []  # no data indexed yet
    try:
        return _client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
    except (UnexpectedResponse, ResponseHandlingException):
        # Return empty instead of exploding the whole request
        return []
