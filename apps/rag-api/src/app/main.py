from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import time

from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import hashlib
from .embeddings import embed_texts
from .vectorstore import ensure_collection, upsert_vectors
from .chunking import chunk_text
from qdrant_client import QdrantClient

from pydantic import BaseModel

from .config import settings
from .extractors import detect_type, extract_text
from .hybrid import ensure_fts

import os
from typing import Dict, Any
from .hybrid import fts_search
from .generation import generate_answer
from .vectorstore import safe_search_vector
from .rerank import mmr
from .streaming import stream_answer

from .vectorstore import QDRANT_URL, COLLECTION

FUSION_K = 60
TOPK_VEC = int(os.getenv("TOPK_VEC", "20"))
TOPK_BM25 = int(os.getenv("TOPK_BM25", "50"))
FUSION_TOPK = int(os.getenv("FUSION_TOPK", "6"))

RERANK_METHOD = os.getenv("RERANK_METHOD", "mmr")
RERANK_K = int(os.getenv("RERANK_K", "6"))
RERANK_LAMBDA = float(os.getenv("RERANK_LAMBDA", "0.7"))


def _apply_rerank(query: str, matches: list[dict[str, Any]], final_k: int | None = None):
    if not matches:
        return []

    k = final_k or RERANK_K
    if RERANK_METHOD == "none":
        return matches[:k]

    # Build vectors for query and candidates
    q_vec = embed_texts([query])[0]
    cand_texts = [m.get("text") or "" for m in matches]
    cand_vecs = embed_texts(cand_texts)

    order = mmr(q_vec, cand_vecs, k=k, lambd=RERANK_LAMBDA)
    return [matches[i] for i in order]


app = FastAPI(title="RAG API", version="0.1.0")

DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
    max_age=600,
)


@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


async def sse_event_gen():
    for i in range(5):
        yield f'data: {{"tick": {i}, "ts": {int(time.time())}}} \n\n'
        await asyncio.sleep(1)


@app.get("/stream")
async def stream():
    return StreamingResponse(sse_event_gen(), media_type="text/event-stream")


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    contents = await file.read()
    mb = len(contents) / (1024 * 1024)
    if mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(413, f"File too large ({mb:.1f}MB > {settings.MAX_UPLOAD_MB}MB)")

    ts = int(time.time())
    raw_dir = settings.DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{ts}_{file.filename}"
    dest.write_bytes(contents)

    kind = detect_type(dest)
    if kind == "unknown":
        raise HTTPException(415, f"Unsupported file type: {dest.suffix}")

    text = extract_text(dest, kind)
    norm_dir = settings.DATA_DIR / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm = norm_dir / (dest.stem + ".txt")
    norm.write_text(text, encoding="utf-8")

    doc_id = sha256_of_file(dest)
    chunks = chunk_text(text) if text else []

    return JSONResponse(
        {
            "doc_id": doc_id,
            "filename": file.filename,
            "kind": kind,
            "bytes": len(contents),
            "chunks": len(chunks),
            "paths": {"raw": str(dest), "normalized": str(norm)},
        }
    )


class EmbedReq(BaseModel):
    normalized_path: str
    doc_id: str | None = None
    kind: str | None = None
    chunk_tokens: int = 400
    overlap: int = 60


@app.post("/embed")
def embed(req: EmbedReq):
    # 1) read normalized text
    from pathlib import Path
    import uuid
    from fastapi import HTTPException
    from .config import settings

    p = Path(req.normalized_path)
    if not p.is_absolute():
        p = Path(settings.DATA_DIR) / "normalized" / p.name
    if not p.exists():
        raise HTTPException(404, f"normalized file not found: {p}")
    if p.is_dir():
        raise HTTPException(400, f"normalized path is a directory, expected a .txt file: {p}")
    if p.suffix.lower() != ".txt":
        raise HTTPException(400, f"normalized path must point to a .txt file: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")

    # 2) chunk (already there)
    chunks = chunk_text(text, chunk_tokens=req.chunk_tokens, overlap=req.overlap)

    # 3) embed in small batches and prepare upserts
    batch = 64
    to_upsert = []
    fts_rows = []
    base_doc_id = req.doc_id or Path(req.normalized_path).stem

    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        vecs = embed_texts(part)
        for j, v in enumerate(vecs):
            idx = i + j
            human_chunk_id = f"{base_doc_id}:{idx}"  # keep this for FTS & payload
            payload = {
                "doc_id": base_doc_id,
                "kind": req.kind,
                "chunk_index": idx,
                "source_path": str(p),
                "text": part[j],
                "chunk_id": human_chunk_id,  # <— add here
            }
            # Qdrant point id must be UUID (idempotent via uuid5 on our human id)
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, human_chunk_id))
            to_upsert.append({"id": point_uuid, "vector": v, "payload": payload})
            fts_rows.append(
                {
                    "chunk_id": human_chunk_id,
                    "text": part[j],
                    "kind": req.kind,
                    "chunk_index": idx,
                }
            )

    if to_upsert:
        upsert_vectors(to_upsert)
    return {"upserted": len(to_upsert)}


class QueryReq(BaseModel):
    query: str
    top_k: int = 5


@app.post("/query")
def query(req: QueryReq):
    vec = embed_texts([req.query])[0]
    hits = safe_search_vector(vec, top_k=req.top_k)
    out = []
    for h in hits:
        pl = h.payload or {}
        out.append(
            {
                "score": getattr(h, "score", None),
                "doc_id": pl.get("doc_id"),
                "kind": pl.get("kind"),
                "chunk_index": pl.get("chunk_index"),
                "source_path": pl.get("source_path"),
                "text": pl.get("text"),
            }
        )
    return {"matches": out}


@app.on_event("startup")
def _startup():
    ensure_collection()
    ensure_fts()


def _clean_fts_query(q: str) -> str:
    q = (q or "").strip()
    # Drop a trailing '?', normalize whitespace
    if q.endswith("?"):
        q = q[:-1]
    return " ".join(q.split())


class HybridQueryReq(BaseModel):
    query: str
    top_k: int | None = None


@app.post("/query_hybrid")
def query_hybrid(req: HybridQueryReq):
    top_k = req.top_k or FUSION_TOPK

    # dense side
    qvec = embed_texts([req.query])[0]
    vhits = safe_search_vector(qvec, top_k=TOPK_VEC)
    khits = fts_search(req.query.strip().rstrip("?"), limit=TOPK_BM25)

    # keyword side
    clean = _clean_fts_query(req.query)
    khits = fts_search(clean, limit=TOPK_BM25)

    # map to ranks
    v_rank: Dict[str, int] = {}
    out_dense: Dict[str, Dict[str, Any]] = {}
    for rank, h in enumerate(sorted(vhits, key=lambda x: -x.score), start=1):
        cid = (
            h.id
            if hasattr(h, "id")
            else str(h.payload.get("doc_id")) + ":" + str(h.payload.get("chunk_index"))
        )
        v_rank[str(cid)] = rank
        out_dense[str(cid)] = {
            "score": h.score,
            "payload": h.payload or {},
        }

    k_rank: Dict[str, int] = {}
    out_kw: Dict[str, Dict[str, Any]] = {}
    for rank, h in enumerate(khits, start=1):
        cid = h["chunk_id"]
        k_rank[cid] = rank
        out_kw[cid] = h

    # RRF fuse
    fused: Dict[str, float] = {}
    for cid, r in v_rank.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (FUSION_K + r)
    for cid, r in k_rank.items():
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (FUSION_K + r)

    # materialize payloads
    def materialize(cid: str) -> Dict[str, Any]:
        if cid in out_dense:
            pl = out_dense[cid]["payload"]
            return {
                "chunk_id": cid,
                "doc_id": pl.get("doc_id"),
                "kind": pl.get("kind"),
                "chunk_index": pl.get("chunk_index"),
                "source_path": pl.get("source_path"),
                "text": pl.get("text"),
            }
        if cid in out_kw:
            h = out_kw[cid]
            return {
                "chunk_id": cid,
                "doc_id": h.get("doc_id"),
                "kind": h.get("kind"),
                "chunk_index": h.get("chunk_index"),
                "source_path": h.get("source_path"),
                "text": h.get("text"),
            }
        return {"chunk_id": cid}

    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[
        : top_k * 3
    ]  # take a wider pool for rerank
    results = [materialize(cid) for cid, _ in ranked]

    # NEW: MMR rerank to final K
    reranked = _apply_rerank(req.query, results, final_k=top_k)
    return {"matches": reranked, "method": "hybrid-rrf+mmr"}


class GenerateReq(BaseModel):
    query: str
    top_k: int | None = None


@app.post("/generate")
def generate(req: GenerateReq):
    # use hybrid retrieval first
    hyb = query_hybrid(HybridQueryReq(query=req.query, top_k=req.top_k))
    contexts = hyb["matches"]
    out = generate_answer(req.query, contexts)
    # include the contexts for transparency/debug
    out["contexts"] = contexts
    return out


@app.get("/admin/collection_info")
def collection_info():
    c = QdrantClient(url=QDRANT_URL)
    ci = c.get_collection(COLLECTION)
    cnt = c.count(COLLECTION, exact=True).count
    return {
        "collection": COLLECTION,
        "dim": ci.config.params.vectors.size,
        "distance": str(ci.config.params.vectors.distance),
        "points_count": cnt,
    }


@app.get("/admin/fts_count")
def fts_count():
    import sqlite3
    from pathlib import Path

    DB = Path("/app/data/hybrid.db")
    if not DB.exists():
        return {"fts_rows": 0}
    con = sqlite3.connect(DB)
    cur = con.execute("SELECT count(*) FROM chunks_fts")
    n = cur.fetchone()[0]
    con.close()
    return {"fts_rows": n}


@app.get("/generate_stream")
def generate_stream(query: str, top_k: int | None = None):
    # Use hybrid to get contexts, then MMR
    hyb = query_hybrid(HybridQueryReq(query=query, top_k=top_k))
    contexts = hyb["matches"]

    async def sse():
        async for line in stream_answer(query, contexts):
            yield line

    return StreamingResponse(sse(), media_type="text/event-stream")
