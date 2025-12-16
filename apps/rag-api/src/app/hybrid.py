import sqlite3
from pathlib import Path
from typing import Dict, Any, List

DB_PATH = Path("/app/data/hybrid.db")


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_fts():
    con = _conn()
    con.execute(
        """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(
      text,
      doc_id UNINDEXED,
      kind UNINDEXED,
      source_path UNINDEXED,
      chunk_index UNINDEXED,
      chunk_id UNINDEXED,
      tokenize = 'unicode61'
    );
    """
    )
    con.commit()
    con.close()


def _escape_fts(q: str) -> str:
    # Wrap as a phrase; escape inner quotes
    return '"' + (q or "").replace('"', '""') + '"'


def fts_search(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Try parameterized MATCH first. If the build rejects it, fall back to a fully literal,
    safely quoted query with an integer-inlined LIMIT (no placeholders).
    """
    con = _conn()
    out: List[Dict[str, Any]] = []
    try:
        cur = con.execute(
            "SELECT chunk_id, text, doc_id, kind, source_path, chunk_index, bm25(chunks_fts) AS bscore "
            "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bscore LIMIT ?",
            (query, int(limit)),
        )
    except sqlite3.OperationalError:
        # Fallback: literal SQL — quoted MATCH, integer LIMIT (sanitized)
        qlit = _escape_fts(query)
        lim = max(1, min(int(limit or 50), 200))  # clamp 1..200
        sql = (
            "SELECT chunk_id, text, doc_id, kind, source_path, chunk_index, bm25(chunks_fts) AS bscore "
            f"FROM chunks_fts WHERE chunks_fts MATCH {qlit} ORDER BY bscore LIMIT {lim}"
        )
        try:
            cur = con.execute(sql)
        except sqlite3.OperationalError:
            # Final safety: if FTS still errors (weird tokens), return empty results
            con.close()
            return out

    for row in cur.fetchall():
        chunk_id, text, doc_id, kind, source_path, chunk_index, bscore = row
        out.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "doc_id": doc_id,
                "kind": kind,
                "source_path": source_path,
                "chunk_index": int(chunk_index),
                "bm25_score": float(bscore),
            }
        )
    con.close()
    return out
