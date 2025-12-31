# memory_module.py
# =========================
# MEMORY MODULE (NO CHROMA) — SQLite(FTS5) + Knowledge Graph + STM
#   Persistent Long-term Memory (LTM): SQLite + optional FTS5 (RAG retrieval + metadata)
#   Semantic Long-term Memory: Knowledge Graph (KG) persisted to GraphML + JSON
#   Short-term Memory (STM): in-session deque (fast context scratchpad)
#
# Design goals:
# - Drop-in compatible with agent prompts: rag_add / rag_search / rag_upsert_single /
#   rag_get_latest_registry / dataset_to_memory / mem_load
# - Deterministic registry: FILE_PATH_REGISTRY stored as singleton doc (overwriteable)
# - Robust as a module: SAFE to import repeatedly (NO auto-bootstrap on import)
# - Retrieval: FTS keyword search + recency decay (prefer newer memory)
#
# IMPORTANT:
# - In notebooks/agents: call set_file_paths(...) then init_and_bootstrap(...) explicitly.
# - Never generates synthetic data; fails loudly if CSV missing.
# =========================

import os, json, time, hashlib, shutil, sqlite3, math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from collections import deque

import pandas as pd
import networkx as nx

# Optional: OpenAI for KG extraction (safe import)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ============================================================
# 0) MODULE-LEVEL PATH STATE (set by set_file_paths)
# ============================================================
FILE_PATH: str = ""
csv_paths: List[str] = []


def set_file_paths(file_path: str = "", paths: Optional[List[str]] = None) -> None:
    """
    Set dataset paths for this module (so it does NOT rely on notebook globals()).

    Typical usage (in AutoGen coder block):
        import memory_module as mem
        mem.set_file_paths(FILE_PATH, csv_paths)
        mem.init_and_bootstrap(paths=mem.csv_paths)
    """
    global FILE_PATH, csv_paths
    FILE_PATH = str(file_path or "").strip()
    if paths is None:
        csv_paths = [FILE_PATH] if FILE_PATH else []
    else:
        csv_paths = [str(p).strip() for p in (paths or []) if str(p).strip()]


def get_file_paths() -> Dict[str, Any]:
    """Convenience getter for debugging."""
    return {"FILE_PATH": FILE_PATH, "csv_paths": list(csv_paths)}


# ============================================================
# 1) PATHS + RESET FLAGS
# ============================================================
RESET_MEMORY = False  # True => wipes SQLite DB + KG files (persistent reset)

WORK_DIR = "/content/pavement_agentic_workspace"
os.makedirs(WORK_DIR, exist_ok=True)

SQLITE_PATH = os.path.join(WORK_DIR, "memory.sqlite")
MEM_PATH = os.path.join(WORK_DIR, "memory.jsonl")  # compatibility placeholder (not used for persistence)

KG_DIR = os.path.join(WORK_DIR, "kg")
os.makedirs(KG_DIR, exist_ok=True)
KG_GRAPHML_PATH = os.path.join(KG_DIR, "semantic_kg.graphml")
KG_JSON_PATH = os.path.join(KG_DIR, "semantic_kg.json")

if RESET_MEMORY:
    if os.path.exists(SQLITE_PATH):
        os.remove(SQLITE_PATH)
    if os.path.exists(KG_DIR):
        shutil.rmtree(KG_DIR, ignore_errors=True)
    os.makedirs(KG_DIR, exist_ok=True)


# ============================================================
# 2) CONFIG
# ============================================================
@dataclass
class RAGConfig:
    llm_model: str = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 6
    decay_lambda_days: float = 0.06   # score *= exp(-lambda * age_days)
    max_text_store_chars: int = 200_000


CFG = RAGConfig()


# ============================================================
# 3) SHORT-TERM MEMORY (STM) — SESSION ONLY
# ============================================================
STM_MAX = 120
_STM = deque(maxlen=STM_MAX)


def stm_add(kind: str, text: str, meta: Optional[dict] = None) -> None:
    _STM.append(
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kind": str(kind),
            "text": str(text).strip(),
            "meta": meta or {},
        }
    )


def stm_get(limit: int = 20) -> List[dict]:
    return list(_STM)[-int(limit):]


def mem_load(limit: Optional[int] = None) -> List[dict]:
    """
    STM ONLY (NOT persistent).
    Use this for quick session recap (debugging / agent trace).
    """
    rows = list(_STM)
    if limit is not None:
        rows = rows[-int(limit):]
    return rows


# ============================================================
# 4) SQLITE INIT (Persistent LTM + FTS)
# ============================================================
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _init_sqlite() -> None:
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS docs (
        uid TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        ts REAL NOT NULL,
        updated_ts REAL NOT NULL,
        text TEXT NOT NULL,
        meta_json TEXT NOT NULL
    );
    """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_kind ON docs(kind);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_docid ON docs(doc_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_ts ON docs(ts);")

    # Optional FTS5
    try:
        cur.execute(
            """
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            uid,
            kind,
            doc_id,
            text,
            tokenize = 'porter'
        );
        """
        )
        conn.commit()
    except Exception:
        # FTS not available; will fallback to LIKE search
        pass

    conn.commit()
    conn.close()


_init_sqlite()


def _fts_available() -> bool:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='docs_fts';")
    ok = cur.fetchone() is not None
    conn.close()
    return ok


# ============================================================
# 5) HELPERS: META SANITATION + CHUNKING + IDS
# ============================================================
def sanitize_meta(meta: Optional[dict]) -> dict:
    meta = meta or {}

    def _to_jsonable(x):
        if x is None or isinstance(x, (str, int, float, bool)):
            return x
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        return str(x)

    return _to_jsonable(meta)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = str(text)
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end]

        if end < n:
            bp = max(piece.rfind("."), piece.rfind("\n"))
            if bp > int(chunk_size * 0.5):
                piece = piece[: bp + 1]
                end = start + bp + 1

        piece = piece.strip()
        if piece:
            chunks.append(piece)

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _stable_uid(kind: str, doc_id: str, chunk_index: int, text: str) -> str:
    h = _md5(f"{kind}|{doc_id}|{chunk_index}|{text}")[:12]
    return f"{kind}:{doc_id}:{chunk_index}:{h}"


# ============================================================
# 6) CORE RAG API: rag_add / rag_upsert_single / rag_search / rag_index_path
# ============================================================
def _insert_or_replace(
    uid: str,
    kind: str,
    doc_id: str,
    text: str,
    meta: dict,
    ts: float,
    force_replace: bool = True,
) -> None:
    text = str(text)
    if len(text) > CFG.max_text_store_chars:
        text = text[: CFG.max_text_store_chars] + "\n[TRUNCATED]"

    meta_json = json.dumps(meta, ensure_ascii=False)
    conn = _connect()
    cur = conn.cursor()

    if force_replace:
        cur.execute(
            """
        INSERT OR REPLACE INTO docs(uid, kind, doc_id, ts, updated_ts, text, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
            (uid, kind, doc_id, ts, ts, text, meta_json),
        )
    else:
        cur.execute(
            """
        INSERT OR IGNORE INTO docs(uid, kind, doc_id, ts, updated_ts, text, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
            (uid, kind, doc_id, ts, ts, text, meta_json),
        )

    if _fts_available():
        cur.execute(
            """
        INSERT OR REPLACE INTO docs_fts(uid, kind, doc_id, text)
        VALUES (?, ?, ?, ?);
        """,
            (uid, kind, doc_id, text),
        )

    conn.commit()
    conn.close()


def rag_add(kind: str, text: str, meta: Optional[dict] = None) -> None:
    """
    Persistent write:
      - STM append (session recap)
      - SQLite insert for each chunk
      - Optional KG extraction for allowlisted kinds (see KG_KINDS_ALLOWLIST)
    """
    meta = sanitize_meta(meta)
    stm_add(kind, text, meta)
    ts = time.time()

    doc_id = str(meta.get("doc_id", "memory"))
    chunks = chunk_text(text, CFG.chunk_size, CFG.chunk_overlap) if text else []

    for i, c in enumerate(chunks or [""]):
        uid = _stable_uid(kind, doc_id, i, c if c else str(text))
        _insert_or_replace(uid, kind, doc_id, c if c else str(text), meta, ts, force_replace=True)

    # KG extraction (optional)
    try:
        if str(kind) in KG_KINDS_ALLOWLIST:
            extraction = extract_triples_llm(kind=kind, text=text, meta=meta)
            kg_add_extraction(_KG, extraction, {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind})
            kg_save(_KG)
            kg_index_facts_to_sqlite_incremental(_KG)
    except Exception:
        # Silent-ish: don’t crash main pipeline if KG fails
        pass


def rag_upsert_single(kind: str, text: str, meta: Optional[dict] = None, fixed_id: Optional[str] = None) -> None:
    """
    Write a SINGLE canonical document (overwriteable).
    Use this for FILE_PATH_REGISTRY and ACTIVE_DATASET_IDS.
    """
    meta = sanitize_meta(meta)
    stm_add(kind, text, meta)
    ts = time.time()

    doc_id = str(meta.get("doc_id", "singleton"))
    uid = fixed_id or f"{kind}:{doc_id}:0"
    _insert_or_replace(uid, kind, doc_id, text, meta, ts, force_replace=True)


def _passes_where_filters(row: dict, where: Optional[dict]) -> bool:
    if not where:
        return True
    meta = row.get("metadata", {}) or {}
    for k, v in where.items():
        if k in ("kind", "doc_id"):
            if row.get(k) != v:
                return False
        else:
            if meta.get(k) != v:
                return False
    return True


def _recency_weight(ts: float) -> float:
    age_days = max(0.0, (time.time() - float(ts)) / 86400.0)
    return math.exp(-CFG.decay_lambda_days * age_days)


def rag_search(query: str, k: int = 5, where: Optional[dict] = None) -> List[dict]:
    """
    Persistent read:
      - Uses FTS5 if available; otherwise LIKE fallback
      - Applies recency decay to prefer newer memory
      - Supports optional metadata filtering via `where`
    Returns: {text, metadata, distance, id, kind, doc_id}
    """
    k = int(k)
    query = str(query).strip()
    out: List[dict] = []

    conn = _connect()
    cur = conn.cursor()

    if _fts_available() and query:
        n_fetch = max(30, k * 6)
        try:
            cur.execute(
                """
            SELECT d.uid, d.kind, d.doc_id, d.ts, d.text, d.meta_json,
                   bm25(docs_fts) AS rank
            FROM docs_fts
            JOIN docs d ON d.uid = docs_fts.uid
            WHERE docs_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?;
            """,
                (query, n_fetch),
            )
            rows = cur.fetchall()
        except Exception:
            rows = []

        for uid, kind, doc_id, ts, text, meta_json, rank in rows:
            meta = json.loads(meta_json) if meta_json else {}
            sim = 1.0 / (1.0 + float(rank if rank is not None else 0.0))
            score = sim * _recency_weight(ts)
            row = {
                "id": uid,
                "kind": kind,
                "doc_id": doc_id,
                "text": text,
                "metadata": meta,
                "distance": float(1.0 - min(0.999999, score)),
                "_score": score,
                "ts": ts,
            }
            if _passes_where_filters(row, where):
                out.append(row)

        out.sort(key=lambda r: (r["distance"], -r["_score"]))
        conn.close()
        return out[:k]

    # Fallback: LIKE search
    n_fetch = max(60, k * 10)
    like = f"%{query}%" if query else "%"

    cur.execute(
        """
    SELECT uid, kind, doc_id, ts, text, meta_json
    FROM docs
    WHERE text LIKE ?
    ORDER BY ts DESC
    LIMIT ?;
    """,
        (like, n_fetch),
    )
    rows = cur.fetchall()
    conn.close()

    for uid, kind, doc_id, ts, text, meta_json in rows:
        meta = json.loads(meta_json) if meta_json else {}
        score = _recency_weight(ts)
        row = {
            "id": uid,
            "kind": kind,
            "doc_id": doc_id,
            "text": text,
            "metadata": meta,
            "distance": float(1.0 - min(0.999999, score)),
            "_score": score,
            "ts": ts,
        }
        if _passes_where_filters(row, where):
            out.append(row)

    out.sort(key=lambda r: (r["distance"], -r["_score"]))
    return out[:k]


def rag_index_path(
    path: str,
    extensions: Optional[List[str]] = None,
    doc_id: Optional[str] = None,
    max_bytes: int = 2_000_000,
) -> int:
    """
    Index project files into persistent memory.
    Safe defaults: .txt/.md/.py/.json only.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"rag_index_path not found: {path}")

    extensions = extensions or [".txt", ".md", ".py", ".json"]
    total_docs = 0

    files = [p] if p.is_file() else list(p.rglob("*"))
    for fp in files:
        if fp.is_dir():
            continue
        if fp.suffix.lower() not in extensions:
            continue

        try:
            size = fp.stat().st_size
            if size > max_bytes:
                continue
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        rag_add(
            kind="indexed_file",
            text=content,
            meta={"doc_id": doc_id or fp.stem, "source": str(fp), "file_type": fp.suffix.lower()},
        )
        total_docs += 1

    return total_docs


# ============================================================
# 7) KNOWLEDGE GRAPH (KG) — semantic long-term memory
# ============================================================
KG_KINDS_ALLOWLIST = {
    "decision",
    "domain_rule",
    "treatment_rule",
    "treatment_thresholds",
    "maintenance_logic",
    "final_conclusion",
    "assumption",
}

TRIPLE_SYSTEM = """You extract semantic memory as a knowledge graph.
Return ONLY valid JSON with this schema:

{
  "entities": [{"id":"...", "type":"...", "aliases":["..."]}],
  "triples": [{"subj":"...", "pred":"...", "obj":"...", "confidence":0.0 to 1.0}]
}

Rules:
- Use compact canonical entity ids (e.g., "IRI", "OverlayThickness_in", "MaintenanceEvent").
- pred must be snake_case verb phrase (e.g., "indicates", "decreased_after", "has_csv", "used_for").
- Keep entities noun-like; do not paste long sentences into entity ids.
- If unsure, output with lower confidence.
- Do NOT add extra keys or commentary.
"""


def kg_new() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.graph["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    G.graph["version"] = "sqlite_fts5_kg_v1"
    return G


def kg_load() -> nx.MultiDiGraph:
    if os.path.exists(KG_GRAPHML_PATH):
        try:
            base = nx.read_graphml(KG_GRAPHML_PATH)
            return nx.MultiDiGraph(base)
        except Exception:
            pass
    return kg_new()


def kg_save(G: nx.MultiDiGraph) -> None:
    try:
        nx.write_graphml(G, KG_GRAPHML_PATH)
    except Exception:
        pass

    nodes = [{"id": str(n), **{k: d[k] for k in d}} for n, d in G.nodes(data=True)]
    edges = []
    for u, v, k, d in G.edges(keys=True, data=True):
        edges.append({"subj": str(u), "obj": str(v), "key": str(k), **{kk: d[kk] for kk in d}})

    Path(KG_JSON_PATH).write_text(
        json.dumps({"graph": dict(G.graph), "nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


_KG = kg_load()


def _get_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None


_OA = _get_openai_client()


def extract_triples_llm(kind: str, text: str, meta: Optional[dict] = None) -> dict:
    if _OA is None:
        raise RuntimeError("OpenAI client not available. KG extraction disabled.")
    meta = meta or {}
    payload = {"kind": str(kind), "meta": meta, "text": str(text)[:6000]}
    resp = _OA.chat.completions.create(
        model=CFG.llm_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": TRIPLE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def kg_add_extraction(G: nx.MultiDiGraph, extraction: dict, provenance: dict) -> None:
    ts = provenance.get("ts", "")
    kind = provenance.get("kind", "")

    for e in extraction.get("entities", []):
        eid = e.get("id")
        if not eid:
            continue
        if eid not in G:
            G.add_node(
                eid,
                type=e.get("type", "unknown"),
                aliases=json.dumps(e.get("aliases", []), ensure_ascii=False),
                created_at=str(ts),
                created_from=str(kind),
            )
        else:
            try:
                old = json.loads(G.nodes[eid].get("aliases", "[]"))
            except Exception:
                old = []
            new = e.get("aliases", []) or []
            merged = sorted(set([str(x) for x in (old + new)]))
            G.nodes[eid]["aliases"] = json.dumps(merged, ensure_ascii=False)

    for t in extraction.get("triples", []):
        s, p, o = t.get("subj"), t.get("pred"), t.get("obj")
        if not (s and p and o):
            continue
        if s not in G:
            G.add_node(s, type="unknown", aliases="[]", created_at=str(ts), created_from=str(kind))
        if o not in G:
            G.add_node(o, type="unknown", aliases="[]", created_at=str(ts), created_from=str(kind))

        conf = float(t.get("confidence", 0.5))
        edge_key = f"{p}:{ts}:{hashlib.md5((s+p+o).encode()).hexdigest()[:6]}"
        G.add_edge(
            s,
            o,
            key=edge_key,
            pred=str(p),
            confidence=conf,
            ts=str(ts),
            kind=str(kind),
        )


def kg_index_facts_to_sqlite_incremental(G: nx.MultiDiGraph) -> int:
    count = 0
    now = time.time()
    for u, v, k, d in G.edges(keys=True, data=True):
        pred = d.get("pred", "related_to")
        conf = float(d.get("confidence", 0.5))
        ts = d.get("ts", "")
        src_kind = d.get("kind", "")

        fact = f"{u} {pred} {v}. (confidence={conf:.2f}, source_kind={src_kind}, ts={ts})"
        hid = hashlib.md5(fact.encode("utf-8")).hexdigest()[:10]
        uid = f"kg_fact:{hid}"

        _insert_or_replace(
            uid=uid,
            kind="kg_fact",
            doc_id="kg",
            text=fact,
            meta={
                "subj": str(u),
                "pred": str(pred),
                "obj": str(v),
                "confidence": conf,
                "source_kind": src_kind,
                "ts": ts,
            },
            ts=now,
            force_replace=True,
        )
        count += 1
    return count


# ============================================================
# 8) DATASET MEMORY: registry + dataset anchors
# ============================================================
ACTIVE_DATASET_IDS: List[str] = []


def _safe_read_csv(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, low_memory=False)


def _dataset_id_for_path(path: str, idx: int) -> str:
    stem = Path(path).stem
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return f"dataset_{idx}_{stem}_{h}"


def rag_register_file_paths(paths: List[str], mapping: Dict[str, str]) -> None:
    # Use MODULE STATE (not notebook globals)
    registry = {
        "FILE_PATH": FILE_PATH,
        "csv_paths": list(csv_paths),
        "normalized_paths": paths,
        "path_to_dataset_id": mapping,
        "how_agents_should_use": [
            "1) rag_get_latest_registry() to get dataset_id(s) deterministically",
            "2) rag_search(f'DATASET COLUMNS {dataset_id}', k=5)",
            "3) rag_search(f'DATASET HEAD {dataset_id}', k=5)",
            "4) rag_search(f'DATASET SCHEMA {dataset_id}', k=5)",
            "5) rag_search(f'DATASET PROFILE {dataset_id}', k=5)",
        ],
        "hard_rules": [
            "Never generate synthetic/mock datasets.",
            "If CSV missing/unreadable, stop and request fix.",
            "Treat FILE_PATH_REGISTRY as the source of truth for dataset_id.",
        ],
    }

    rag_upsert_single(
        kind="file_path_registry",
        text="FILE_PATH_REGISTRY\n" + json.dumps(registry, indent=2, ensure_ascii=False),
        meta={"doc_id": "FILE_PATH_REGISTRY", "artifact": "registry"},
        fixed_id="file_path_registry:FILE_PATH_REGISTRY:0",
    )


def rag_get_latest_registry() -> dict:
    res = rag_search(
        "FILE_PATH_REGISTRY",
        k=5,
        where={"kind": "file_path_registry", "doc_id": "FILE_PATH_REGISTRY"},
    )
    if not res:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT text FROM docs WHERE uid = ?;", ("file_path_registry:FILE_PATH_REGISTRY:0",))
        row = cur.fetchone()
        conn.close()
        if not row:
            return {}
        raw = row[0]
    else:
        raw = res[0]["text"]

    if "\n" in raw:
        raw = raw.split("\n", 1)[1].strip()
    try:
        return json.loads(raw)
    except Exception:
        return {}


def dataset_to_memory(
    df: Any,
    file_path: Optional[str] = None,
    dataset_id: str = "dataset",
    sample_rows: int = 8,
) -> None:
    if isinstance(df, pd.DataFrame):
        resolved_df = df
        resolved_path = file_path
    else:
        if not file_path or not isinstance(file_path, str):
            raise ValueError("dataset_to_memory: df is not a DataFrame; file_path must be a CSV path string.")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"dataset_to_memory: CSV not found at file_path:\n{file_path}")
        resolved_df = _safe_read_csv(file_path)
        resolved_path = file_path

    ts_txt = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_base = {
        "doc_id": dataset_id,
        "dataset_id": dataset_id,
        "source": str(resolved_path) if resolved_path else "",
        "ts": ts_txt,
        "rows": int(resolved_df.shape[0]),
        "cols": int(resolved_df.shape[1]),
    }

    dtypes = resolved_df.dtypes.astype(str).to_dict()
    rag_add(
        "dataset_schema",
        f"DATASET SCHEMA {dataset_id}\n" + json.dumps(dtypes, indent=2, ensure_ascii=False)[:12000],
        {**meta_base, "artifact": "schema"},
    )

    rag_add(
        "dataset_columns",
        f"DATASET COLUMNS {dataset_id}\n"
        + json.dumps({"shape": resolved_df.shape, "columns": resolved_df.columns.tolist()}, indent=2, ensure_ascii=False)[
            :12000
        ],
        {**meta_base, "artifact": "columns"},
    )

    try:
        head_md = resolved_df.head(int(sample_rows)).to_markdown(index=False)
    except Exception:
        head_md = str(resolved_df.head(int(sample_rows)))
    rag_add("dataset_head", f"DATASET HEAD {dataset_id}\n" + head_md, {**meta_base, "artifact": "head"})

    try:
        missing = (resolved_df.isna().mean() * 100).round(2).to_dict()
        num = resolved_df.select_dtypes(include="number")
        desc_dict = num.describe().transpose().round(3).to_dict() if num.shape[1] > 0 else {}
        profile = {
            "dataset_id": dataset_id,
            "source": meta_base["source"],
            "shape": list(resolved_df.shape),
            "missing_pct_top": dict(sorted(missing.items(), key=lambda x: -x[1])[:30]),
            "numeric_describe": desc_dict,
            "how_to_retrieve": [
                "rag_get_latest_registry()",
                f"rag_search('DATASET COLUMNS {dataset_id}', k=5)",
                f"rag_search('DATASET HEAD {dataset_id}', k=5)",
                f"rag_search('DATASET SCHEMA {dataset_id}', k=5)",
                f"rag_search('DATASET PROFILE {dataset_id}', k=5)",
            ],
        }
        rag_add(
            "dataset_profile",
            f"DATASET PROFILE {dataset_id}\n" + json.dumps(profile, indent=2, ensure_ascii=False)[:12000],
            {**meta_base, "artifact": "profile"},
        )
    except Exception as e:
        rag_add(
            "dataset_profile",
            f"DATASET PROFILE {dataset_id}\nProfile generation failed: {str(e)[:200]}",
            {**meta_base, "artifact": "profile"},
        )

    rag_add(
        "dataset_anchor",
        f"DATASET ANCHOR {dataset_id}\nSOURCE: {meta_base['source']}\nSHAPE: {resolved_df.shape}",
        {**meta_base, "artifact": "anchor"},
    )

    global ACTIVE_DATASET_IDS
    if dataset_id not in ACTIVE_DATASET_IDS:
        ACTIVE_DATASET_IDS.append(dataset_id)

    rag_upsert_single(
        "active_datasets",
        "ACTIVE_DATASET_IDS\n" + json.dumps(ACTIVE_DATASET_IDS, indent=2),
        {"doc_id": "ACTIVE_DATASET_IDS", "artifact": "active_list"},
        fixed_id="active_datasets:ACTIVE_DATASET_IDS:0",
    )

    stm_add("dataset_stored", f"Stored dataset_id={dataset_id} shape={resolved_df.shape}", meta_base)


# ============================================================
# 9) BOOTSTRAP (EXPLICIT) — no auto-run on import
# ============================================================
def _normalize_csv_paths_from_module_state() -> List[str]:
    paths: List[str] = []
    if isinstance(csv_paths, list) and len(csv_paths) > 0:
        paths = [str(x).strip() for x in csv_paths if str(x).strip()]
    elif isinstance(FILE_PATH, str) and FILE_PATH.strip():
        paths = [FILE_PATH.strip()]
    return list(dict.fromkeys(paths))


def bootstrap_from_paths(paths: List[str], sample_rows: int = 8) -> None:
    paths = [str(p).strip() for p in (paths or []) if str(p).strip()]
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise RuntimeError("No CSV paths provided. Use set_file_paths(...) then init_and_bootstrap(...).")

    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        rag_add("run_start", "Run started but missing CSV paths:\n" + "\n".join(missing), {"doc_id": "run", "csvs": paths})
        rag_register_file_paths(paths, {})
        raise FileNotFoundError("Missing CSV paths:\n" + "\n".join(missing))

    rag_add("run_start", f"Run started. CSV files: {paths}", {"doc_id": "run", "csvs": paths})

    mapping: Dict[str, str] = {p: _dataset_id_for_path(p, i) for i, p in enumerate(paths, start=1)}
    rag_register_file_paths(paths, mapping)

    for p, dsid in mapping.items():
        dataset_to_memory(df="__LOAD_FROM_FILE__", file_path=p, dataset_id=dsid, sample_rows=sample_rows)


def init_and_bootstrap(paths: Optional[List[str]] = None, sample_rows: int = 8) -> dict:
    """
    Convenience initializer for notebooks/agents:
      - If `paths` provided, bootstraps from those
      - Else uses module state (FILE_PATH/csv_paths set via set_file_paths)
    Returns registry dict.
    """
    use_paths = paths if paths is not None else _normalize_csv_paths_from_module_state()
    bootstrap_from_paths(use_paths, sample_rows=sample_rows)
    return rag_get_latest_registry()


def health_check() -> Dict[str, Any]:
    """
    Lightweight diagnostics (safe to call anytime).
    """
    return {
        "WORK_DIR": WORK_DIR,
        "SQLITE_PATH": SQLITE_PATH,
        "KG_DIR": KG_DIR,
        "fts_enabled": _fts_available(),
        "stm_len": len(_STM),
        "kg_nodes": int(_KG.number_of_nodes()),
        "kg_edges": int(_KG.number_of_edges()),
        "module_paths": get_file_paths(),
    }


# ============================================================
# 10) CLI ENTRYPOINT (only runs when executed directly)
# ============================================================
if __name__ == "__main__":
    print("health_check:", json.dumps(health_check(), indent=2))
    # Optional: run bootstrap if paths are set here
    # Example:
    # set_file_paths("/path/to/data.csv", ["/path/to/data.csv"])
    # reg = init_and_bootstrap(sample_rows=8)
    # print("registry(latest):", json.dumps(reg, indent=2)[:1200])
 
