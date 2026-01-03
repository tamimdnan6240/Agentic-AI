# -------------------------
# Paths / workspace
# -------------------------
WORK_DIR = "/content/pavement_agentic_workspace"
os.makedirs(WORK_DIR, exist_ok=True)

MEM_PATH = os.path.join(WORK_DIR, "memory.jsonl")
KG_PATH  = os.path.join(WORK_DIR, "knowledge_graph.graphml")
REGISTRY_PATH = os.path.join(WORK_DIR, "file_path_registry.json")  # registry mapping path -> dataset_id

# -------------------------
# TF-IDF cache (avoid refitting every search)
# -------------------------
_tfidf_cache = {
    "dirty": True,
    "rows": [],
    "corpus": [],
    "vect": None,
    "X": None,
}

def _mem_load_raw() -> List[Dict[str, Any]]:
    """Load JSONL memory safely."""
    if not os.path.exists(MEM_PATH):
        return []
    rows = []
    with open(MEM_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # skip corrupt lines
                pass
    return rows

def _tfidf_rebuild_if_needed():
    """Rebuild TF-IDF matrix if new memories were added."""
    global _tfidf_cache
    if not _tfidf_cache["dirty"]:
        return

    rows = _mem_load_raw()
    corpus = [str(r.get("text", "")) for r in rows]

    if not corpus:
        _tfidf_cache = {"dirty": False, "rows": [], "corpus": [], "vect": None, "X": None}
        return

    # ✅ small safety: if corpus exists but extremely small, TF-IDF still works fine
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
    X = vect.fit_transform(corpus)

    _tfidf_cache.update({
        "dirty": False,
        "rows": rows,
        "corpus": corpus,
        "vect": vect,
        "X": X
    })

# -------------------------
# Memory: add/load/search
# -------------------------
def mem_add(kind: str, text: str, meta: Optional[dict] = None):
    """Append a record to JSONL memory."""
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kind": str(kind),
        "text": str(text).strip(),
        "meta": meta or {}
    }
    with open(MEM_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # mark TF-IDF cache dirty
    _tfidf_cache["dirty"] = True

def mem_load() -> List[Dict[str, Any]]:
    """Return all memory rows."""
    return _mem_load_raw()

def mem_search(query: str, k: int = 5, kind: Optional[str] = None) -> List[Tuple[float, Dict[str, Any]]]:
    """TF-IDF similarity search over memory.jsonl."""
    _tfidf_rebuild_if_needed()
    if _tfidf_cache["X"] is None:
        return []

    rows = _tfidf_cache["rows"]
    vect = _tfidf_cache["vect"]
    X = _tfidf_cache["X"]

    q = vect.transform([query])
    sims = cosine_similarity(q, X).ravel()
    order = sims.argsort()[::-1]

    out = []
    for idx in order:
        score = float(sims[idx])
        if score <= 0:
            break
        if kind and rows[idx].get("kind") != kind:
            continue
        out.append((score, rows[idx]))
        if len(out) >= k:
            break
    return out

# -------------------------
# Knowledge Graph (NetworkX) — persisted
# -------------------------
def _kg_load() -> nx.MultiDiGraph:
    """Load KG from GraphML if present; else new graph."""
    if os.path.exists(KG_PATH):
        try:
            g = nx.read_graphml(KG_PATH)
            mg = nx.MultiDiGraph()
            mg.add_nodes_from(g.nodes(data=True))
            mg.add_edges_from(g.edges(data=True))
            return mg
        except Exception:
            pass
    return nx.MultiDiGraph()

_KG = _kg_load()

def _kg_save():
    """Save KG to GraphML (flattened to DiGraph for portability)."""
    g = nx.DiGraph()
    g.add_nodes_from(_KG.nodes(data=True))
    for u, v, data in _KG.edges(data=True):
        g.add_edge(u, v, **data)
    nx.write_graphml(g, KG_PATH)

def kg_add_fact(subj: str, pred: str, obj: str,
                confidence: float = 1.0,
                meta: Optional[dict] = None,
                index_to_memory: bool = True):
    """
    Add a semantic fact to the KG:
    (subj) -[pred]-> (obj)
    """
    meta = meta or {}
    subj = str(subj); pred = str(pred); obj = str(obj)

    _KG.add_node(subj)
    _KG.add_node(obj)
    _KG.add_edge(subj, obj, relation=pred, confidence=float(confidence), **meta)
    _kg_save()

    if index_to_memory:
        mem_add(
            kind="kg_fact",
            text=f"KG FACT: ({subj}) -[{pred}]-> ({obj}) conf={confidence}",
            meta={"subj": subj, "pred": pred, "obj": obj, "confidence": float(confidence), **meta}
        )

def kg_search(node_contains: Optional[str] = None,
              relation: Optional[str] = None,
              limit: int = 20) -> List[Dict[str, Any]]:
    """Search KG edges by node substring and/or relation."""
    out = []
    needle = node_contains.lower() if node_contains else None

    for u, v, data in _KG.edges(data=True):
        rel = data.get("relation")
        if relation and rel != relation:
            continue
        if needle:
            hay = f"{u} {v}".lower()
            if needle not in hay:
                continue

        out.append({
            "subj": u,
            "pred": rel,
            "obj": v,
            "attrs": dict(data)
        })
        if len(out) >= limit:
            break
    return out

# -------------------------
# Optional helpers: dataset registry + artifacts
# -------------------------
def _stable_dataset_id(file_path: str) -> str:
    st = os.stat(file_path)
    raw = f"{file_path}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]

def registry_build_from_csv_paths(csv_paths: List[str]) -> Dict[str, Any]:
    """Create/update a simple registry mapping path -> dataset_id."""
    mapping = {p: _stable_dataset_id(p) for p in csv_paths}
    reg = {
        "created_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "path_to_dataset_id": mapping,
        "latest_path": list(mapping.keys())[-1],
        "latest_dataset_id": list(mapping.values())[-1],
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

    mem_add("registry", f"Registry built with {len(mapping)} datasets.", reg)
    return reg

def registry_load() -> Optional[Dict[str, Any]]:
    if not os.path.exists(REGISTRY_PATH):
        return None
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def store_dataset_artifacts(file_path: str, dataset_id: str):
    """Store columns/head/dtypes in memory + KG facts."""
    df = pd.read_csv(file_path)
    artifacts = {
        "dataset_id": dataset_id,
        "file_path": file_path,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "head": df.head(5).to_dict(orient="records"),
    }
    mem_add(
        kind="dataset_artifacts",
        text=f"DATASET COLUMNS {dataset_id}: " + ", ".join(list(df.columns)[:50]),
        meta={"dataset_id": dataset_id, "file_path": file_path, "artifacts": artifacts}
    )

    kg_add_fact(f"dataset:{dataset_id}", "file_path", file_path, confidence=1.0, meta={}, index_to_memory=False)
    for c in df.columns:
        kg_add_fact(f"dataset:{dataset_id}", "has_column", c, confidence=1.0, meta={}, index_to_memory=False)

def get_dataset_artifacts(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Convenience: fetch stored artifacts meta for a dataset_id."""
    hits = mem_search(f"DATASET COLUMNS {dataset_id}", k=5, kind="dataset_artifacts")
    if not hits:
        return None
    # choose best hit
    return hits[0][1].get("meta", {}).get("artifacts")

# -------------------------
# Quick health check
# -------------------------
def memory_health_check() -> bool:
    os.makedirs(WORK_DIR, exist_ok=True)
    if not os.path.exists(MEM_PATH):
        open(MEM_PATH, "a", encoding="utf-8").close()
    return True

assert memory_health_check()

print("WORK_DIR:", WORK_DIR)
print("MEM_PATH:", MEM_PATH)
print("KG_PATH :", KG_PATH)
print("REGISTRY_PATH:", REGISTRY_PATH)

# ============================================================
# ✅ SECOND FIX: `mm` wrapper to match your agents' prompts
# ============================================================
class MM:
    def health_check(self) -> bool:
        return memory_health_check()

    def rag_add(self, kind: str, text: str, meta: Optional[dict] = None):
        mem_add(kind, text, meta)
        return True

    def rag_search(self, query: str, k: int = 5, kind: Optional[str] = None):
        return mem_search(query, k=k, kind=kind)

    def rag_get_latest_registry(self) -> Dict[str, Any]:
        reg = registry_load()
        if reg is None:
            raise RuntimeError(
                f"Registry not found at {REGISTRY_PATH}. "
                f"Run registry_build_from_csv_paths(csv_paths) before calling mm.rag_get_latest_registry()."
            )
        return reg

    def kg_add_fact(self, subj: str, pred: str, obj: str,
                    confidence: float = 1.0, meta: Optional[dict] = None,
                    index_to_memory: bool = True):
        kg_add_fact(subj, pred, obj, confidence=confidence, meta=meta, index_to_memory=index_to_memory)
        return True

mm = MM()
print("mm.health_check():", mm.health_check())

