"""
File Index — semantic search over local files using ChromaDB + Ollama embeddings.

Indexes text files from configured directories, chunked and embedded via
nomic-embed-text (already available in Ollama). Results feed into the
conductor as a tool: "find files about X".
"""

import asyncio
import hashlib
import logging
import os
from pathlib import Path

import chromadb
import httpx

logger = logging.getLogger("aiia.indexer")

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

INDEXED_DIRS = [
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "proxy-ai",
    Path.home() / "aiia-brain" / "AIIA-public",
]

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", ".cache", "coverage",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".ts", ".tsx", ".js", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".env", ".sh",
    ".html", ".css", ".sql", ".rst", ".csv",
}

MAX_FILE_BYTES = 512_000  # skip files larger than 512KB


def _get_collection() -> chromadb.Collection:
    db_path = Path.home() / ".aiia" / "file_index"
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        name="files",
        metadata={"hnsw:space": "cosine"},
    )


async def _embed(texts: list[str]) -> list[list[float]]:
    results = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for text in texts:
            resp = await client.post(
                OLLAMA_EMBED_URL,
                json={"model": EMBED_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
    return results


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def _file_id(path: Path, chunk_idx: int) -> str:
    return hashlib.md5(f"{path}:{chunk_idx}".encode()).hexdigest()


def _should_index(path: Path) -> bool:
    if path.stat().st_size > MAX_FILE_BYTES:
        return False
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    for part in path.parts:
        if part in SKIP_DIRS or part.startswith("."):
            return False
    return True


async def index_file(path: Path, collection: chromadb.Collection) -> int:
    try:
        text = path.read_text(errors="ignore")
    except Exception as e:
        logger.debug(f"Skipping {path}: {e}")
        return 0

    chunks = _chunk_text(text)
    if not chunks:
        return 0

    embeddings = await _embed(chunks)
    ids = [_file_id(path, i) for i in range(len(chunks))]
    metadatas = [
        {"path": str(path), "name": path.name, "ext": path.suffix, "chunk": i}
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


async def run_index(dirs: list[Path] | None = None) -> dict:
    dirs = dirs or INDEXED_DIRS
    collection = _get_collection()
    total_files = 0
    total_chunks = 0

    for base in dirs:
        if not base.exists():
            continue
        for root, subdirs, files in os.walk(base):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in files:
                path = Path(root) / fname
                try:
                    if _should_index(path):
                        chunks = await index_file(path, collection)
                        if chunks:
                            total_files += 1
                            total_chunks += chunks
                except Exception as e:
                    logger.debug(f"Error indexing {path}: {e}")

    logger.info(f"Indexed {total_files} files → {total_chunks} chunks")
    return {"files": total_files, "chunks": total_chunks}


async def search_files(query: str, n_results: int = 8) -> list[dict]:
    collection = _get_collection()

    try:
        embeddings = await _embed([query])
    except Exception as e:
        logger.warning(f"Embed failed: {e}")
        return []

    results = collection.query(
        query_embeddings=embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    seen_paths: set[str] = set()
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        path = meta.get("path", "")
        score = round(1 - dist, 3)
        hits.append({
            "path": path,
            "name": meta.get("name", ""),
            "score": score,
            "excerpt": doc[:300].strip(),
            "first": path not in seen_paths,
        })
        seen_paths.add(path)

    return hits
