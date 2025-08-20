# src/RAG-Tools/embedder.py
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "../data/chroma_db"
CHUNKS_DIR = "../data/chunks"
COLLECTION_NAME = "educational_chunks"
BATCH_SIZE = 100


def get_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_collection(chroma_dir: str = CHROMA_DIR, name: str = COLLECTION_NAME):
    client = PersistentClient(path=chroma_dir)
    return client.get_or_create_collection(name=name)


def load_chunks(chunks_dir: str = CHUNKS_DIR) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for path in Path(chunks_dir).glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                all_chunks.append(json.loads(line))
    return all_chunks


def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_id": meta.get("source_id", ""),
        "title": meta.get("title", ""),
        "subject": meta.get("subject", ""),
        "classe": meta.get("classe", ""),
        "anno": meta.get("anno", ""),
        "created_at": meta.get("created_at", ""),
    }


def batch_iter(lst: List[Any], size: int) -> List[List[Any]]:
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def embed_all(
    fresh: bool = False,
    chroma_dir: str = CHROMA_DIR,
    chunks_dir: str = CHUNKS_DIR,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
) -> Tuple[int, int]:
    """
    Ritorna (num_chunks_totali, num_documenti_in_collection_dopo).
    """
    # init
    if fresh and os.path.exists(chroma_dir):
        # reset directory
        import shutil
        shutil.rmtree(chroma_dir)

    model = get_model()
    collection = get_collection(chroma_dir, collection_name)

    chunks = load_chunks(chunks_dir)
    total = len(chunks)

    for batch in batch_iter(chunks, batch_size):
        texts = [c["text"] for c in batch]
        ids = [c.get("chunk_id") or c.get("id") for c in batch]
        metadatas = [clean_metadata(c.get("metadata", {})) for c in batch]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    try:
        count = collection.count()
    except Exception:
        count = -1
    return total, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Reset ChromaDB before embedding")
    args = parser.parse_args()
    total, count = embed_all(fresh=args.fresh)
    print(f"Total chunks to embed: {total}")
    print(f"Documenti nella collection: {count}")


if __name__ == "__main__":
    main()
