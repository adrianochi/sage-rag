import os
import json
import glob
import argparse
import shutil

from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

# === CONFIG ===
CHROMA_DIR = "chroma_db"
CHUNKS_DIR = "../data/chunks"
COLLECTION_NAME = "educational_chunks"
BATCH_SIZE = 100

# === INIT MODEL & DB ===
print("Loading embedding model and ChromaDB...")
model = SentenceTransformer("all-MiniLM-L6-v2")
parser = argparse.ArgumentParser()
parser.add_argument("--fresh", action="store_true", help="Reset ChromaDB before embedding")
args = parser.parse_args()

if args.fresh and os.path.exists(CHROMA_DIR):
    print("Resetting ChromaDB directory...")
    shutil.rmtree(CHROMA_DIR)
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# === LOAD CHUNKS ===
def load_chunks():
    all_chunks = []
    for path in Path(CHUNKS_DIR).glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_chunks.append(data)
    return all_chunks

chunks = load_chunks()
print(f"Total chunks to embed: {len(chunks)}")

def clean_metadata(meta):
    return {
        "source_id": meta.get("source_id", ""),
        "title": meta.get("title", ""),
        "subject": meta.get("subject", ""),
        "level": meta.get("level", ""),
        "created_at": meta.get("created_at", "")
    }


# === EMBED & ADD ===
for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="🔗 Embedding chunks"):
    batch = chunks[i:i + BATCH_SIZE]
    texts = [chunk["text"] for chunk in batch]
    ids = [chunk.get("chunk_id") or chunk.get("id") for chunk in batch]
    metadatas = [clean_metadata(chunk.get("metadata", {})) for chunk in batch]


    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


print("All chunks embedded and added to ChromaDB!")
print(f"Documenti nella collection: {collection.count()}")
