import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

SOURCE_INDEX_PATH = "../data/fonte_index.json"
CLEANED_DIR       = Path("../data/cleaned")
CHUNK_DIR         = Path("../data/chunks")

CHUNK_SIZE_WORDS = 500   # ≈ 500 parole
OVERLAP_WORDS    = 50    # sovrapposizione

# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #

def load_source_index() -> List[Dict]:
    if Path(SOURCE_INDEX_PATH).exists():
        with open(SOURCE_INDEX_PATH, "r", encoding="utf-8") as fp:
            return json.load(fp)
    raise FileNotFoundError("Source index not found – run add_source.py first")

def get_source_meta(source_id: str, index: List[Dict]) -> Dict:
    for entry in index:
        if entry["id"] == source_id:
            return entry
    raise KeyError(f"Metadata for source_id '{source_id}' not found")

def split_text_into_chunks(
    text: str,
    size: int = CHUNK_SIZE_WORDS,
    overlap: int = OVERLAP_WORDS
) -> List[str]:
    words = re.split(r"\s+", text.strip())
    chunks = []
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

# -------------------------------------------------------------------- #
# Main pipeline
# -------------------------------------------------------------------- #

def chunk_all_sources():
    index = load_source_index()
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    for txt_file in CLEANED_DIR.glob("*.txt"):
        source_id = txt_file.stem
        print(f"Chunking {source_id} ...")

        meta = get_source_meta(source_id, index)

        with open(txt_file, "r", encoding="utf-8") as fp:
            raw_text = fp.read()

        chunks = split_text_into_chunks(raw_text)

        out_path = CHUNK_DIR / f"{source_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as fp:
            for i, chunk in enumerate(chunks):
                record = {
                    "id": f"{source_id}_{i}",
                    "text": chunk,
                    "metadata": {
                        "source_id": source_id,
                        "title": meta["titolo"],
                        "subject": meta["materia"],
                        "level": meta["livello"],
                        "created_at": datetime.now().isoformat()
                    }
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"   ➜ {len(chunks)} chunks → {out_path}")

# -------------------------------------------------------------------- #

if __name__ == "__main__":
    chunk_all_sources()
