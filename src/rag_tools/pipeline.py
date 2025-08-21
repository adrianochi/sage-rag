#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot pipeline to automate:
CSV (classe/anno/keyword) -> download Wikipedia HTML -> clean text -> chunk -> embed in ChromaDB

Requirements (pip):
  beautifulsoup4 requests tqdm sentence-transformers chromadb
Optional but recommended:
  unidecode (to help with URL slugs if you use non-ASCII titles)

Usage examples:
  python pipeline.py                       # -> processa TUTTI i CSV in sources_csv/, poi embed
  python pipeline.py --csv-dir sources_csv # -> idem, cartella custom
  python pipeline.py --csv sources_csv/topics_storia_primaria.csv
  python pipeline.py --csv-dir sources_csv --limit 50 --skip-embed
  python pipeline.py --csv-dir sources_csv --fresh-db

Directory layout creato sotto ./data :
  data/raw/       (downloaded HTML)
  data/cleaned/   (plain text extracted)
  data/chunks/    (jsonl chunks)
  data/chroma_db/ (Chroma persistence)
  data/fonte_index.json (metadata registry)

CSV columns expected:
  materia, classe, anno, titolo, keyword_wikipedia
- materia: 'storia' | 'geografia' | ecc.
- classe : 'prim' | 'sec1' | 'sec2'
- anno   : intero (es. 1..5)
- titolo : human-friendly topic name
- keyword_wikipedia: page title su it.wikipedia.org

Idempotente: se l'HTML grezzo esiste, salta il download (a meno di --force).
Batch embed eseguito UNA volta alla fine di tutti i CSV (a meno di --skip-embed).
"""

import os, re, csv, json, sys, time, shutil, argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ----------------- Config -----------------
DATA_DIR      = Path("../../data")
RAW_DIR       = DATA_DIR / "raw"
CLEANED_DIR   = DATA_DIR / "cleaned"
CHUNKS_DIR    = DATA_DIR / "chunks"
CHROMA_DIR    = DATA_DIR / "chroma_db"
SOURCE_INDEX  = DATA_DIR / "fonte_index.json"

DEFAULT_CSV_DIR = Path("sources_csv")

CHUNK_SIZE = 500   # words
OVERLAP    = 50    # words
BATCH_SIZE = 100

WIKI_BASE  = "https://it.wikipedia.org/wiki/"

# -------------- Utils ---------------------
def safe_slug(s: str) -> str:
    s = s.strip().replace(" ", "_")
    return s

def page_url_from_keyword(keyword: str) -> str:
    return WIKI_BASE + quote(safe_slug(keyword))

def clean_filename_from_url(url: str) -> str:
    path = url.split("/wiki/", 1)[-1]
    return path.replace("/", "_") or "index"

def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, CLEANED_DIR, CHUNKS_DIR, CHROMA_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def load_fonte_index():
    if SOURCE_INDEX.exists():
        with open(SOURCE_INDEX, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_fonte_index(idx):
    SOURCE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(SOURCE_INDEX, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)

def upsert_source_metadata(source):
    idx = load_fonte_index()
    kept = [s for s in idx if s.get("id") != source["id"]]
    kept.append(source)
    save_fonte_index(kept)

def download_html(url: str, force: bool=False, user_agent: str="Mozilla/5.0"):
    ensure_dirs()
    filename = clean_filename_from_url(url) + ".html"
    out_path = RAW_DIR / filename
    if out_path.exists() and not force:
        return filename, out_path

    headers = {"User-Agent": user_agent}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    out_path.write_text(r.text, encoding="utf-8")
    return filename, out_path

def extract_text_from_html(html_path: Path) -> Path:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", {"id": "mw-content-text"}) or soup.find("article") or soup.body
    tags = content.find_all(["h1", "h2", "h3", "p", "li"]) if content else []
    text = "\n\n".join(t.get_text(" ", strip=True) for t in tags).strip()
    out_path = CLEANED_DIR / (html_path.stem + ".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path

def split_into_chunks(text: str, size: int=CHUNK_SIZE, overlap: int=OVERLAP):
    words = re.split(r"\s+", text.strip())
    chunks = []
    start = 0
    n = len(words)
    if n == 0:
        return []
    while start < n:
        end = min(start + size, n)
        chunks.append(" ".join(words[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def chunk_and_write(source_id: str, title: str, subject: str, classe: str, anno: int, cleaned_path: Path):
    raw_text = cleaned_path.read_text(encoding="utf-8")
    chunks = split_into_chunks(raw_text)
    out_path = CHUNKS_DIR / f"{source_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as fp:
        for i, ch in enumerate(chunks):
            record = {
                "id": f"{source_id}_{i}",
                "text": ch,
                "metadata": {
                    "source_id": source_id,
                    "title": title,
                    "subject": subject,
                    "classe": classe,
                    "anno": int(anno),
                    "created_at": datetime.now().isoformat()
                }
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path, len(chunks)

# ----------------- Embedding -----------------
def embed_all(chroma_dir: Path = CHROMA_DIR, batch_size: int = BATCH_SIZE):
    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer
    client = PersistentClient(path=str(chroma_dir))
    col = client.get_or_create_collection(name="educational_chunks")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    files = list(CHUNKS_DIR.glob("*.jsonl"))
    total = 0
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for _ in f:
                total += 1

    print(f"\nEmbedding ~{total} chunks da {len(files)} file ...")
    docs, ids, metas = [], [], []
    k = 0

    def flush():
        nonlocal docs, ids, metas, k
        if not docs:
            return
        embeds = model.encode(docs, convert_to_numpy=True).tolist()
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
        k += len(docs)
        print(f"  -> aggiunti {len(docs)} (totale {k})")
        docs.clear(); ids.clear(); metas.clear()

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs.append(obj["text"])
                ids.append(obj.get("id"))
                metas.append({
                    "source_id": obj["metadata"]["source_id"],
                    "title": obj["metadata"]["title"],
                    "subject": obj["metadata"]["subject"],
                    "classe": obj["metadata"]["classe"],
                    "anno": int(obj["metadata"]["anno"]),
                    "created_at": obj["metadata"]["created_at"]
                })
                if len(docs) >= batch_size:
                    flush()
    flush()
    print("Done. (Il count esatto della collection richiede una query separata.)")

# ----------------- Core -----------------
def process_csv_file(csv_path: Path, limit: int = 0, force: bool = False):
    """Processa un singolo CSV end-to-end (download -> clean -> chunk)."""
    print(f"\n>>> Processing CSV: {csv_path.name}")
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"[SKIP FILE] {csv_path}: {e}")
        return

    if limit and limit > 0:
        rows = rows[:limit]

    for r in tqdm(rows, desc=f"Topics ({csv_path.name})"):
        try:
            materia = r["materia"].strip()
            classe  = r["classe"].strip()
            anno    = int(r["anno"])
            titolo  = r["titolo"].strip()
            kw      = r["keyword_wikipedia"].strip()
        except KeyError as e:
            print(f"[SKIP ROW] Colonna mancante {e} in {csv_path.name}")
            continue
        except Exception as e:
            print(f"[SKIP ROW] Errore parsing riga in {csv_path.name}: {e}")
            continue

        url = page_url_from_keyword(kw)
        try:
            filename, raw_path = download_html(url, force=force)
        except Exception as e:
            print(f"[SKIP] {kw} -> {e}")
            continue

        source_id = filename.replace(".html", "")
        upsert_source_metadata({
            "id": source_id,
            "titolo": titolo,
            "materia": materia,
            "classe": classe,
            "anno": anno,
            "fonte": url,
            "formato": "html",
            "salvato_il": datetime.now().isoformat()
        })

        cleaned_path = extract_text_from_html(raw_path)
        _, n_chunks = chunk_and_write(
            source_id=source_id,
            title=titolo,
            subject=materia,
            classe=classe,
            anno=anno,
            cleaned_path=cleaned_path
        )
        print(f"  -> {kw}: {n_chunks} chunks")

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Singolo CSV (materia,classe,anno,titolo,keyword_wikipedia)")
    ap.add_argument("--csv-dir", default=str(DEFAULT_CSV_DIR), help="Cartella con più CSV (default: sources_csv)")
    ap.add_argument("--limit", type=int, default=0, help="Processa al massimo N righe per CSV (0 = tutte)")
    ap.add_argument("--force", action="store_true", help="Forza redownload HTML anche se esiste")
    ap.add_argument("--fresh-db", action="store_true", help="Resetta la cartella ChromaDB prima dell'embed finale")
    ap.add_argument("--skip-embed", action="store_true", help="Esegue tutto tranne l'embed finale")
    args = ap.parse_args()

    ensure_dirs()

    if args.fresh_db and CHROMA_DIR.exists():
        print("Resetting ChromaDB directory ...")
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    csv_paths = []

    # 1) Se è passato --csv, usalo
    if args.csv:
        p = Path(args.csv)
        if p.exists() and p.is_file():
            csv_paths.append(p)
        else:
            print(f"[WARN] --csv non trovato: {p}")

    # 2) Aggiungi tutti i .csv nella cartella --csv-dir (default: sources_csv/)
    if args.csv_dir:
        dir_path = Path(args.csv_dir)
        if dir_path.exists() and dir_path.is_dir():
            found = sorted(dir_path.glob("*.csv"))
            if found:
                # Evita duplicati se lo stesso file è già in lista
                existing = {str(p.resolve()) for p in csv_paths}
                csv_paths += [p for p in found if str(p.resolve()) not in existing]
        else:
            print(f"[WARN] Cartella CSV non trovata: {dir_path}")

    # 3) Se non c'è nulla, prova default 'sources_csv/' come fallback (già gestito da default arg)
    if not csv_paths:
        print("[ERROR] Nessun CSV trovato. Passa --csv o metti file in sources_csv/.")
        sys.exit(1)

    print(f"Trovati {len(csv_paths)} file CSV da processare.")
    for csv_file in csv_paths:
        process_csv_file(csv_file, limit=args.limit, force=args.force)

    if not args.skip_embed:
        embed_all()

if __name__ == "__main__":
    main()
