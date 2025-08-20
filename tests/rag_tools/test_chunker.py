import os
import json
import tempfile
from pathlib import Path
import pytest

from src.rag_tools import chunker


def test_split_text_into_chunks_basic():
    text = " ".join([f"word{i}" for i in range(1200)])  # 1200 parole
    chunks = chunker.split_text_into_chunks(text, size=200, overlap=20)

    # deve generare circa (1200-200)/(200-20) ≈ 6 chunks
    assert len(chunks) > 5
    assert all(isinstance(c, str) for c in chunks)
    # controllo overlap: l'ultima parola del primo chunk deve comparire anche nel secondo
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_get_source_meta_found_and_not_found():
    index = [{"id": "doc1", "titolo": "T", "materia": "storia", "classe": "2", "anno": 2024}]
    meta = chunker.get_source_meta("doc1", index)
    assert meta["titolo"] == "T"

    with pytest.raises(KeyError):
        chunker.get_source_meta("missing", index)


def test_load_source_index_and_chunk_all_sources(tmp_path, monkeypatch):
    # setup fake index.json
    fake_index = [{
        "id": "s1",
        "titolo": "Titolo1",
        "materia": "mate",
        "classe": "3",
        "anno": 2025
    }]
    index_path = tmp_path / "fonte_index.json"
    index_path.write_text(json.dumps(fake_index), encoding="utf-8")

    # setup fake cleaned file
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    cleaned_file = cleaned_dir / "s1.txt"
    cleaned_file.write_text(" ".join([f"w{i}" for i in range(600)]), encoding="utf-8")

    # patch paths
    monkeypatch.setattr(chunker, "SOURCE_INDEX_PATH", str(index_path))
    monkeypatch.setattr(chunker, "CLEANED_DIR", cleaned_dir)
    monkeypatch.setattr(chunker, "CHUNK_DIR", tmp_path / "chunks")

    # run chunk_all_sources
    chunker.chunk_all_sources()

    out_file = tmp_path / "chunks" / "s1.jsonl"
    assert out_file.exists()

    # verifica contenuto JSONL
    with open(out_file, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    assert all("text" in r for r in lines)
    assert all("metadata" in r for r in lines)
    assert lines[0]["metadata"]["title"] == "Titolo1"
    assert lines[0]["metadata"]["subject"] == "mate"
