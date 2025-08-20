import json
from pathlib import Path
import numpy as np

import pytest
from src.rag_tools import embedder


# ----------------- Dummies -----------------
class DummyModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, convert_to_numpy=True):
        self.calls.append(list(texts))
        return np.array([[0.1, 0.2] for _ in texts])


class DummyCollection:
    def __init__(self):
        self.add_calls = []
        self._count = 0

    def add(self, ids, embeddings, documents, metadatas):
        self.add_calls.append(
            dict(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        )
        self._count += len(ids)

    def count(self):
        return self._count


# ----------------- Tests -----------------
def test_clean_metadata_defaults():
    m = embedder.clean_metadata({})
    assert m == {
        "source_id": "",
        "title": "",
        "subject": "",
        "classe": "",
        "anno": "",
        "created_at": "",
    }


def test_batch_iter_chunks_ok():
    data = list(range(5))
    chunks = list(embedder.batch_iter(data, 2))
    assert chunks == [[0, 1], [2, 3], [4]]


def test_load_chunks_and_embed_pipeline(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()

    (chunks_dir / "a.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "a_0", "text": "t0", "metadata": {"title": "T0", "subject": "S", "classe": "C", "anno": 1}}),
                json.dumps({"id": "a_1", "text": "t1", "metadata": {"title": "T1"}}),
            ]
        ),
        encoding="utf-8",
    )
    (chunks_dir / "b.jsonl").write_text(
        json.dumps({"id": "b_0", "text": "t2", "metadata": {}}),
        encoding="utf-8",
    )

    dummy_model = DummyModel()
    dummy_collection = DummyCollection()
    monkeypatch.setattr(embedder, "get_model", lambda: dummy_model)
    monkeypatch.setattr(embedder, "get_collection", lambda *a, **k: dummy_collection)

    total, count = embedder.embed_all(
        fresh=False,
        chroma_dir=str(tmp_path / "chroma"),
        chunks_dir=str(chunks_dir),
    )

    assert total == 3
    assert count == 3

    # ordine non garantito: verifica per contenuto
    assert len(dummy_collection.add_calls) == 1
    call = dummy_collection.add_calls[0]

    expected_ids = {"a_0": "t0", "a_1": "t1", "b_0": "t2"}
    expected_titles = {"a_0": "T0", "a_1": "T1", "b_0": ""}

    assert set(call["ids"]) == set(expected_ids.keys())
    for i, _id in enumerate(call["ids"]):
        assert call["documents"][i] == expected_ids[_id]
        assert call["metadatas"][i]["title"] == expected_titles[_id]


def test_embed_all_honors_batch_size(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "chunks2"
    chunks_dir.mkdir()
    lines = [json.dumps({"id": f"id{i}", "text": f"t{i}", "metadata": {}}) for i in range(5)]
    (chunks_dir / "x.jsonl").write_text("\n".join(lines), encoding="utf-8")

    dummy_model = DummyModel()
    dummy_collection = DummyCollection()
    monkeypatch.setattr(embedder, "get_model", lambda: dummy_model)
    monkeypatch.setattr(embedder, "get_collection", lambda *a, **k: dummy_collection)

    total, count = embedder.embed_all(
        fresh=False,
        chroma_dir=str(tmp_path / "chroma2"),
        chunks_dir=str(chunks_dir),
        batch_size=2,
    )

    assert total == 5
    assert count == 5
    assert len(dummy_collection.add_calls) == 3
    assert len(dummy_collection.add_calls[0]["ids"]) == 2


def test_embed_all_fresh_resets_dir(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma3"
    chroma_dir.mkdir()

    chunks_dir = tmp_path / "chunks3"
    chunks_dir.mkdir()
    (chunks_dir / "c.jsonl").write_text(
        json.dumps({"id": "c0", "text": "t", "metadata": {}}),
        encoding="utf-8",
    )

    dummy_model = DummyModel()
    dummy_collection = DummyCollection()
    monkeypatch.setattr(embedder, "get_model", lambda: dummy_model)
    monkeypatch.setattr(embedder, "get_collection", lambda *a, **k: dummy_collection)

    total, count = embedder.embed_all(
        fresh=True,
        chroma_dir=str(chroma_dir),
        chunks_dir=str(chunks_dir),
    )
    assert total == 1
    assert count == 1
