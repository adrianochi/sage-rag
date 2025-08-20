# tests/services/test_retriever_chain_random.py
import sys
import types
import pytest

# ---------- PRE-IMPORT PATCH: stub dei moduli pesanti ----------
# Dummy embedder con .encode() che ritorna qualcosa che ha .tolist()
class _DummyEmbedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        # simuliamo un np.array-like che ha .tolist()
        class _Arr(list):
            def tolist(self):
                return list(self)
        return _Arr([0.1, 0.2, 0.3])

# Dummy collection/client Chroma
class _DummyCollection:
    def __init__(self):
        self.last_query = None
        # valore di default, lo cambieremo nei test
        self.result = {"documents": [[]], "metadatas": [[]]}

    def query(self, **kwargs):
        self.last_query = kwargs
        return self.result

class _DummyClient:
    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, name):
        return self._collection

# Dummy ChatGroq (evita richieste reali)
class _DummyChatGroq:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def invoke(self, messages):
        class _Resp: pass
        _Resp.content = '{"ok":true}'
        return _Resp

# istanze condivise per ispezionare lo stato nel modulo importato
_dummy_collection = _DummyCollection()

# costruiamo moduli fake e iniettiamoli in sys.modules
_fake_sentencetrf = types.SimpleNamespace(SentenceTransformer=_DummyEmbedder)
_fake_chromadb   = types.SimpleNamespace(PersistentClient=lambda path: _DummyClient(_dummy_collection))
_fake_groq       = types.SimpleNamespace(ChatGroq=_DummyChatGroq)

sys.modules.setdefault("sentence_transformers", _fake_sentencetrf)
sys.modules.setdefault("chromadb", _fake_chromadb)
sys.modules.setdefault("langchain_groq", _fake_groq)

# ---------- ora possiamo importare il modulo sotto test ----------
from src.services import retriever_chain


# Utility LLM per i test di build_rag_chain
class DummyLLM:
    def __init__(self, content):
        self._content = content
        self.invoked_with = None

    def invoke(self, messages):
        self.invoked_with = messages
        class _Resp: pass
        _Resp.content = self._content
        return _Resp


import numpy as np

def test_query_chunks_filters_and_dedup(monkeypatch):
    # Evita randomicità
    monkeypatch.setattr(retriever_chain.random, "shuffle", lambda seq: None)

    # Mock embedder → array con .tolist()
    monkeypatch.setattr(
        retriever_chain,
        "embedder",
        type("E", (), {"encode": lambda self, x: np.array([0.1, 0.2, 0.3])})()
    )

    # Patch _hash_doc per sicurezza (identifica univocamente per contenuto)
    monkeypatch.setattr(retriever_chain, "_hash_doc", lambda s: f"HASH::{s.strip()}")

    # Prepara candidati con duplicati (stessa lunghezza docs/metas)
    docs = ["A", "B", "A", "C", "C", "D"]
    metas = [
        {"subject": "mate", "classe": "3", "anno": 2025, "title": "tA"},
        {"subject": "mate", "classe": "3", "anno": 2025, "title": "tB"},
        {"subject": "mate", "classe": "3", "anno": 2025, "title": "tA_dup"},
        {"subject": "mate", "classe": "3", "anno": 2025, "title": "tC"},
        {"subject": "mate", "classe": "3", "anno": 2025, "title": "tC_dup"},
        {"subject": "mate", "classe": "3", "anno": 2025, "title": "tD"},
    ]
    fake_result = {"documents": [docs], "metadatas": [metas]}

    # DummyCollection che restituisce esattamente fake_result
    class DummyCollection:
        def __init__(self, result):
            self.result = result
            self.last_query = None
        def query(self, **kwargs):
            self.last_query = kwargs
            return self.result

    dummy_collection = DummyCollection(fake_result)
    # Patcha proprio l’oggetto collection dentro il modulo importato
    monkeypatch.setattr(retriever_chain, "collection", dummy_collection)

    out_docs, out_metas = retriever_chain.query_chunks(
        "quanto fa 2+2?", subject="mate", classe="3", anno=2025
    )

    # dedup: A,B,C,D (ordine preservato perché shuffle è no-op)
    assert out_docs == ["A", "B", "C", "D"]
    assert [m["title"] for m in out_metas] == ["tA", "tB", "tC", "tD"]

    # Verifica filtri
    where = dummy_collection.last_query.get("where")
    assert "$and" in where
    assert {"subject": "mate"} in where["$and"]
    assert {"classe": "3"} in where["$and"]
    assert {"anno": 2025} in where["$and"]

    assert dummy_collection.last_query["n_results"] == retriever_chain.CANDIDATE_LIMIT
    assert dummy_collection.last_query["include"] == ["documents", "metadatas"]



def test_query_chunks_no_results():
    _dummy_collection.result = {"documents": [[]], "metadatas": [[]]}
    out_docs, out_metas = retriever_chain.query_chunks("ciao", subject="storia")
    assert out_docs == []
    assert out_metas == []


def test_build_context_formatting():
    docs = ["testo A", "testo B"]
    metas = [
        {"title": "TitoloA", "classe": "2", "anno": 2024, "subject": "storia"},
        {"title": "TitoloB", "classe": "1", "anno": 2023, "subject": "geo"},
    ]
    ctx = retriever_chain.build_context(docs, metas)
    assert "[Fonte: TitoloA | Classe: 2 2024 | Materia: storia]" in ctx
    assert "testo B" in ctx


def test_build_rag_chain_no_docs(monkeypatch):
    monkeypatch.setattr(retriever_chain, "query_chunks", lambda *a, **kw: ([], []))
    dummy = DummyLLM('{"ignored":true}')
    chain = retriever_chain.build_rag_chain(dummy)
    res = chain.invoke({"query": "ciao"})
    assert res["result"] == "{}"
    assert res["source_documents"] == []


def test_build_rag_chain_with_docs(monkeypatch):
    docs = ["DOC"]
    metas = [{"title": "Tit", "classe": "3", "anno": 2025, "subject": "mate"}]
    monkeypatch.setattr(retriever_chain, "query_chunks", lambda *a, **kw: (docs, metas))

    dummy = DummyLLM('{"ok":true}')
    chain = retriever_chain.build_rag_chain(dummy)
    res = chain.invoke({"query": "quanto fa 2+2?", "subject": "mate", "classe": "3", "anno": 2025})

    assert res["result"] == '{"ok":true}'
    assert res["source_documents"] == docs

    # Il prompt deve contenere Fonti e la domanda
    messages = dummy.invoked_with
    assert any("Fonti:" in m.content for m in messages)
    assert any("quanto fa 2+2?" in m.content for m in messages)
