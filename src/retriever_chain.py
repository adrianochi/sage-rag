from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import os

CHROMA_DIR = "../data/chroma_db"
COLLECTION_NAME = "educational_chunks"
CHUNK_LIMIT = 4

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

def query_chunks(question: str, subject: str = None, classe: str = None, anno: int = None):
    embedding = embedder.encode(question).tolist()

    filters = {}
    if subject:
        filters["subject"] = subject
    if classe:
        filters["classe"] = classe
    if anno is not None:
        filters["anno"] = anno

    query_args = {
        "query_embeddings": [embedding],
        "n_results": CHUNK_LIMIT,
        "include": ["documents", "metadatas"]
    }

    # 🔧 Fix per filtro compatibile con ChromaDB
    if len(filters) == 1:
        # Se c'è solo un filtro, passalo diretto
        query_args["where"] = next(iter([{k: v} for k, v in filters.items()]))
    elif len(filters) > 1:
        # Se ce ne sono più di uno, usa $and
        query_args["where"] = {"$and": [{k: v} for k, v in filters.items()]}

    # 🔍 Debug (opzionale)
    print("🔎 Filtro usato:", query_args.get("where"))

    results = collection.query(**query_args)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    print("📎 RISULTATI TROVATI:")
    for i, meta in enumerate(metas):
        print(f" - {i+1}. subject={meta.get('subject')}, classe={meta.get('classe')}, anno={meta.get('anno')}, title={meta.get('title')}")

    return docs, metas

def build_context(docs, metas):
    parts = []
    for i in range(len(docs)):
        source = metas[i].get("title", "Sconosciuto")
        classe = metas[i].get("classe", "?")
        anno = metas[i].get("anno", "?")
        subject = metas[i].get("subject", "?")
        text = docs[i]
        parts.append(f"[Fonte: {source} | Classe: {classe} {anno} | Materia: {subject}]\n{text}")
    return "\n\n".join(parts)

def build_rag_chain(llm):
    def invoke(input_dict):
        query = input_dict["query"]
        subject = input_dict.get("subject")
        classe = input_dict.get("classe")
        anno = input_dict.get("anno")

        docs, metas = query_chunks(query, subject=subject, classe=classe, anno=anno)

        if not docs:
            return {"result": "{}", "source_documents": []}

        context = build_context(docs, metas)
        prompt = f"""Use the following sources to answer the question in a simple way, suitable for the indicated class and grade level.
All the content should be written in Italian.

Fonti:
{context}

Domanda: {query}
Risposta (usa solo le fonti, altrimenti non rispondere):"""

        messages = [
            SystemMessage(content="Sei un assistente educativo. Rispondi solo usando le fonti fornite."),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        return {
            "result": response.content,
            "source_documents": docs
        }

    return type("FakeChain", (), {"invoke": staticmethod(invoke)})()
