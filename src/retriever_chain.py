from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import os

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "educational_chunks"
CHUNK_LIMIT = 4

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

def query_chunks(question: str, subject: str = None, level: str = None):
    embedding = embedder.encode(question).tolist()

    conditions = []
    if subject:
        conditions.append({"subject": subject})
    if level:
        conditions.append({"level": level})

    query_args = {
        "query_embeddings": [embedding],
        "n_results": CHUNK_LIMIT,
        "include": ["documents", "metadatas"]
    }

    if conditions:
        query_args["where"] = {"$and": conditions} if len(conditions) > 1 else conditions[0]

    results = collection.query(**query_args)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    # 👇 DEBUG
    print("📎 RISULTATI TROVATI:")
    for i, meta in enumerate(metas):
        print(f" - {i+1}. subject={meta.get('subject')}, level={meta.get('level')}, title={meta.get('title')}")

    return docs, metas



def build_context(docs, metas):
    parts = []
    for i in range(len(docs)):
        source = metas[i].get("title", "Sconosciuto")
        level = metas[i].get("level", "?")
        subject = metas[i].get("subject", "?")
        text = docs[i]
        parts.append(f"[Fonte: {source} | Livello: {level} | Materia: {subject}]\n{text}")
    return "\n\n".join(parts)

def build_rag_chain(llm):
    # fittizio, per compatibilità con il resto del codice
    def invoke(input_dict):
        query = input_dict["query"]
        subject = input_dict.get("subject")
        level = input_dict.get("level")
        docs, metas = query_chunks(query, subject=subject, level=level)

        if not docs:
            return {"result": "{}", "source_documents": []}

        context = build_context(docs, metas)
        prompt = f"""Usa le seguenti fonti per rispondere alla domanda in modo semplice e adatto al livello indicato.

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
