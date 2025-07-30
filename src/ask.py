import os
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# === CONFIG ===
CHROMA_DIR = "chroma_db"
CHUNK_LIMIT = 4  # massimo numero di chunk da includere

# === INIT ===
print("🔍 Loading embedding model and ChromaDB...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="educational_chunks")

# === LLM ===
llm = ChatGroq(
    model="llama3-8b-8192",  # puoi cambiare in "mixtral-8x7b-32768" se vuoi
    api_key=os.getenv("GROQ_API_KEY")  # deve essere settata nel tuo ambiente
)

# === FUNCTIONS ===
def get_top_chunks(query: str, k: int = CHUNK_LIMIT):
    q_emb = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[q_emb],
                           n_results=k,
                           include=["documents", "metadatas"])
    n_matches = len(res["ids"][0]) if res["ids"] else 0
    print(f"Trovati {n_matches} chunk simili.")
    return res["documents"][0], res["metadatas"][0]

def build_context(docs, metas):
    parts = []
    for i in range(len(docs)):
        source = metas[i].get("title", "Sconosciuto")
        level = metas[i].get("level", "?")
        subject = metas[i].get("subject", "?")
        text = docs[i]
        parts.append(f"[Fonte: {source} | Livello: {level} | Materia: {subject}]\n{text}")
    return "\n\n".join(parts)

def ask_llm(prompt: str) -> str:
    messages = [
        SystemMessage(content="Sei un assistente educativo. Rispondi in modo chiaro e semplice usando solo le informazioni fornite."),
        HumanMessage(content=prompt)
    ]
    response = llm(messages)
    return response.content

# === MAIN LOOP ===
if __name__ == "__main__":
    print("Fai una domanda (\"exit\" per uscire)")

    while True:
        query = input("\n🔎 Domanda: ").strip()
        if query.lower() in ["exit", "quit", "esci"]:
            break

        docs, metas = get_top_chunks(query)
        if not docs:
            print("Nessuna risposta trovata.")
            continue

        context = build_context(docs, metas)
        prompt = f"""Usa le seguenti fonti per rispondere alla domanda in modo semplice e adatto al livello indicato.

Fonti:
{context}

Domanda: {query}
Risposta:"""

        print("\nRisposta generata:\n")
        print(ask_llm(prompt))
