# ask2.py – Educational RAG CLI with Groq + LangChain

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === CONFIG ===
CHROMA_DIR = "chroma_db"
CHUNK_LIMIT = 4
LLM_PROVIDER = "groq"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Assicurati che sia settato!

# === EMBEDDINGS & VECTOR DB ===
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedder
)

# === LLM SELECTION ===
def get_llm(provider: str):
    if provider == "groq":
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192",
            temperature=0
        )
    raise ValueError(f"LLM provider non supportato: {provider}")

llm = get_llm(LLM_PROVIDER)

# === CUSTOM PROMPT ===
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Sei un assistente educativo. Devi creare dei quiz in base al soggetto della domanda. I quiz possono essere di due tipi. A scelta multipla o a risposta aperta, scegli tu. Usa solo le fonti qui sotto per rispondere alla domanda in modo chiaro, conciso e adatto al livello scolastico indicato.
Non inventare nulla. Se le fonti non contengono la risposta, rispondi "Non ho trovato una risposta nelle fonti disponibili."

Fonti:
{context}

Domanda:
{question}

Risposta:
"""
)

# === RETRIEVAL QA CHAIN ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": CHUNK_LIMIT}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# === CLI ===
if __name__ == "__main__":
    print("🧠 Educational RAG attivo – Digita una domanda (\"exit\" per uscire)")
    while True:
        query = input("\n🔎 Domanda: ").strip()
        if query.lower() in ["exit", "quit", "esci"]:
            break

        result = qa_chain.invoke({"query": query})
        print("\n📚 Risposta generata:\n")
        print(result["result"])
