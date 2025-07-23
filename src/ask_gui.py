import tkinter as tk
from tkinter import ttk, scrolledtext

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import ollama

# CONFIG
CHROMA_DIR = "chroma_db"
CHUNK_LIMIT = 4
AVAILABLE_MODELS = ["llama3", "mistral", "gemma"]

# INIT
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="educational_chunks")

def get_top_chunks(query: str, k: int = CHUNK_LIMIT):
    q_emb = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas"])
    return res["documents"][0], res["metadatas"][0]

def build_context(docs, metas):
    return "\n\n".join([
        f"[Fonte: {m.get('title', 'Sconosciuto')} | Livello: {m.get('level', '?')} | Materia: {m.get('subject', '?')}]\n{d}"
        for d, m in zip(docs, metas)
    ])

def ask_llm(prompt: str, model: str):
    try:
        res = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Sei un assistente educativo. Rispondi in modo chiaro e semplice usando solo le informazioni fornite."},
                {"role": "user", "content": prompt},
            ]
        )
        return res["message"]["content"]
    except Exception as e:
        return f"❌ Errore LLM: {e}"

# GUI
def handle_query():
    query = entry_question.get().strip()
    selected_model = model_combobox.get().strip()
    output_area.delete("1.0", tk.END)

    if not query:
        output_area.insert(tk.END, "⚠️ Inserisci una domanda.\n")
        return

    docs, metas = get_top_chunks(query)
    if not docs:
        output_area.insert(tk.END, "❌ Nessuna risposta trovata nei documenti.\n")
        return

    context = build_context(docs, metas)
    prompt = f"""Usa le seguenti fonti per rispondere alla domanda in modo semplice e adatto al livello indicato.

Fonti:
{context}

Domanda: {query}
Risposta:"""

    output_area.insert(tk.END, "🧠 Sto generando la risposta...\n\n")
    root.update()
    answer = ask_llm(prompt, selected_model)
    output_area.delete("1.0", tk.END)
    output_area.insert(tk.END, answer)


root = tk.Tk()
root.title("SAGE RAG Assistant")
# === Layout Config ===
'''root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)'''  # bottone

# === Riga 0: Domanda (etichetta + entry larga) ===
label_question = tk.Label(root, text="Domanda:")
label_question.grid(row=0, column=0)

entry_question = tk.Entry(root, width=80)
entry_question.grid(row=0, column=1)

# === Riga 1: Modello (etichetta + combobox) + bottone ===
label_model = tk.Label(root, text="Modello:")
label_model.grid(row=1, column=0)

model_combobox = ttk.Combobox(root, values=AVAILABLE_MODELS, width=20)
model_combobox.set("llama3")
model_combobox.grid(row=1, column=1)

ask_button = tk.Button(root, text="Ask", command=handle_query)
ask_button.grid(row=1, column=2)

# === Riga 2: Area Output ===
output_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=25)
output_area.grid(row=2, column=0)

# Fai crescere l'area output con la finestra
#root.grid_rowconfigure(2, weight=1)

root.mainloop()
