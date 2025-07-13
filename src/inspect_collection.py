import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "educational_chunks"

print("🔍 Caricamento ChromaDB...")
client = PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

count = collection.count()
print(f"📦 Numero totale di chunk nella collezione: {count}")

if count == 0:
    print("⚠️ La collezione è vuota. Assicurati di aver eseguito 'embedder.py'")
else:
    print("\n📄 Esempi di documenti:")
    results = collection.get(include=["metadatas", "documents"])
    
    for i in range(min(10, count)):  # mostra max 10 per non impazzire
        meta = results["metadatas"][i]
        text = results["documents"][i][:200].replace("\n", " ")

        print(f"\n🔹 chunk_id: {results['ids'][i]}")
        print(f"   📚 title  : {meta.get('title', '-')}")
        print(f"   📖 subject: {meta.get('subject', '-')}")
        print(f"   👶 level  : {meta.get('level', '-')}")
        print(f"   ✏️ text   : {text}...")
