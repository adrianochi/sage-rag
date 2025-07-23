from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

CHROMA_DIR = "chroma_db"
CHUNK_LIMIT = 4

def build_rag_chain(llm):
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedder)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": CHUNK_LIMIT}),
        return_source_documents=True
    )
