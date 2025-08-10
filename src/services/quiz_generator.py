import json
import re
from llm_provider import get_llm
from retriever_chain import build_rag_chain
from services.prompt_builder import build_prompt

def generate_quiz_from_data(data):
    """
    Genera un quiz usando RAG + LLM in base ai parametri forniti.
    Ritorna sempre un dizionario con:
    - status: 1 (ok), 2 (no fonti), 3 (errore parsing), 4 (campi mancanti)
    - data: contenuto del quiz o {}
    """

    quiz_type = data.get("type")
    category = data.get("category")
    classe = data.get("classe")
    anno = data.get("anno")
    difficulty = data.get("difficulty")
    llm_provider = data.get("llmProvider")

    # 🔍 Validazione: tutti i campi devono essere presenti
    required_fields = [quiz_type, category, classe, anno, difficulty, llm_provider]
    if any(field is None or field == "" for field in required_fields):
        return {"status": 4, "data": {}}

    # 🚀 Inizializza LLM + RAG
    try:
        llm = get_llm(llm_provider)
        rag_chain = build_rag_chain(llm)
    except Exception:
        return {"status": 4, "data": {}}

    prompt = build_prompt(quiz_type, category, difficulty)

    rag_response = rag_chain.invoke({
        "query": prompt,
        "subject": category,
        "classe": classe,
        "anno": anno
    })

    print("📚 RAG result raw:", rag_response.get("result", ""))

    if not rag_response.get("source_documents"):
        return {"status": 2, "data": {}}

    # 📝 Parsing JSON dal risultato LLM
    try:
        match = re.search(r"\{.*\}", rag_response["result"], re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")

        quiz_data = json.loads(match.group())
        if not isinstance(quiz_data, dict):
            raise ValueError("Not a dict")

        return {"status": 1, "data": quiz_data}

    except Exception as e:
        print("JSON PARSE FAILED:", str(e))
        return {"status": 3, "data": {}}
