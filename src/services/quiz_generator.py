# services/quiz_generator.py

import json
import re
from services.llm_provider import get_llm
from services.retriever_chain import build_rag_chain
from services.prompt_builder import build_prompt


from validators.validator_schemas import validate_quiz_data


def generate_quiz_from_data(data):
    """
    Genera un quiz usando RAG + LLM in base ai parametri forniti.

    Ritorna sempre:
      - status:
          1 -> success
          2 -> fonti non disponibili (retriever non ha trovato chunk)
          3 -> errore parsing output LLM (JSON non valido)
          4 -> richiesta non valida / init LLM fallito / schema non rispettato
      - data: contenuto del quiz o {}
    """

    quiz_type = data.get("type")
    category = data.get("category")
    classe = data.get("classe")
    anno = data.get("anno")
    difficulty = data.get("difficulty")
    llm_provider = data.get("llmProvider")

    # ✅ Validazione preliminare dei campi in ingresso (niente default)
    required_fields = [quiz_type, category, classe, anno, difficulty, llm_provider]
    if any(field is None or field == "" for field in required_fields):
        return {"status": 4, "data": {}}

    # Inizializza LLM + RAG
    try:
        llm = get_llm(llm_provider)
        rag_chain = build_rag_chain(llm)
    except Exception:
        #init fallito (es. chiave mancante, modello inesistente...)
        return {"status": 4, "data": {}}

    # Prompt super-esplicito (in ENG), output in ITA, JSON puro
    prompt = build_prompt(quiz_type, category, difficulty)

    # Query RAG con filtri
    rag_response = rag_chain.invoke({
        "query": prompt,
        "subject": category,
        "classe": classe,
        "anno": anno
    })

    print("RAG result raw:", rag_response.get("result", ""))

    # Nessuna fonte rilevante
    if not rag_response.get("source_documents"):
        return {"status": 2, "data": {}}

    # Parsing del JSON prodotto dall'LLM
    try:
        match = re.search(r"\{.*\}", rag_response["result"], re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")

        quiz_data = json.loads(match.group())
        if not isinstance(quiz_data, dict):
            raise ValueError("Not a dict")
    except Exception as e:
        print("JSON PARSE FAILED:", str(e))
        return {"status": 3, "data": {}}

    # Coerenza minima: forziamo la categoria a quella richiesta (evita drift del modello)
    # Se preferisci: invece di forzare potremmo fallire. Per ora forziamo.
    quiz_data["category"] = category

    # Validazione contro JSON Schema del tipo specifico
    # Se non rispetta lo schema → status 4 (richiesta non valida secondo i requisiti)
    is_valid, error = validate_quiz_data(quiz_data)
    if not is_valid:
        print("SCHEMA VALIDATION FAILED:", error)
        return {"status": 4, "data": {}}

    # Tutto ok
    return {"status": 1, "data": quiz_data}
