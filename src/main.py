from flask import Flask, request, jsonify, render_template_string
import json
import re
from llm_provider import get_llm
from retriever_chain import build_rag_chain

app = Flask(__name__)

# === HTML FORM (solo per testing da browser) ===
FORM_HTML = """<!DOCTYPE html><html><head><title>Quiz Generator</title></head><body>
<h1>Genera Quiz</h1>
<form method="post" action="/form">
    <label>Tipo quiz:</label><br>
    <input type="text" name="type" value="quiz"><br><br>
    <label>Categoria:</label><br>
    <input type="text" name="category" value="scienze"><br><br>
    <label>Classe (es. prim, sec1, sec2):</label><br>
    <input type="text" name="classe" value="prim"><br><br>
    <label>Anno (es. 1, 2, 3, 5):</label><br>
    <input type="number" name="anno" value="1"><br><br>
    <label>Difficoltà (1-10):</label><br>
    <input type="number" name="difficulty" value="3"><br><br>
    <label>LLM Provider:</label><br>
    <input type="text" name="llmProvider" value="groq"><br><br>
    <button type="submit">Invia</button>
</form>
{% if result %}
    <h2>Risultato:</h2>
    <textarea rows="20" cols="80">{{ result }}</textarea>
{% endif %}
</body></html>
"""

@app.route("/form", methods=["GET", "POST"])
def quiz_form():
    if request.method == "POST":

        # Se JSON (es. da Postman o curl)
        if request.is_json:
            data = request.get_json()
            data["llmProvider"] = data.get("llmProvider", "groq")

            quiz = generate_quiz_from_data(data)
            return jsonify(quiz)

        # Se richiesta da browser (form HTML)
        try:
            payload = {
                "type": request.form["type"],
                "category": request.form["category"],
                "classe": request.form["classe"],
                "anno": int(request.form["anno"]),
                "difficulty": int(request.form["difficulty"]),
                "llmProvider": request.form["llmProvider"]  # obbligatorio da form
            }
        except (KeyError, ValueError):
            return render_template_string(FORM_HTML, result=json.dumps({"status": 4, "data": {}}, indent=4))

        quiz = generate_quiz_from_data(payload)

        if "application/json" in request.headers.get("Accept", ""):
            return jsonify(quiz)

        return render_template_string(FORM_HTML, result=json.dumps(quiz, indent=4, ensure_ascii=False))

    return render_template_string(FORM_HTML)



# === GENERATORE ===
def generate_quiz_from_data(data):
    quiz_type = data.get("type")
    category = data.get("category")
    classe = data.get("classe")
    anno = data.get("anno")
    difficulty = data.get("difficulty")
    llm_provider = data.get("llmProvider")

    required_fields = [quiz_type, category, classe, anno, difficulty, llm_provider]
    if any(field is None or field == "" for field in required_fields):
        return {"status": 4, "data": {}}

    # Istanzia LLM e RAG
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

    print("📚 RAG result raw:", rag_response["result"])

    if not rag_response["source_documents"]:
        return {"status": 2, "data": {}}

    try:
        match = re.search(r"\{.*\}", rag_response["result"], re.DOTALL)
        if not match:
            raise ValueError("No JSON object found")

        quiz_data = json.loads(match.group())
        if not isinstance(quiz_data, dict):
            raise ValueError("Not a dict")

        return {
            "status": 1,
            "data": quiz_data
        }

    except Exception as e:
        print("❌ JSON PARSE FAILED:", str(e))
        return {
            "status": 3,
            "data": {}
        }

# === PROMPT BUILDER ===
def build_prompt(quiz_type, category, difficulty):
    return (
        f"You are a quiz generator for children. Generate a single Italian quiz of type '{quiz_type}' "
        f"about the category '{category}', suitable for the given level. The difficulty value must be the number {difficulty}. "
        f"Use only the provided sources. The quiz must follow strictly one of the following JSON schemas:\n\n"

        f"▶ For 'quiz':\n"
        f"{{'type': 'quiz', 'category': ..., 'question': ..., 'difficulty': ..., 'options': [...], 'answer': ...}}\n\n"

        f"▶ For 'matching':\n"
        f"{{'type': 'matching', 'category': ..., 'question': ..., 'difficulty': ..., 'pairs': [{{'left': ..., 'right': ...}}]}}\n\n"

        f"▶ For 'memory':\n"
        f"{{'type': 'memory', 'category': ..., 'question': ..., 'difficulty': ..., 'pairs': [{{'front': ..., 'back': ...}}]}}\n\n"

        f"▶ For 'sorting':\n"
        f"{{'type': 'sorting', 'category': ..., 'question': ..., 'difficulty': ..., 'items': [...], 'solution': [...]}}\n\n"

        f"Only output the raw JSON object. No extra text. Only return a single valid JSON object. Use double quotes around all keys and string values. Do not use single quotes. Always respond in **ITALIANO**.\n"
        f"Return an empty JSON (i.e. '{{}}') if the context is insufficient."
    )

# === RUN ===
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=False)
