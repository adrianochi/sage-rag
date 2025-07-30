from flask import Flask, request, jsonify, render_template_string
import json
import random
from llm_provider import get_llm
from retriever_chain import build_rag_chain

app = Flask(__name__)
llm = get_llm("groq")
rag_chain = build_rag_chain(llm)

def generate_quiz_from_data(data):
    import re

    category = data.get("category", "scienze")
    difficulty = data.get("difficulty", 3)
    age_range = data.get("ageRange", [7, 10])
    quiz_type = data.get("type", "quiz")

    subject = category  # puoi anche mappare se necessario

    level = map_age_to_level(age_range)
    print(level)

    query = (
        f"Genera un quiz JSON di tipo '{quiz_type}'"
        f"sulla categoria '{category}', con difficoltà indefinita. "
        f"Il formato deve essere adatto al seguente schema JSON:\n"
        f"{{'type': ..., 'category': ..., 'question': ..., 'ageRange': ..., 'difficulty': ..., 'options': [...], 'answer': ...}}\n"
        f"Usa solo le informazioni fornite nel contesto. "
        f"Se non trovi nulla di rilevante, non generare nessun quiz e risondi solo con un JSON VUOTO.\n"
        f"Rispondi SOLO con il JSON valido, nulla prima o dopo."
        f"Rispondi SEMPRE in lingua ITALIANA"
    )

    # 🔍 Esegui RAG
    result = rag_chain.invoke({
    "query": query,
    "subject": subject,
    "level": level
    })


    # 🧾 Estrai i documenti usati come contesto
    docs = result.get("source_documents", [])

    print("📚 Risposta RAG grezza:", result["result"])

    # 🧪 Parse JSON dalla risposta
    try:
        match = re.search(r"\{.*\}", result["result"], re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in output")

        json_data = json.loads(match.group())
        if isinstance(json_data, dict):
            json_data["id"] = random.randint(1, 1000)
            return json_data
        else:
            raise ValueError("Formato non valido (non è un dizionario)")
    except Exception as e:
        return {"error": f"Errore parsing JSON: {str(e)}", "raw_result": result["result"]}



@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    if request.is_json:
        data = request.get_json()
    else:
        data = {
            "category": request.form.get("category", "scienze"),
            "difficulty": int(request.form.get("difficulty", 3)),
            "ageRange": [int(request.form.get("age_min", 7)), int(request.form.get("age_max", 10))],
            "type": request.form.get("type", "quiz")
        }
    result = generate_quiz_from_data(data)
    return jsonify(result)


# HTML template semplice
FORM_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Quiz Generator</title>
</head>
<body>
    <h1>Genera Quiz</h1>
    <form method="post" action="/form">
        <label>Tipo quiz:</label><br>
        <input type="text" name="type" value="quiz"><br><br>
        <label>Categoria:</label><br>
        <input type="text" name="category" value="scienze"><br><br>
        <label>Range età (min,max):</label><br>
        <input type="text" name="ageRange" value="7,10"><br><br>
        <label>Difficoltà (1-10):</label><br>
        <input type="number" name="difficulty" value="3"><br><br>
        <button type="submit">Invia</button>
    </form>

    {% if result %}
        <h2>Risultato:</h2>
        <textarea rows="20" cols="80">{{ result }}</textarea>
    {% endif %}
</body>
</html>
"""

@app.route("/form", methods=["GET", "POST"])
def quiz_form():
    if request.method == "POST":
        quiz_type = request.form.get("type", "quiz")
        category = request.form.get("category", "scienze")
        age_range_str = request.form.get("ageRange", "7,10")
        difficulty = int(request.form.get("difficulty", 3))
        age_range = [int(x.strip()) for x in age_range_str.split(",")]

        payload = {
            "type": quiz_type,
            "category": category,
            "ageRange": age_range,
            "difficulty": difficulty
        }

        quiz = generate_quiz_from_data(payload)
        return render_template_string(FORM_HTML, result=json.dumps(quiz, indent=4, ensure_ascii=False))

    return render_template_string(FORM_HTML)

def map_age_to_level(age_range):
    min_age = age_range[0]
    if min_age <= 10:
        return "elementari"
    elif min_age <= 13:
        return "medie"
    else:
        return "superiori"

if __name__ == "__main__":
    app.run(debug=True)