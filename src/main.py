from flask import Flask, request, jsonify
import random
from llm_provider import get_llm
from retriever_chain import build_rag_chain

app = Flask(__name__)
llm = get_llm("groq")
rag_chain = build_rag_chain(llm)

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    data = request.get_json()
    category = data.get("category", "scienze")
    difficulty = data.get("difficulty", 3)
    age_range = data.get("ageRange", [7, 10])
    quiz_type = data.get("type", "quiz")

    query = (
        f"Genera un quiz JSON di tipo '{quiz_type}' per bambini di età {age_range}, "
        f"sulla categoria '{category}', con difficoltà {difficulty}. "
        f"Il formato deve essere adatto al seguente schema JSON:\n"
        f"{{'type': ..., 'category': ..., 'question': ..., 'ageRange': ..., 'difficulty': ..., 'options': [...], 'answer': ...}}\n"
        f"Rispondi SOLO con il JSON valido, nulla prima o dopo."
    )

    result = rag_chain.invoke({"query": query})
    print("📚 Risposta RAG grezza:", result["result"])

    try:
        json_data = eval(result["result"])  # fallback rapido per test
        if isinstance(json_data, dict):
            json_data["id"] = random.randint(1, 1000)
            return jsonify(json_data)
        else:
            raise ValueError("Formato non valido")
    except Exception as e:
        return jsonify({"error": f"Errore parsing JSON: {str(e)}", "raw_result": result["result"]}), 400

if __name__ == "__main__":
    app.run(debug=True)
