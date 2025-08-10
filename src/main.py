from flask import Flask, request, jsonify, render_template
from services.quiz_generator import generate_quiz_from_data
import json

app = Flask(__name__)

@app.route("/form", methods=["GET", "POST"])
def quiz_form():
    if request.method == "POST":
        if request.is_json:
            # Richiesta JSON (es. Postman)
            data = request.get_json()
            data["llmProvider"] = data.get("llmProvider", "groq")
            quiz = generate_quiz_from_data(data)
            return jsonify(quiz)

        try:
            # Richiesta da browser (form)
            payload = {
                "type": request.form["type"],
                "category": request.form["category"],
                "classe": request.form["classe"],
                "anno": int(request.form["anno"]),
                "difficulty": int(request.form["difficulty"]),
                "llmProvider": request.form["llmProvider"]
            }
        except (KeyError, ValueError):
            error_json = json.dumps({"status": 4, "data": {}}, indent=4, ensure_ascii=False)
            return render_template("form.html", result=error_json)

        quiz = generate_quiz_from_data(payload)
        # Qui lo trasformo in stringa JSON formattata
        quiz_json = json.dumps(quiz, indent=4, ensure_ascii=False)
        return render_template("form.html", result=quiz_json)

    # GET → pagina vuota
    return render_template("form.html", result=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=False)
