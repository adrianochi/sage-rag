'''from flask import Flask, request, jsonify
import random

app = Flask(__name__)

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    data = request.json

    quiz_type = data.get("type")
    category = data.get("category")
    age_range = data.get("ageRange", [6, 10])
    difficulty = data.get("difficulty", 3)

    # Logica dummy per esempio
    if quiz_type == "quiz":
        question = "Qual è il pianeta più vicino al Sole?"
        options = ["Venere", "Terra", "Mercurio", "Marte"]
        answer = "Mercurio"
        response = {
            "id": random.randint(1, 1000),
            "type": "quiz",
            "category": category,
            "question": question,
            "ageRange": age_range,
            "difficulty": difficulty,
            "options": options,
            "answer": answer
        }

    elif quiz_type == "memory":
        response = {
            "id": random.randint(1, 1000),
            "type": "memory",
            "category": category,
            "question": "Abbina l'animale al suo verso",
            "ageRange": age_range,
            "difficulty": difficulty,
            "pairs": [
                {"front": "Cane", "back": "Bau"},
                {"front": "Gatto", "back": "Miao"}
            ]
        }

    elif quiz_type == "matching":
        response = {
            "id": random.randint(1, 1000),
            "type": "matching",
            "category": category,
            "question": "Abbina la parola al suo sinonimo",
            "ageRange": age_range,
            "difficulty": difficulty,
            "pairs": [
                {"left": "Grande", "right": "Enorme"},
                {"left": "Allegro", "right": "Felice"}
            ]
        }

    else:
        return jsonify({"error": "Tipo di quiz non supportato"}), 400

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
'''