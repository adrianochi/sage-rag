# tests/test_schema_validation.py
import os
import json
from jsonschema import validate, ValidationError

SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "..", "schemas")

def validate_quiz_data(data):
    schema_path = os.path.join(SCHEMA_DIR, f"{data['type']}_schema.json")
    if not os.path.exists(schema_path):
        return False, f"Schema non trovato per tipo: {data['type']}"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    try:
        validate(instance=data, schema=schema)
        return True, None
    except ValidationError as e:
        return False, e.message

# Percorso alla cartella con esempi di quiz da testare
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

def run_tests():
    if not os.path.exists(EXAMPLES_DIR):
        print(f"❌ La cartella {EXAMPLES_DIR} non esiste. Creala e metti file JSON di esempio.")
        return

    for filename in os.listdir(EXAMPLES_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(EXAMPLES_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(f"\n🔍 Testing {filename}...")
            is_valid, error = validate_quiz_data(data)

            if is_valid:
                print("✅ Valido secondo schema")
            else:
                print(f"❌ NON valido → {error}")

if __name__ == "__main__":
    run_tests()
