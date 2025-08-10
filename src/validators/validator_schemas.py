import json
import os
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
