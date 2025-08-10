# tests/test_schema_validation.py
import os
import sys
import json

def _add_src_to_syspath():
    # Trova la root progetto assumendo che questa file sia in <root>/tests/...
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, ".."))
    src_dir = os.path.join(project_root, "src")
    if not os.path.isdir(src_dir):
        # fallback: se i test sono già dentro src/tests
        alt_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
        alt_src = os.path.join(alt_root, "src")
        if os.path.isdir(alt_src):
            src_dir = alt_src

    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return src_dir

src_dir = _add_src_to_syspath()

try:
    from validators.validator_schemas import validate_quiz_data
except ModuleNotFoundError as e:
    print("❌ Import fallito:", e)
    print("ℹ️  Controlla queste cose:")
    print(f"   - Esiste la cartella: {os.path.join(src_dir, 'validators')}")
    print(f"   - Esiste il file: {os.path.join(src_dir, 'validators', 'validator_schemas.py')}")
    print("   - La cartella si chiama esattamente 'validators' (non 'vaildators')")
    print("   - In alternativa lancia con: PYTHONPATH=src python3 tests/test_schema_validation.py")
    raise

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
