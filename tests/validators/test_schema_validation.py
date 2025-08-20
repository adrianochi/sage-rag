import os
import json
import tempfile
import pytest

from src.validators import validator_schemas


@pytest.fixture
def schema_dir(tmp_path, monkeypatch):
    # crea uno schema temporaneo valido
    schema_content = {
        "type": "object",
        "properties": {
            "type": {"const": "quiz"},
            "question": {"type": "string"},
            "answer": {"type": "string"}
        },
        "required": ["type", "question", "answer"]
    }

    schema_file = tmp_path / "quiz_schema.json"
    schema_file.write_text(json.dumps(schema_content), encoding="utf-8")

    # patcha SCHEMA_DIR per puntare alla cartella temporanea
    monkeypatch.setattr(validator_schemas, "SCHEMA_DIR", str(tmp_path))
    return tmp_path


def test_validate_quiz_data_valid(schema_dir):
    data = {
        "type": "quiz",
        "question": "Quanto fa 2+2?",
        "answer": "4"
    }
    valid, error = validator_schemas.validate_quiz_data(data)
    assert valid is True
    assert error is None


def test_validate_quiz_data_missing_required(schema_dir):
    data = {
        "type": "quiz",
        "question": "Quanto fa 2+2?"
        # manca "answer"
    }
    valid, error = validator_schemas.validate_quiz_data(data)
    assert valid is False
    assert "answer" in error  # il messaggio di errore contiene il campo mancante


def test_validate_quiz_data_schema_not_found(monkeypatch):
    # forza SCHEMA_DIR a cartella vuota
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(validator_schemas, "SCHEMA_DIR", tmpdir)
        data = {
            "type": "quiz",
            "question": "test",
            "answer": "ok"
        }
        valid, error = validator_schemas.validate_quiz_data(data)
        assert valid is False
        assert "Schema non trovato" in error
