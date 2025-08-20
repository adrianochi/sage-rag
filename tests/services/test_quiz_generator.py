import pytest
from src.services import quiz_generator


class DummyRagChain:
    def __init__(self, response):
        self._response = response

    def invoke(self, _):
        return self._response


def make_base_data():
    return {
        "type": "quiz",
        "category": "matematica",
        "classe": "3",
        "anno": "2025",
        "difficulty": 5,
        "llmProvider": "groq",
    }


def test_missing_fields_returns_status_4():
    data = make_base_data()
    data["category"] = ""  # campo mancante
    result = quiz_generator.generate_quiz_from_data(data)
    assert result["status"] == 4
    assert result["data"] == {}


def test_llm_init_failure(monkeypatch):
    data = make_base_data()

    monkeypatch.setattr(quiz_generator, "get_llm", lambda provider: (_ for _ in ()).throw(Exception("fail")))
    result = quiz_generator.generate_quiz_from_data(data)

    assert result["status"] == 4
    assert result["data"] == {}


def test_no_sources_found(monkeypatch):
    data = make_base_data()

    monkeypatch.setattr(quiz_generator, "get_llm", lambda provider: object())
    monkeypatch.setattr(quiz_generator, "build_rag_chain", lambda llm: DummyRagChain({"result": "{}", "source_documents": []}))
    monkeypatch.setattr(quiz_generator, "build_prompt", lambda *_: "prompt")

    result = quiz_generator.generate_quiz_from_data(data)
    assert result["status"] == 2
    assert result["data"] == {}


def test_invalid_json_response(monkeypatch):
    data = make_base_data()

    monkeypatch.setattr(quiz_generator, "get_llm", lambda provider: object())
    monkeypatch.setattr(quiz_generator, "build_rag_chain", lambda llm: DummyRagChain({"result": "niente json", "source_documents": ["doc"]}))
    monkeypatch.setattr(quiz_generator, "build_prompt", lambda *_: "prompt")

    result = quiz_generator.generate_quiz_from_data(data)
    assert result["status"] == 3
    assert result["data"] == {}


def test_schema_validation_failure(monkeypatch):
    data = make_base_data()

    monkeypatch.setattr(quiz_generator, "get_llm", lambda provider: object())
    monkeypatch.setattr(quiz_generator, "build_rag_chain", lambda llm: DummyRagChain({"result": '{"type": "quiz", "category": "wrong"}', "source_documents": ["doc"]}))
    monkeypatch.setattr(quiz_generator, "build_prompt", lambda *_: "prompt")
    monkeypatch.setattr(quiz_generator, "validate_quiz_data", lambda _: (False, "schema error"))

    result = quiz_generator.generate_quiz_from_data(data)
    assert result["status"] == 4
    assert result["data"] == {}


def test_successful_quiz(monkeypatch):
    data = make_base_data()

    json_response = '{"type": "quiz", "category": "whatever", "question": "Q?", "difficulty": 5, "options": ["a","b"], "answer": "a"}'

    monkeypatch.setattr(quiz_generator, "get_llm", lambda provider: object())
    monkeypatch.setattr(quiz_generator, "build_rag_chain", lambda llm: DummyRagChain({"result": json_response, "source_documents": ["doc"]}))
    monkeypatch.setattr(quiz_generator, "build_prompt", lambda *_: "prompt")
    monkeypatch.setattr(quiz_generator, "validate_quiz_data", lambda _: (True, None))

    result = quiz_generator.generate_quiz_from_data(data)

    assert result["status"] == 1
    assert result["data"]["type"] == "quiz"
    assert result["data"]["category"] == data["category"]  # viene forzata
    assert "question" in result["data"]
