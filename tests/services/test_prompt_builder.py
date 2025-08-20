import re
from src.services.prompt_builder import build_prompt


def test_build_prompt_basic_content():
    quiz_type = "quiz"
    category = "matematica"
    difficulty = 5

    prompt = build_prompt(quiz_type, category, difficulty)

    # Deve contenere i parametri
    assert quiz_type in prompt
    assert category in prompt
    assert str(difficulty) in prompt

    # Deve contenere le specifiche degli schemi
    assert '"type": "quiz"' in prompt
    assert '"type": "matching"' in prompt
    assert '"type": "memory"' in prompt
    assert '"type": "sorting"' in prompt

    # Deve contenere le istruzioni sulla difficoltà
    assert "Difficulty 1-3" in prompt
    assert "Difficulty 4-6" in prompt
    assert "Difficulty 7-10" in prompt

    # Deve specificare che la risposta è in ITALIANO
    assert "ITALIANO" in prompt


def test_build_prompt_quotes_format():
    """
    Controlla che vengano usate solo doppie virgolette per JSON.
    """
    prompt = build_prompt("quiz", "storia", 3)

    # Nessuna chiave JSON con single quote
    assert not re.search(r"\{'.*':", prompt)

    # Deve avere chiavi con doppi apici
    assert re.search(r'\{"type": "quiz"', prompt)


def test_build_prompt_difficulty_edge_cases():
    """
    Verifica che anche difficoltà ai limiti (1 e 10) compaiano correttamente.
    """
    prompt_low = build_prompt("quiz", "geografia", 1)
    prompt_high = build_prompt("quiz", "geografia", 10)

    assert "1" in prompt_low
    assert "10" in prompt_high
