import pytest
from src.services import llm_provider


class DummyGroq:
    def __init__(self, groq_api_key, model, temperature):
        self.groq_api_key = groq_api_key
        self.model = model
        self.temperature = temperature


class DummyAnthropic:
    def __init__(self, anthropic_api_key, model, temperature):
        self.anthropic_api_key = anthropic_api_key
        self.model = model
        self.temperature = temperature


def test_get_llm_groq(monkeypatch):
    monkeypatch.setattr(llm_provider, "ChatGroq", DummyGroq)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")

    llm = llm_provider.get_llm("groq")
    assert isinstance(llm, DummyGroq)
    assert llm.groq_api_key == "fake-groq"
    assert llm.model == "llama3-8b-8192"
    assert llm.temperature == 0.9


def test_get_llm_claude(monkeypatch):
    monkeypatch.setattr(llm_provider, "ChatAnthropic", DummyAnthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")

    llm = llm_provider.get_llm("claude")
    assert isinstance(llm, DummyAnthropic)
    assert llm.anthropic_api_key == "fake-anthropic"
    assert "claude" in llm.model
    assert llm.temperature == 0.9


def test_get_llm_invalid_provider():
    with pytest.raises(ValueError) as excinfo:
        llm_provider.get_llm("openai")
    assert "Unknown LLM provider" in str(excinfo.value)
