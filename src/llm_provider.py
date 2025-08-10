import os

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic  # nuova import

def get_llm(provider):
    if provider == "groq":
        return ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-8b-8192",
            temperature=0.9
        )
    elif provider == "claude":
        return ChatAnthropic(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-haiku-20241022", 
            temperature=0.9
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
