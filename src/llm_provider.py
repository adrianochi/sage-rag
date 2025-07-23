import os
from langchain_groq import ChatGroq

def get_llm(provider_name: str = "groq"):
    if provider_name == "groq":
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.5
        )
    else:
        raise ValueError(f"Provider LLM non supportato: {provider_name}")
