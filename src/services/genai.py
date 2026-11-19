

import os
from services.gemini import get_gemini_chat_model, get_gemini_embeddings
from services.openai import get_openai_chat_model, get_openai_embeddings

genai = os.getenv("GENAI_PROVIDER", "openai") 

def get_embedding():
    if genai == "gemini":
        return get_gemini_embeddings({
            "model": os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"),
            "api_key": os.getenv("GOOGLE_API_KEY"),
        })

    return get_openai_embeddings({
        "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    })
    
def get_genai_chat_model():
    if genai == "gemini":
        return get_gemini_chat_model({
            "model": os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.5-flash-lite"),
            "api_key": os.getenv("GOOGLE_API_KEY"),
        })

    return get_openai_chat_model({
        "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        "api_key": os.getenv("OPENAI_API_KEY"),
    })
    
    