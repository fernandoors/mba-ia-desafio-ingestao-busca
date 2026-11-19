import getpass
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def get_gemini_embeddings(args):
    """
    Returns an instance of GoogleGenerativeAIEmbeddings with the specified model.
    """
    model = args.get("model", "gemini-embedding-001")
    api_key = args.get("api_key", None)
    
    if api_key is None:
        api_key = getpass.getpass("Enter your Google Generative AI API key: ")
    
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
    )

def get_gemini_chat_model(args):
    """
    Returns an instance of ChatGoogleGenerativeAI with the specified model.
    """
    model = args.get("model", "gemini-3.5-flash-lite")
    api_key = args.get("api_key", None)
    
    if api_key is None:
        api_key = getpass.getpass("Enter your Google Generative AI API key: ")
    
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
    )