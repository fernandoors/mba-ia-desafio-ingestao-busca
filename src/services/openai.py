import getpass
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
def get_openai_embeddings(args):
    """
    Returns an instance of OpenAIEmbeddings with the specified model.
    """
    model = args.get("model", "text-embedding-3-small")
    api_key = args.get("api_key", None)
    
    if api_key is None:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
    
    return OpenAIEmbeddings(
        model=model,
        openai_api_key=api_key,
    )

def get_openai_chat_model(args):
    """
    Returns an instance of OpenAI with the specified model.
    """
    model = args.get("model", "gpt-6-luna")
    api_key = args.get("api_key", None)
    
    if api_key is None:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
    
    return ChatOpenAI(
        model_name=model,
        openai_api_key=api_key,
    )