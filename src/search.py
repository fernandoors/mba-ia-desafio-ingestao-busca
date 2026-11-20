import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector

from services.genai import get_embedding, get_genai_chat_model

load_dotenv()

FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(question: str) -> str:
    if not question.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    connection = os.getenv("DATABASE_URL")
    if not connection:
        raise ValueError("Defina DATABASE_URL no ambiente.")

    vector_store = PGVector(
        embeddings=get_embedding(),
        connection=connection,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME") or "pdf_documents",
        use_jsonb=True,
    )
    documents = vector_store.similarity_search(question, k=10)
    if not documents:
        return FALLBACK_ANSWER

    contexto = "\n\n".join(document.page_content for document in documents)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
    model = get_genai_chat_model()
    return (model | StrOutputParser()).invoke(prompt).strip()