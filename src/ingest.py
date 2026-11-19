import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.genai import get_embedding

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():
    if not PDF_PATH:
        raise ValueError("Defina PDF_PATH no ambiente.")
    if not os.path.isfile(PDF_PATH):
        raise FileNotFoundError(f"PDF não encontrado: {PDF_PATH}")

    connection = os.getenv("DATABASE_URL")
    if not connection:
        raise ValueError("Defina DATABASE_URL no ambiente.")

    documents = PyPDFLoader(PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1_000,
        chunk_overlap=150,
    ).split_documents(documents)

    return PGVector.from_documents(
        documents=chunks,
        embedding=get_embedding(),
        connection=connection,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME", "pdf_documents"),
        use_jsonb=True,
    )


if __name__ == "__main__":
    ingest_pdf()