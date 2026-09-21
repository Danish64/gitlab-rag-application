import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ingest import chunk_documents, load_documents

load_dotenv()

CHROMA_PATH = "./chroma_db"


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def create_vectorstore(force_rebuild=False):
    """Create vectorstore. Skips if already exists unless force_rebuild=True."""

    if os.path.exists(CHROMA_PATH) and not force_rebuild:
        print("Vectorstore already exists. Loading from disk...")
        print("(Use force_rebuild=True to re-embed)")
        return load_vectorstore()

    embeddings = get_embeddings()
    documents = load_documents()
    chunks = chunk_documents(documents)

    vector_store = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )

    print(f"Vectorstore created with {len(chunks)} chunks")

    return vector_store


def load_vectorstore():
    """Load existing vectorstore from disk."""
    embeddings = get_embeddings()
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore


def query_vectorstore(query, k=5):
    """Query the vectorstore."""
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results


if __name__ == "__main__":
    vectorstore = create_vectorstore()

    # Test query
    results = query_vectorstore("How do I request PTO?")
    print(f"\n--- TEST QUERY: 'How do I request PTO?' ---")
    for i, doc in enumerate(results):
        print(f"\n[{i + 1}] {doc.metadata['source']}")
        print(f"{doc.page_content[:200]}...")
