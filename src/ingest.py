from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# load from /documents directory


def load_documents(path="documents/"):
    loader = DirectoryLoader(
        "documents", glob="**/*.md", loader_cls=TextLoader, show_progress=True
    )

    docs = list(loader.lazy_load())

    docs = [d for d in docs if len(d.page_content) > 100]

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if len(c.page_content) > 50]

    return chunks
